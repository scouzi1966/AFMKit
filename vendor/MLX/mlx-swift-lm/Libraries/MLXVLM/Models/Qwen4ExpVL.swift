//
//  Qwen4ExpVL.swift
//  mlx-swift-lm
//
//  Vision-language wrapper for Qwen/Qwen3.8-Flash-Next. The checkpoint uses
//  the Qwen3-VL vision tower and processor with the Qwen4 experimental trunk.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public struct Qwen4ExpVLConfiguration: Decodable, Sendable {
    let modelType: String
    let textConfig: Qwen4ExpTextConfiguration
    let visionConfig: Qwen3VLConfiguration.VisionConfiguration
    let ngramTable: Qwen4ExpNGramTableConfiguration?
    private let configuredImageTokenID: Int?
    private let configuredVideoTokenID: Int?
    private let configuredVisionStartTokenID: Int?

    var imageTokenID: Int { configuredImageTokenID ?? 248_056 }
    var videoTokenID: Int { configuredVideoTokenID ?? 248_057 }
    var visionStartTokenID: Int { configuredVisionStartTokenID ?? 248_053 }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case ngramTable = "ngram_table"
        case configuredImageTokenID = "image_token_id"
        case configuredVideoTokenID = "video_token_id"
        case configuredVisionStartTokenID = "vision_start_token_id"
    }
}

public final class Qwen4ExpVL: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Qwen4ExpModel

    private let config: Qwen4ExpVLConfiguration

    public init(_ config: Qwen4ExpVLConfiguration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfig)
        _languageModel.wrappedValue = Qwen4ExpModel(
            Qwen4ExpConfiguration(
                modelType: config.modelType,
                textConfig: config.textConfig,
                ngramTable: config.ngramTable))
    }

    public var vocabularySize: Int { languageModel.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.loraLayers }
    public var consumesHostTokenIDs: Bool { languageModel.consumesHostTokenIDs }

    public func configureMappedNGramTable(url: URL) throws {
        try languageModel.configureMappedNGramTable(url: url)
    }

    public func applyMappedNGramResidency(
        _ policy: QwenNGramResidencyPolicy
    ) throws {
        try languageModel.applyMappedNGramResidency(policy)
    }

    /// Loads the text trunk's in-checkpoint native MTP head while preserving
    /// this multimodal container. Text-only requests may use the returned
    /// generator; requests carrying media continue through ordinary VLM
    /// generation.
    public func makeEmbeddedMTPGenerator(
        modelDirectory: URL,
        depth: Int = 3,
        groupSize: Int = 64,
        bits: Int = 4,
        mode: QuantizationMode = .affine,
        verificationPolicy: MTPVerificationPolicy = .strictSingletonEquivalent
    ) throws -> Qwen4ExpMTPGenerator {
        let head = try languageModel.loadEmbeddedMTPHead(
            modelDirectory: modelDirectory,
            groupSize: groupSize,
            bits: bits,
            mode: mode)
        return Qwen4ExpMTPGenerator(
            model: languageModel,
            head: head,
            depth: depth,
            verificationPolicy: verificationPolicy)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    private func frames(images: [THW]?, videos: [THW]?) -> [THW] {
        (images ?? []) + (videos ?? [])
    }

    private func splitIndices(_ sizes: [Int]) -> [Int] {
        var total = 0
        return sizes.dropLast().map {
            total += $0
            return total
        }
    }

    private func merge(
        features: MLXArray, embeddings: MLXArray, inputIDs: MLXArray
    ) throws -> MLXArray {
        let special = (inputIDs .== MLXArray(config.imageTokenID))
            .|| (inputIDs .== MLXArray(config.videoTokenID))
        let expanded = broadcast(
            expandedDimensions(special, axis: -1), to: embeddings.shape)
        guard expanded.sum().item(Int.self) == features.size else {
            throw VLMError.processing(
                "Visual feature/token mismatch: \(features.dim(0)) features for "
                    + "\(special.sum().item(Int.self)) media tokens")
        }

        let indices = expanded.flattened().asArray(Bool.self).enumerated().compactMap {
            $0.element ? UInt32($0.offset) : nil
        }
        var result = embeddings.flattened()
        result[MLXArray(indices)] = features.flattened()
        return result.reshaped(embeddings.shape)
    }

    private func positionIDs(
        inputIDs: MLXArray,
        imageFrames: [THW]?,
        videoFrames: [THW]?
    ) -> (positions: MLXArray, deltas: MLXArray) {
        Qwen3VLLanguage.getRopeIndex(
            inputIds: inputIDs,
            imageGridTHW: imageFrames,
            videoGridTHW: videoFrames,
            spatialMergeSize: config.visionConfig.spatialMergeSize,
            imageTokenId: config.imageTokenID,
            videoTokenId: config.videoTokenID,
            visionStartTokenId: config.visionStartTokenID,
            attentionMask: nil)
    }

    public func prepare(
        _ input: LMInput,
        cache: [KVCache],
        windowSize _: Int?
    ) throws -> PrepareResult {
        let inputIDs = input.text.tokens
        let imageFrames = input.image?.frames
        let videoFrames = input.video?.frames
        var embeddings: MLXArray?
        var pixels = [MLXArray]()
        let dtype = visionModel.patchEmbed.proj.weight.dtype

        if let image = input.image { pixels.append(image.pixels.asType(dtype)) }
        if let video = input.video { pixels.append(video.pixels.asType(dtype)) }

        let allFrames = frames(images: imageFrames, videos: videoFrames)
        if !pixels.isEmpty, !allFrames.isEmpty {
            let textEmbeddings = languageModel.embedTokens(inputIDs)
            let (visionHidden, _) = visionModel(concatenated(pixels), gridTHW: allFrames)
            let mergeSize = config.visionConfig.spatialMergeSize
            let sizes = allFrames.map { $0.product / (mergeSize * mergeSize) }
            let slices = visionHidden.split(indices: splitIndices(sizes))
            let features = concatenated(slices).asType(textEmbeddings.dtype)
            embeddings = try merge(
                features: features, embeddings: textEmbeddings, inputIDs: inputIDs)
        }

        let positionState = positionIDs(
            inputIDs: inputIDs, imageFrames: imageFrames, videoFrames: videoFrames)
        let logits = languageModel.forward(
            inputIDs: inputIDs, inputEmbeddings: embeddings,
            positionIDs: positionState.positions, cache: cache)
        return .logits(LMOutput(
            logits: logits,
            state: .init(positionDeltas: positionState.deltas)))
    }

    public func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        callAsFunction(
            input, cache: cache, state: state, hostTokenIDs: nil)
    }

    public func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?,
        hostTokenIDs: [Int]?
    ) -> LMOutput {
        let inputs = input.tokens
        var positions: MLXArray?
        if let cache, let ropeDeltas = state?.positionDeltas {
            let offset = cache.map(\.offset).max() ?? 0
            let batch = inputs.dim(0)
            let length = inputs.dim(1)
            var base = MLXArray(0 ..< length).asType(.int32)
            base = broadcast(base[.newAxis, 0...], to: [batch, length])
            var delta = MLXArray(offset).asType(.int32) + ropeDeltas.asType(.int32)
            if delta.dim(0) == 1 && batch > 1 {
                delta = repeated(delta, count: batch, axis: 0)
            }
            positions = broadcast(
                (base + delta)[.newAxis, 0..., 0...], to: [3, batch, length])
        }
        return LMOutput(
            logits: languageModel.forward(
                inputIDs: inputs, positionIDs: positions, cache: cache,
                hostTokenIDs: hostTokenIDs),
            state: state)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(
            LMInput.Text(tokens: inputs), cache: cache, state: nil).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in languageModel.sanitize(weights: weights) {
            result["language_model.\(key)"] = value
        }

        let visionWeights: [String: MLXArray] = Dictionary(
            uniqueKeysWithValues: weights.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("vision_tower.") else { return nil }
            return (String(key.dropFirst("vision_tower.".count)), value)
        })
        for (key, value) in visionModel.sanitize(weights: visionWeights) {
            result["vision_tower.\(key)"] = value
        }
        return result
    }
}
