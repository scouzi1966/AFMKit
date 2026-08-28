//
//  GLM5NextVL.swift
//  mlx-swift-lm
//
//  Self-contained MLX Swift vision-language implementation for GLM-5.3 Flash.
//  The text trunk lives in MLXLLM/GLM5Next.swift; this file owns the published
//  glm5_next_vision tower, nested processor configuration, and media embedding.
//


import CoreImage
import Foundation
import MLX
import MLXFast
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Processor

public struct GLM5NextProcessorConfiguration: Codable, Sendable {
    public struct MediaConfiguration: Codable, Sendable {
        public let imageMean: [CGFloat]
        public let imageStd: [CGFloat]
        public let mergeSize: Int
        public let patchSize: Int
        public let temporalPatchSize: Int
        public let minImageTokens: Int
        public let maxImageTokens: Int
        public let fps: Double?

        enum CodingKeys: String, CodingKey {
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case mergeSize = "merge_size"
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case minImageTokens = "min_image_tokens"
            case maxImageTokens = "max_image_tokens"
            case fps
        }

        var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
            (imageMean[0], imageMean[1], imageMean[2])
        }

        var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
            (imageStd[0], imageStd[1], imageStd[2])
        }

        public var minPixels: Int {
            minImageTokens * patchSize * patchSize * mergeSize * mergeSize
        }

        public var maxPixels: Int {
            maxImageTokens * patchSize * patchSize * mergeSize * mergeSize
        }
    }

    public let imageProcessor: MediaConfiguration
    public let videoProcessor: MediaConfiguration?
    public let processorClass: String

    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
        case videoProcessor = "video_processor"
        case processorClass = "processor_class"
    }
}

public struct GLM5NextProcessor: UserInputProcessor {
    private let config: GLM5NextProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: GLM5NextProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private func preprocess(
        images: [CIImage],
        configuration: GLM5NextProcessorConfiguration.MediaConfiguration,
        processing: UserInput.Processing?
    ) throws -> (MLXArray, THW) {
        let processed = images.map { MediaProcessing.apply($0, processing: processing) }
        guard let first = processed.first else {
            throw VLMError.imageProcessingFailure("No image provided")
        }

        let extent = first.extent.size
        let (height, width) = try QwenVL.targetSize(
            height: Int(extent.height),
            width: Int(extent.width),
            factor: configuration.patchSize * configuration.mergeSize,
            minPixels: configuration.minPixels,
            maxPixels: configuration.maxPixels)
        let target = CGSize(width: width, height: height)
        let normalized = processed
            .map { MediaProcessing.resampleBicubic($0, to: target) }
            .map {
                MediaProcessing.normalize(
                    $0,
                    mean: configuration.imageMeanTuple,
                    std: configuration.imageStdTuple)
            }
            .map { MediaProcessing.asMLXArray($0) }

        return try QwenVL.patchify(
            images: normalized,
            mergeSize: configuration.mergeSize,
            patchSize: configuration.patchSize,
            temporalPatchSize: configuration.temporalPatchSize)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Qwen3VLMessageGenerator().generate(from: input)
        let template: ChatTemplateArgument?
        if let override = input.additionalContext?["chatTemplateOverride"] as? String {
            template = .literal(override)
        } else {
            template = nil
        }
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            chatTemplate: template,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: input.tools,
            additionalContext: input.additionalContext)

        if input.images.isEmpty, input.videos.isEmpty {
            let tokens = MLXArray(promptTokens).expandedDimensions(axis: 0)
            return LMInput(
                text: .init(tokens: tokens, mask: ones(like: tokens).asType(.int8)))
        }

        let imageConfiguration = config.imageProcessor
        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let images = try input.images.map {
                try preprocess(
                    images: [$0.asCIImage()],
                    configuration: imageConfiguration,
                    processing: input.processing)
            }
            processedImage = .init(
                pixels: concatenated(images.map(\.0)),
                frames: images.map(\.1))
            promptTokens = try QwenVL.replacePaddingTokens(
                in: promptTokens,
                frames: images.map(\.1),
                paddingToken: "<|image|>",
                mergeSize: imageConfiguration.mergeSize,
                tokenizer: tokenizer)
        }

        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            guard let videoConfiguration = config.videoProcessor else {
                throw VLMError.videoNotSupported("glm5_next")
            }
            var videos = [(MLXArray, THW)]()
            for video in input.videos {
                var resizedSize: CGSize = .zero
                let sequence = try await MediaProcessing.asProcessedSequence(
                    video,
                    targetFPS: { _ in videoConfiguration.fps ?? 2 }
                ) { frame in
                    let processed = MediaProcessing.apply(
                        frame.frame, processing: input.processing)
                    if resizedSize == .zero {
                        let extent = processed.extent.size
                        let (height, width) = try QwenVL.targetSize(
                            height: Int(extent.height),
                            width: Int(extent.width),
                            factor: videoConfiguration.patchSize
                                * videoConfiguration.mergeSize,
                            minPixels: videoConfiguration.minPixels,
                            maxPixels: videoConfiguration.maxPixels)
                        resizedSize = CGSize(width: width, height: height)
                    }
                    let resized = MediaProcessing.resampleBicubic(
                        processed, to: resizedSize)
                    let normalized = MediaProcessing.normalize(
                        resized,
                        mean: videoConfiguration.imageMeanTuple,
                        std: videoConfiguration.imageStdTuple)
                    return VideoFrame(
                        frame: normalized, timeStamp: frame.timeStamp)
                }
                videos.append(
                    try QwenVL.patchify(
                        images: sequence.frames,
                        mergeSize: videoConfiguration.mergeSize,
                        patchSize: videoConfiguration.patchSize,
                        temporalPatchSize: videoConfiguration.temporalPatchSize))
            }
            processedVideo = .init(
                pixels: concatenated(videos.map(\.0)),
                frames: videos.map(\.1))
            promptTokens = try QwenVL.replacePaddingTokens(
                in: promptTokens,
                frames: videos.map(\.1),
                paddingToken: "<|video|>",
                mergeSize: videoConfiguration.mergeSize,
                tokenizer: tokenizer)
        }

        let tokens = MLXArray(promptTokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: tokens, mask: ones(like: tokens).asType(.int8)),
            image: processedImage,
            video: processedVideo)
    }
}

// MARK: - Configuration

public struct GLM5NextVLConfiguration: Decodable, Sendable {
    let modelType: String
    let languageConfiguration: GLM5NextConfiguration
    let visionConfiguration: GLM5NextVisionConfiguration
    let imageTokenID: Int
    let videoTokenID: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case visionConfiguration = "vision_config"
        case imageTokenID = "image_token_id"
        case videoTokenID = "video_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "glm5_next"
        languageConfiguration = try GLM5NextConfiguration(from: decoder)
        visionConfiguration = try container.decode(
            GLM5NextVisionConfiguration.self, forKey: .visionConfiguration)
        imageTokenID = try container.decode(Int.self, forKey: .imageTokenID)
        videoTokenID = try container.decode(Int.self, forKey: .videoTokenID)
    }
}

public struct GLM5NextVisionConfiguration: Decodable, Sendable {
    let modelType: String
    let depth: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let outHiddenSize: Int
    let numHeads: Int
    let patchSize: Int
    let temporalPatchSize: Int
    let spatialMergeSize: Int
    let projectionIntermediateSize: Int
    let rmsNormEps: Float
    let attentionBias: Bool
    let hiddenAct: String
    let swigluLimit: Float
    let inChannels: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case depth
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case outHiddenSize = "out_hidden_size"
        case numHeads = "num_heads"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case projectionIntermediateSize = "projection_intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case hiddenAct = "hidden_act"
        case swigluLimit = "swiglu_limit"
        case inChannels = "in_channels"
    }
}

// MARK: - Vision tower

enum GLM5NextVision {
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, 0 ..< half]], axis: -1)
    }

    static func applyRotary(
        queries: MLXArray,
        keys: MLXArray,
        cosine: MLXArray,
        sine: MLXArray
    ) -> (MLXArray, MLXArray) {
        let qType = queries.dtype
        let kType = keys.dtype
        let cosValues = expandedDimensions(cosine, axis: -2).asType(.float32)
        let sinValues = expandedDimensions(sine, axis: -2).asType(.float32)
        let q = queries.asType(.float32)
        let k = keys.asType(.float32)
        return (
            (q * cosValues + rotateHalf(q) * sinValues).asType(qType),
            (k * cosValues + rotateHalf(k) * sinValues).asType(kType))
    }

    final class RotaryEmbedding {
        let dimension: Int
        let theta: Float

        init(dimension: Int, theta: Float = 10_000) {
            self.dimension = dimension
            self.theta = theta
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            let powers = MLXArray(stride(from: 0, to: dimension, by: 2))
                .asType(.float32) / Float(dimension)
            let inverse = 1 / pow(MLXArray(theta), powers)
            return outer(MLXArray(0 ..< sequenceLength).asType(.float32), inverse)
        }
    }

    final class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo(key: "proj") var projection: Conv3d
        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let hiddenSize: Int

        init(_ config: GLM5NextVisionConfiguration) {
            patchSize = config.patchSize
            temporalPatchSize = config.temporalPatchSize
            inChannels = config.inChannels
            hiddenSize = config.hiddenSize
            let kernel = IntOrTriple([temporalPatchSize, patchSize, patchSize])
            _projection.wrappedValue = Conv3d(
                inputChannels: inChannels,
                outputChannels: hiddenSize,
                kernelSize: kernel,
                stride: kernel,
                bias: true)
        }

        func callAsFunction(_ input: MLXArray) -> MLXArray {
            let patches = input.reshaped(
                -1, inChannels, temporalPatchSize, patchSize, patchSize
            ).movedAxis(source: 1, destination: 4)
            return projection(patches).reshaped(-1, hiddenSize)
        }
    }

    final class PatchMerger: Module, UnaryLayer {
        let swigluLimit: Float
        @ModuleInfo var proj: Linear
        @ModuleInfo(key: "post_projection_norm") var postProjectionNorm: LayerNorm
        @ModuleInfo(key: "gate_proj") var gateProjection: Linear
        @ModuleInfo(key: "up_proj") var upProjection: Linear
        @ModuleInfo(key: "down_proj") var downProjection: Linear

        init(_ config: GLM5NextVisionConfiguration) {
            swigluLimit = config.swigluLimit
            _proj.wrappedValue = Linear(config.outHiddenSize, config.outHiddenSize, bias: false)
            _postProjectionNorm.wrappedValue = LayerNorm(dimensions: config.outHiddenSize)
            _gateProjection.wrappedValue = Linear(
                config.outHiddenSize, config.projectionIntermediateSize, bias: false)
            _upProjection.wrappedValue = Linear(
                config.outHiddenSize, config.projectionIntermediateSize, bias: false)
            _downProjection.wrappedValue = Linear(
                config.projectionIntermediateSize, config.outHiddenSize, bias: false)
        }

        func callAsFunction(_ input: MLXArray) -> MLXArray {
            let projected = gelu(postProjectionNorm(proj(input)))
            let gate = minimum(gateProjection(projected), swigluLimit)
            let up = clip(upProjection(projected), min: -swigluLimit, max: swigluLimit)
            return downProjection(silu(gate) * up)
        }
    }

    final class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float
        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear
        @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm

        init(_ config: GLM5NextVisionConfiguration) {
            numHeads = config.numHeads
            headDim = config.hiddenSize / config.numHeads
            scale = pow(Float(headDim), -0.5)
            _qkv.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSize * 3, bias: config.attentionBias)
            _proj.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSize, bias: config.attentionBias)
            _queryNorm.wrappedValue = RMSNorm(
                dimensions: headDim, eps: config.rmsNormEps)
            _keyNorm.wrappedValue = RMSNorm(
                dimensions: headDim, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ input: MLXArray,
            cumulativeSequenceLengths: [Int],
            cosine: MLXArray,
            sine: MLXArray
        ) -> MLXArray {
            let length = input.dim(0)
            let projected = qkv(input).reshaped(length, 3, numHeads, headDim)
                .transposed(1, 0, 2, 3)
            let parts = split(projected, parts: 3, axis: 0)
            var queries = queryNorm(parts[0][0, 0..., 0..., 0...])
            var keys = keyNorm(parts[1][0, 0..., 0..., 0...])
            let values = parts[2][0, 0..., 0..., 0...]
            (queries, keys) = applyRotary(
                queries: queries, keys: keys, cosine: cosine, sine: sine)

            queries = queries.transposed(1, 0, 2)[.newAxis, 0..., 0..., 0...]
            keys = keys.transposed(1, 0, 2)[.newAxis, 0..., 0..., 0...]
            let transposedValues = values.transposed(1, 0, 2)[.newAxis, 0..., 0..., 0...]

            var outputs = [MLXArray]()
            for index in 1 ..< cumulativeSequenceLengths.count {
                let start = cumulativeSequenceLengths[index - 1]
                let end = cumulativeSequenceLengths[index]
                outputs.append(
                    MLXFast.scaledDotProductAttention(
                        queries: queries[0..., 0..., start ..< end, 0...],
                        keys: keys[0..., 0..., start ..< end, 0...],
                        values: transposedValues[0..., 0..., start ..< end, 0...],
                        scale: scale,
                        mask: nil))
            }
            return proj(
                concatenated(outputs, axis: 2)
                    .transposed(0, 2, 1, 3)
                    .reshaped(length, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        let swigluLimit: Float
        @ModuleInfo(key: "gate_proj") var gateProjection: Linear
        @ModuleInfo(key: "up_proj") var upProjection: Linear
        @ModuleInfo(key: "down_proj") var downProjection: Linear

        init(_ config: GLM5NextVisionConfiguration) {
            swigluLimit = config.swigluLimit
            _gateProjection.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: config.attentionBias)
            _upProjection.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: config.attentionBias)
            _downProjection.wrappedValue = Linear(
                config.intermediateSize, config.hiddenSize, bias: config.attentionBias)
        }

        func callAsFunction(_ input: MLXArray) -> MLXArray {
            let gate = minimum(gateProjection(input), swigluLimit)
            let up = clip(upProjection(input), min: -swigluLimit, max: swigluLimit)
            return downProjection(silu(gate) * up)
        }
    }

    final class Block: Module {
        @ModuleInfo var norm1: RMSNorm
        @ModuleInfo var norm2: RMSNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo var mlp: MLP

        init(_ config: GLM5NextVisionConfiguration) {
            _norm1.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _norm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _attention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
        }

        func callAsFunction(
            _ input: MLXArray,
            cumulativeSequenceLengths: [Int],
            cosine: MLXArray,
            sine: MLXArray
        ) -> MLXArray {
            var hidden = input + attention(
                norm1(input),
                cumulativeSequenceLengths: cumulativeSequenceLengths,
                cosine: cosine,
                sine: sine)
            hidden = hidden + mlp(norm2(hidden))
            return hidden
        }
    }

    final class VisionModel: Module {
        let config: GLM5NextVisionConfiguration
        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        @ModuleInfo(key: "blocks") var blocks: [Block]
        @ModuleInfo var merger: PatchMerger
        @ModuleInfo var downsample: Conv2d
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: RMSNorm
        let rotaryEmbedding: RotaryEmbedding

        init(_ config: GLM5NextVisionConfiguration) {
            self.config = config
            _patchEmbed.wrappedValue = PatchEmbed(config)
            _blocks.wrappedValue = (0 ..< config.depth).map { _ in Block(config) }
            _merger.wrappedValue = PatchMerger(config)
            _downsample.wrappedValue = Conv2d(
                inputChannels: config.hiddenSize,
                outputChannels: config.outHiddenSize,
                kernelSize: IntOrPair(config.spatialMergeSize),
                stride: IntOrPair(config.spatialMergeSize),
                bias: true)
            _postLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            rotaryEmbedding = RotaryEmbedding(
                dimension: (config.hiddenSize / config.numHeads) / 2)
        }

        private func rotaryPositions(_ grids: [THW]) -> (MLXArray, MLXArray) {
            let merge = config.spatialMergeSize
            var coordinates = [MLXArray]()
            for grid in grids {
                let mergedH = grid.h / merge
                let mergedW = grid.w / merge
                let blockRows = MLXArray(0 ..< mergedH).asType(.int32)
                    .reshaped(mergedH, 1, 1, 1)
                let blockColumns = MLXArray(0 ..< mergedW).asType(.int32)
                    .reshaped(1, mergedW, 1, 1)
                let intra = MLXArray(0 ..< merge).asType(.int32)
                let rows = broadcast(
                    blockRows * merge + intra.reshaped(1, 1, merge, 1),
                    to: [mergedH, mergedW, merge, merge])
                let columns = broadcast(
                    blockColumns * merge + intra.reshaped(1, 1, 1, merge),
                    to: [mergedH, mergedW, merge, merge])
                var gridCoordinates = stacked(
                    [rows.flattened(), columns.flattened()], axis: -1)
                if grid.t > 1 {
                    gridCoordinates = tiled(gridCoordinates, repetitions: [grid.t, 1])
                }
                coordinates.append(gridCoordinates)
            }
            let positions = concatenated(coordinates, axis: 0)
            let maxGrid = grids.map { max($0.h, $0.w) }.max() ?? 1
            let frequencies = rotaryEmbedding(sequenceLength: maxGrid)
            let height = frequencies[positions[0..., 0].asType(.int32), 0...]
            let width = frequencies[positions[0..., 1].asType(.int32), 0...]
            let embedding = concatenated([height, width, height, width], axis: -1)
            return (cos(embedding), sin(embedding))
        }

        private func cumulativeSequenceLengths(_ grids: [THW]) -> [Int] {
            var result = [0]
            for grid in grids {
                for _ in 0 ..< grid.t {
                    result.append((result.last ?? 0) + grid.h * grid.w)
                }
            }
            return result
        }

        func callAsFunction(_ pixels: MLXArray, gridTHW: [THW]) -> MLXArray {
            var hidden = patchEmbed(pixels)
            let (cosine, sine) = rotaryPositions(gridTHW)
            let cumulative = cumulativeSequenceLengths(gridTHW)
            for block in blocks {
                hidden = block(
                    hidden,
                    cumulativeSequenceLengths: cumulative,
                    cosine: cosine,
                    sine: sine)
            }
            hidden = postLayerNorm(hidden)
            let merge = config.spatialMergeSize
            hidden = hidden.reshaped(-1, merge, merge, config.hiddenSize)
            hidden = downsample(hidden).reshaped(-1, config.outHiddenSize)
            return merger(hidden)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var result = [String: MLXArray]()
            for (key, value) in weights where !key.contains("position_ids") {
                if key.contains("patch_embed.proj.weight") {
                    result[key] = value.ndim == 5 && value.dim(-1) == config.inChannels
                        ? value : value.transposed(0, 2, 3, 4, 1)
                } else if key == "downsample.weight" {
                    result[key] = value.ndim == 4 && value.dim(-1) == config.hiddenSize
                        ? value : value.transposed(0, 2, 3, 1)
                } else {
                    result[key] = value
                }
            }
            return result
        }
    }
}

// MARK: - Multimodal wrapper

public final class GLM5NextVLModel: Module, VLMModel, KVCacheDimensionProvider {
    let config: GLM5NextVLConfiguration
    @ModuleInfo(key: "vision_model") var visionModel: GLM5NextVision.VisionModel
    @ModuleInfo(key: "language_model") var languageModel: GLM5NextModel

    public init(_ config: GLM5NextVLConfiguration) {
        self.config = config
        _visionModel.wrappedValue = GLM5NextVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = GLM5NextModel(config.languageConfiguration)
    }

    public var vocabularySize: Int { languageModel.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.loraLayers }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    private func merge(
        features: MLXArray,
        embeddings: MLXArray,
        inputIDs: MLXArray
    ) throws -> MLXArray {
        let mask = (inputIDs .== MLXArray(config.imageTokenID))
            .|| (inputIDs .== MLXArray(config.videoTokenID))
        let expanded = broadcast(expandedDimensions(mask, axis: -1), to: embeddings.shape)
        guard expanded.sum().item(Int.self) == features.size else {
            throw VLMError.processing(
                "GLM visual feature/token mismatch: \(features.dim(0)) features for "
                    + "\(mask.sum().item(Int.self)) media tokens")
        }
        let indices = expanded.flattened().asArray(Bool.self).enumerated().compactMap {
            $0.element ? UInt32($0.offset) : nil
        }
        var result = embeddings.flattened()
        result[MLXArray(indices)] = features.flattened()
        return result.reshaped(embeddings.shape)
    }

    public func prepare(
        _ input: LMInput,
        cache: [KVCache],
        windowSize _: Int?
    ) throws -> PrepareResult {
        let inputIDs = input.text.tokens
        let grids = (input.image?.frames ?? []) + (input.video?.frames ?? [])
        var pixelParts = [MLXArray]()
        let dtype = visionModel.patchEmbed.projection.weight.dtype
        if let image = input.image { pixelParts.append(image.pixels.asType(dtype)) }
        if let video = input.video { pixelParts.append(video.pixels.asType(dtype)) }

        var embeddings: MLXArray?
        if !pixelParts.isEmpty, !grids.isEmpty {
            let textEmbeddings = languageModel.embedTokens(inputIDs)
            let features = visionModel(concatenated(pixelParts), gridTHW: grids)
                .asType(textEmbeddings.dtype)
            embeddings = try merge(
                features: features,
                embeddings: textEmbeddings,
                inputIDs: inputIDs)
        }
        let logits = languageModel.forward(
            inputIDs: inputIDs,
            inputEmbeddings: embeddings,
            cache: cache)
        return .logits(LMOutput(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in languageModel.sanitize(weights: weights) {
            result["language_model.\(key)"] = value
        }

        var vision = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("model.visual.") {
                vision[String(key.dropFirst("model.visual.".count))] = value
            } else if key.hasPrefix("visual.") {
                vision[String(key.dropFirst("visual.".count))] = value
            } else if key.hasPrefix("vision_model.") {
                vision[String(key.dropFirst("vision_model.".count))] = value
            }
        }
        for (key, value) in visionModel.sanitize(weights: vision) {
            result["vision_model.\(key)"] = value
        }
        return result
    }
}
