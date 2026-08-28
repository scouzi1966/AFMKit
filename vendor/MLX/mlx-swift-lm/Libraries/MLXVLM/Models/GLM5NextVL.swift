//
//  GLM5NextVL.swift
//  mlx-swift-lm
//
//  Self-contained MLX Swift vision-language implementation for GLM-5.3 Flash.
//  The text trunk lives in MLXLLM/GLM5Next.swift; this file owns the published
//  glm5_next_vision tower, nested processor configuration, and media embedding.
//


@preconcurrency import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXFast
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Processor

private func glmCheckedProduct(_ factors: Int...) -> Int? {
    var product = 1
    for factor in factors {
        guard factor > 0 else { return nil }
        let (next, overflow) = product.multipliedReportingOverflow(by: factor)
        guard !overflow else { return nil }
        product = next
    }
    return product
}

public struct GLM5NextProcessorConfiguration: Codable, Sendable {
    public struct MediaConfiguration: Codable, Sendable {
        public let imageMean: [CGFloat]
        public let imageStd: [CGFloat]
        public let mergeSize: Int
        public let patchSize: Int
        public let temporalPatchSize: Int
        public let minImageTokens: Int
        public let maxImageTokens: Int
        let patchExpandFactor: Int?
        let maxFrames: Int?
        public let fps: Double?

        enum CodingKeys: String, CodingKey {
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case mergeSize = "merge_size"
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case minImageTokens = "min_image_tokens"
            case maxImageTokens = "max_image_tokens"
            case patchExpandFactor = "patch_expand_factor"
            case maxFrames = "max_frames"
            case fps
        }

        var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
            (imageMean[0], imageMean[1], imageMean[2])
        }

        var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
            (imageStd[0], imageStd[1], imageStd[2])
        }

        public var minPixels: Int {
            glmCheckedProduct(
                minImageTokens, patchSize, patchSize, mergeSize, mergeSize) ?? 0
        }

        public var maxPixels: Int {
            glmCheckedProduct(
                maxImageTokens, patchSize, patchSize, mergeSize, mergeSize) ?? 0
        }

        var effectivePatchExpandFactor: Int { patchExpandFactor ?? 1 }
        var effectiveMaximumFrames: Int { maxFrames ?? 2_048 }
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
    struct ResizePlan: Equatable {
        let targetHeight: Int
        let targetWidth: Int
        let contentHeight: Int
        let contentWidth: Int
        let alignedFrames: Int
    }

    private let config: GLM5NextProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: GLM5NextProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    /// Port of Transformers `glm5_next.smart_resize` plus its aspect-preserving
    /// content fit. Token limits are spatiotemporal token counts, not bare pixels.
    static func resizePlan(
        numFrames: Int,
        height: Int,
        width: Int,
        configuration: GLM5NextProcessorConfiguration.MediaConfiguration
    ) throws -> ResizePlan {
        let temporal = configuration.temporalPatchSize
        guard numFrames > 0, height > 0, width > 0, temporal > 0,
              configuration.minImageTokens > 0,
              configuration.maxImageTokens >= configuration.minImageTokens
        else {
            throw VLMError.imageProcessingFailure("Invalid GLM media geometry")
        }

        guard let factor = glmCheckedProduct(
            configuration.patchSize,
            configuration.mergeSize,
            configuration.effectivePatchExpandFactor),
            let pixelsPerToken = glmCheckedProduct(temporal, factor, factor),
            let minimumPixels = glmCheckedProduct(
                configuration.minImageTokens, pixelsPerToken),
            let maximumPixels = glmCheckedProduct(
                configuration.maxImageTokens, pixelsPerToken)
        else {
            throw VLMError.imageProcessingFailure("GLM media geometry overflow")
        }

        func align(_ value: Int) throws -> Int {
            let quotient = value / factor
            guard !value.isMultiple(of: factor) else { return value }
            guard let aligned = glmCheckedProduct(quotient + 1, factor) else {
                throw VLMError.imageProcessingFailure("GLM aligned geometry overflow")
            }
            return aligned
        }

        let frameGroups = numFrames / temporal
        let remainder = numFrames % temporal
        let distanceToNextGroup = temporal - remainder
        let roundUp = remainder > distanceToNextGroup
            || (remainder == distanceToNextGroup && !frameGroups.isMultiple(of: 2))
        let (roundedFrameGroups, groupOverflow) = frameGroups.addingReportingOverflow(
            roundUp ? 1 : 0)
        guard !groupOverflow,
              let roundedFrames = glmCheckedProduct(
                max(1, roundedFrameGroups), temporal),
              let minimumPatchPixels = glmCheckedProduct(
                roundedFrames, factor, factor),
              maximumPixels >= minimumPatchPixels
        else {
            throw VLMError.imageProcessingFailure(
                "GLM max_image_tokens is too small for one aligned patch")
        }
        let alignedFrames = max(temporal, roundedFrames)

        var targetHeight = try align(height)
        var targetWidth = try align(width)
        guard var budget = glmCheckedProduct(
            alignedFrames, targetHeight, targetWidth)
        else {
            throw VLMError.imageProcessingFailure("GLM media pixel budget overflow")
        }
        if budget < minimumPixels {
            let sourcePixels = Double(numFrames) * Double(height) * Double(width)
            guard sourcePixels.isFinite, sourcePixels > 0 else {
                throw VLMError.imageProcessingFailure("GLM source geometry overflow")
            }
            let scale = sqrt(Double(minimumPixels) / sourcePixels)
            let scaledHeight = ceil(Double(height) * scale)
            let scaledWidth = ceil(Double(width) * scale)
            guard scaledHeight.isFinite, scaledHeight < Double(Int.max),
                  scaledWidth.isFinite, scaledWidth < Double(Int.max)
            else {
                throw VLMError.imageProcessingFailure("GLM resized geometry overflow")
            }
            targetHeight = try align(max(1, Int(scaledHeight)))
            targetWidth = try align(max(1, Int(scaledWidth)))
            guard let scaledBudget = glmCheckedProduct(
                alignedFrames, targetHeight, targetWidth)
            else {
                throw VLMError.imageProcessingFailure("GLM media pixel budget overflow")
            }
            budget = scaledBudget
        }
        if budget > maximumPixels {
            var low = 1
            var high = height
            var bestHeight = factor
            var bestWidth = factor
            while low <= high {
                let contentHeight = (low + high) / 2
                let proportionalWidth = floor(
                    Double(width) * Double(contentHeight) / Double(height))
                guard proportionalWidth.isFinite, proportionalWidth < Double(Int.max)
                else {
                    throw VLMError.imageProcessingFailure("GLM aspect ratio overflow")
                }
                let contentWidth = max(1, Int(proportionalWidth))
                let candidateHeight = try align(contentHeight)
                let candidateWidth = try align(contentWidth)
                if let candidateBudget = glmCheckedProduct(
                    alignedFrames, candidateHeight, candidateWidth),
                    candidateBudget <= maximumPixels
                {
                    bestHeight = candidateHeight
                    bestWidth = candidateWidth
                    low = contentHeight + 1
                } else {
                    high = contentHeight - 1
                }
            }
            targetHeight = bestHeight
            targetWidth = bestWidth
        }

        var scale = min(
            Double(targetHeight) / Double(height),
            Double(targetWidth) / Double(width))
        let sourcePixels = Double(numFrames) * Double(height) * Double(width)
        if sourcePixels >= Double(minimumPixels) {
            scale = min(1, scale)
        }
        let contentHeight = max(
            1, min(targetHeight, Int(floor(Double(height) * scale))))
        let contentWidth = max(
            1, min(targetWidth, Int(floor(Double(width) * scale))))
        return ResizePlan(
            targetHeight: targetHeight,
            targetWidth: targetWidth,
            contentHeight: contentHeight,
            contentWidth: contentWidth,
            alignedFrames: alignedFrames)
    }

    static func resizeAndPad(_ image: CIImage, plan: ResizePlan) -> CIImage {
        let resized = MediaProcessing.resampleBicubic(
            image,
            to: CGSize(width: plan.contentWidth, height: plan.contentHeight))
        guard plan.contentWidth != plan.targetWidth
                || plan.contentHeight != plan.targetHeight else { return resized }
        let background = CIImage(color: .black).cropped(
            to: CGRect(
                x: 0, y: 0,
                width: plan.targetWidth, height: plan.targetHeight))
        // Core Image's origin is bottom-left. Translating upward leaves padding
        // on the right and bottom in tensor (top-left-origin) coordinates.
        let placed = resized.transformed(
            by: CGAffineTransform(
                translationX: 0,
                y: CGFloat(plan.targetHeight - plan.contentHeight)))
        return placed.composited(over: background)
    }

    /// Current Transformers GLM5 Next sampling: compute a floored target count,
    /// sample source-frame thresholds, stabilize duplicate indices, then make
    /// the temporal sequence even by repeating its final real frame.
    static func checkedFrameCount(duration: Double, fps: Double) throws -> Int {
        let count = floor(duration * fps)
        guard duration.isFinite, duration > 0, fps.isFinite, fps > 0,
              count.isFinite, count > 0, count < Double(Int.max)
        else {
            throw VLMError.imageProcessingFailure("GLM video sampling count overflow")
        }
        return Int(count)
    }

    static func videoSampleIndices(
        totalFrames: Int,
        sourceFPS: Double,
        duration: Double? = nil,
        targetFPS: Double,
        maximumFrames: Int
    ) throws -> [Int] {
        guard totalFrames > 0, sourceFPS > 0, targetFPS > 0, maximumFrames > 0 else {
            throw VLMError.imageProcessingFailure("Invalid GLM video sampling metadata")
        }

        let maximumFrameIndex = totalFrames - 1
        let resolvedDuration = duration
            ?? (Double(maximumFrameIndex) / sourceFPS).rounded(.toNearestOrEven) + 1
        let targetCount = min(
            try checkedFrameCount(duration: resolvedDuration, fps: targetFPS),
            maximumFrames)
        guard targetCount > 0 else {
            throw VLMError.imageProcessingFailure("GLM video duration produced no frames")
        }

        func linspace(_ start: Int, _ end: Int, count: Int) -> [Int] {
            guard count > 1 else { return [start] }
            return (0..<count).map { position in
                Int(
                    Double(start)
                        + Double(end - start) * Double(position) / Double(count - 1))
            }
        }

        var indices: [Int]
        if totalFrames < targetCount {
            indices = linspace(0, maximumFrameIndex, count: targetCount)
        } else {
            let maximumSeconds = floor(resolvedDuration)
            var currentSecond = 0.0
            indices = []
            for frameIndex in 0..<totalFrames {
                let timestamp = Double(frameIndex) / sourceFPS
                if timestamp >= currentSecond {
                    currentSecond += 1 / targetFPS
                    indices.append(frameIndex)
                    if currentSecond >= maximumSeconds { break }
                }
            }
        }

        if indices.count < targetCount {
            let start = indices.first ?? 0
            let end = indices.last ?? maximumFrameIndex
            indices = linspace(start, end, count: targetCount)
        } else if indices.count > targetCount {
            indices = linspace(0, maximumFrameIndex, count: targetCount)
        }

        var seen = Set<Int>()
        indices = indices.filter { seen.insert($0).inserted }
        guard let finalIndex = indices.last else {
            throw VLMError.imageProcessingFailure("GLM video sampling produced no frames")
        }
        if !indices.count.isMultiple(of: 2) {
            indices.append(finalIndex)
        }
        return indices
    }

    private static func inferredFrameRate(_ frames: [VideoFrame]) -> Double? {
        let deltas = zip(frames, frames.dropFirst()).compactMap { first, second in
            let seconds = (second.timeStamp - first.timeStamp).seconds
            return seconds > 0 && seconds.isFinite ? seconds : nil
        }.sorted()
        guard let median = deltas.isEmpty ? nil : deltas[deltas.count / 2] else {
            return nil
        }
        return 1 / median
    }

    static func sampleVideo(
        _ video: UserInput.Video,
        targetFPS: Double,
        maximumFrames: Int
    ) async throws -> [VideoFrame] {
        switch video {
        case .frames(let frames):
            guard !frames.isEmpty else {
                throw VLMError.imageProcessingFailure(
                    "GLM in-memory video requires at least one source frame")
            }
            // Transformers defaults metadata without an FPS to 24 for prompt
            // timestamp construction. Keep pre-sampled one-frame inputs usable.
            let sourceFPS = inferredFrameRate(frames) ?? 24
            let indices = try videoSampleIndices(
                totalFrames: frames.count,
                sourceFPS: sourceFPS,
                targetFPS: targetFPS,
                maximumFrames: maximumFrames)
            return indices.map { frames[$0] }

        case .url(let url):
            return try await sampleVideo(
                AVURLAsset(url: url), targetFPS: targetFPS, maximumFrames: maximumFrames)
        case .avAsset(let asset):
            return try await sampleVideo(
                asset, targetFPS: targetFPS, maximumFrames: maximumFrames)
        }
    }

    private static func sampleVideo(
        _ asset: AVAsset,
        targetFPS: Double,
        maximumFrames: Int
    ) async throws -> [VideoFrame] {
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw VLMError.noVideoTrackFound
        }
        guard try await track.load(.isDecodable) else {
            throw VLMError.videoNotDecodable
        }
        let sourceFPS = Double(try await track.load(.nominalFrameRate))
        let duration = try await asset.load(.duration)
        guard duration.seconds.isFinite, duration.seconds > 0, sourceFPS > 0 else {
            throw VLMError.imageProcessingFailure("Invalid GLM AVAsset metadata")
        }
        // A media duration is the exclusive end of its last source frame.
        let totalFrames = try checkedFrameCount(
            duration: duration.seconds, fps: sourceFPS)
        let indices = try videoSampleIndices(
            totalFrames: totalFrames,
            sourceFPS: sourceFPS,
            duration: duration.seconds,
            targetFPS: targetFPS,
            maximumFrames: maximumFrames)

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        var seenIndices = Set<Int>()
        let uniqueIndices = indices.filter { seenIndices.insert($0).inserted }
        let times = uniqueIndices.map {
            CMTime(seconds: Double($0) / sourceFPS, preferredTimescale: 60_000)
        }
        var framesByIndex = [Int: VideoFrame]()
        for await result in generator.images(for: times) {
            switch result {
            case .success(requestedTime: let requested, let image, actualTime: let actual):
                let sourceIndex = Int((requested.seconds * sourceFPS).rounded())
                framesByIndex[sourceIndex] = VideoFrame(
                    frame: CIImage(
                        cgImage: image,
                        options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!]),
                    timeStamp: actual)
            case .failure(requestedTime: _, let error):
                throw error
            }
        }
        guard framesByIndex.count == uniqueIndices.count else {
            throw VLMError.imageProcessingFailure("GLM AVAsset frame extraction was incomplete")
        }
        return try indices.map { index in
            guard let frame = framesByIndex[index] else {
                throw VLMError.imageProcessingFailure(
                    "GLM AVAsset frame extraction omitted source index \(index)")
            }
            return frame
        }
    }

    static func preprocess(
        images: [CIImage],
        configuration: GLM5NextProcessorConfiguration.MediaConfiguration,
        processing: UserInput.Processing?,
        budgetFrameCount: Int
    ) throws -> (MLXArray, THW) {
        let processed = images
            .map(MediaProcessing.inSRGBToneCurveSpace)
            .map { MediaProcessing.apply($0, processing: processing) }
        guard let first = processed.first else {
            throw VLMError.imageProcessingFailure("No image provided")
        }

        let extent = first.extent.size
        let plan = try Self.resizePlan(
            numFrames: budgetFrameCount,
            height: Int(extent.height),
            width: Int(extent.width),
            configuration: configuration)
        let normalized = processed
            .map { Self.resizeAndPad($0, plan: plan) }
            .map {
                MediaProcessing.normalize(
                    $0,
                    mean: configuration.imageMeanTuple,
                    std: configuration.imageStdTuple)
            }
            .map {
                MediaProcessing.asMLXArray(
                    $0, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
            }

        return try QwenVL.patchify(
            images: normalized,
            mergeSize: configuration.mergeSize,
            patchSize: configuration.patchSize,
            temporalPatchSize: configuration.temporalPatchSize)
    }

    static func replaceImagePlaceholders(
        in promptTokens: [Int],
        grids: [THW],
        imageTokenID: Int,
        mergeSize: Int
    ) throws -> [Int] {
        let placeholderCount = promptTokens.count(where: { $0 == imageTokenID })
        guard placeholderCount == grids.count else {
            throw VLMError.processing(
                "Number of GLM image placeholders does not match image inputs")
        }
        var gridIndex = 0
        return promptTokens.flatMap { token -> [Int] in
            guard token == imageTokenID else { return [token] }
            defer { gridIndex += 1 }
            let count = grids[gridIndex].product / (mergeSize * mergeSize)
            return Array(repeating: imageTokenID, count: count)
        }
    }

    static func replaceVideoPlaceholders(
        in promptTokens: [Int],
        grids: [THW],
        timestamps: [[Double]],
        videoTokenID: Int,
        imageTokenID: Int,
        imageStartTokenID: Int,
        imageEndTokenID: Int,
        mergeSize: Int,
        temporalPatchSize: Int,
        timestampTokens: (Double) -> [Int]
    ) throws -> [Int] {
        let placeholderCount = promptTokens.count(where: { $0 == videoTokenID })
        guard placeholderCount == grids.count, timestamps.count == grids.count else {
            throw VLMError.processing(
                "Number of GLM video placeholders does not match video inputs")
        }
        var videoIndex = 0
        return promptTokens.flatMap { token -> [Int] in
            guard token == videoTokenID else { return [token] }
            defer { videoIndex += 1 }
            let grid = grids[videoIndex]
            let tokensPerFrame = grid.h * grid.w / (mergeSize * mergeSize)
            let candidates = stride(
                from: 0,
                to: timestamps[videoIndex].count,
                by: temporalPatchSize
            ).map { timestamps[videoIndex][$0] }
            var replacement = [Int]()
            for frame in 0..<grid.t {
                let timestamp = candidates.indices.contains(frame)
                    ? candidates[frame] : (candidates.last ?? 0)
                replacement.append(imageStartTokenID)
                replacement.append(
                    contentsOf: repeatElement(imageTokenID, count: tokensPerFrame))
                replacement.append(imageEndTokenID)
                replacement.append(contentsOf: timestampTokens(timestamp))
            }
            return replacement
        }
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

        guard let imageTokenID = tokenizer.convertTokenToId("<|image|>"),
              let videoTokenID = tokenizer.convertTokenToId("<|video|>"),
              let imageStartTokenID = tokenizer.convertTokenToId("<|begin_of_image|>"),
              let imageEndTokenID = tokenizer.convertTokenToId("<|end_of_image|>")
        else {
            throw VLMError.processing("GLM multimodal special tokens are unavailable")
        }

        let imageConfiguration = config.imageProcessor
        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let images = try input.images.map {
                try Self.preprocess(
                    images: [$0.asCIImage()],
                    configuration: imageConfiguration,
                    processing: input.processing,
                    budgetFrameCount: imageConfiguration.temporalPatchSize)
            }
            processedImage = .init(
                pixels: concatenated(images.map(\.0)),
                frames: images.map(\.1))
            promptTokens = try Self.replaceImagePlaceholders(
                in: promptTokens,
                grids: images.map(\.1),
                imageTokenID: imageTokenID,
                mergeSize: imageConfiguration.mergeSize)
        }

        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            guard let videoConfiguration = config.videoProcessor else {
                throw VLMError.videoNotSupported("glm5_next")
            }
            var videos = [(MLXArray, THW)]()
            var videoTimestamps = [[Double]]()
            for video in input.videos {
                let sampledFrames = try await Self.sampleVideo(
                    video,
                    targetFPS: videoConfiguration.fps ?? 2,
                    maximumFrames: videoConfiguration.effectiveMaximumFrames)
                videos.append(try Self.preprocess(
                    images: sampledFrames.map(\.frame),
                    configuration: videoConfiguration,
                    processing: input.processing,
                    budgetFrameCount: sampledFrames.count))
                videoTimestamps.append(sampledFrames.map { $0.timeStamp.seconds })
            }
            processedVideo = .init(
                pixels: concatenated(videos.map(\.0)),
                frames: videos.map(\.1))
            promptTokens = try Self.replaceVideoPlaceholders(
                in: promptTokens,
                grids: videos.map(\.1),
                timestamps: videoTimestamps,
                videoTokenID: videoTokenID,
                imageTokenID: imageTokenID,
                imageStartTokenID: imageStartTokenID,
                imageEndTokenID: imageEndTokenID,
                mergeSize: videoConfiguration.mergeSize,
                temporalPatchSize: videoConfiguration.temporalPatchSize
            ) { timestamp in
                let text = String(
                    format: "%.1f seconds",
                    locale: Locale(identifier: "en_US_POSIX"),
                    timestamp)
                return tokenizer.encode(text: text)
            }
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
    let imageStartTokenID: Int
    let imageEndTokenID: Int
    let videoStartTokenID: Int
    let videoEndTokenID: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case visionConfiguration = "vision_config"
        case imageTokenID = "image_token_id"
        case videoTokenID = "video_token_id"
        case imageStartTokenID = "image_start_token_id"
        case imageEndTokenID = "image_end_token_id"
        case videoStartTokenID = "video_start_token_id"
        case videoEndTokenID = "video_end_token_id"
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
        imageStartTokenID = try container.decode(Int.self, forKey: .imageStartTokenID)
        imageEndTokenID = try container.decode(Int.self, forKey: .imageEndTokenID)
        videoStartTokenID = try container.decode(Int.self, forKey: .videoStartTokenID)
        videoEndTokenID = try container.decode(Int.self, forKey: .videoEndTokenID)
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

public final class GLM5NextVLModel: Module, VLMModel, KVCacheDimensionProvider,
    LanguageModelWeightFilter
{
    struct ModalityTokenPositions: Equatable {
        let images: [Int]
        let videos: [Int]
    }

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

    static func modalityTokenPositions(
        inputIDs: [Int],
        imageTokenID: Int,
        videoStartTokenID: Int,
        videoEndTokenID: Int
    ) throws -> ModalityTokenPositions {
        var videoDepth = 0
        var images = [Int]()
        var videos = [Int]()
        for (index, token) in inputIDs.enumerated() {
            if token == videoStartTokenID {
                videoDepth += 1
            } else if token == videoEndTokenID {
                guard videoDepth > 0 else {
                    throw VLMError.processing("Unbalanced GLM video boundary tokens")
                }
                videoDepth -= 1
            } else if token == imageTokenID {
                if videoDepth > 0 {
                    videos.append(index)
                } else {
                    images.append(index)
                }
            }
        }
        guard videoDepth == 0 else {
            throw VLMError.processing("Unbalanced GLM video boundary tokens")
        }
        return ModalityTokenPositions(images: images, videos: videos)
    }

    static func merge(
        features: MLXArray,
        embeddings: MLXArray,
        tokenPositions: [Int],
        modality: String
    ) throws -> MLXArray {
        var maskValues = Array(repeating: false, count: embeddings.dim(1))
        for position in tokenPositions where maskValues.indices.contains(position) {
            maskValues[position] = true
        }
        let mask = MLXArray(maskValues)[.newAxis, 0...]
        let expanded = broadcast(expandedDimensions(mask, axis: -1), to: embeddings.shape)
        guard expanded.sum().item(Int.self) == features.size else {
            throw VLMError.processing(
                "GLM \(modality) feature/token mismatch: \(features.dim(0)) features for "
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
        let positions: ModalityTokenPositions
        if input.image != nil || input.video != nil {
            positions = try Self.modalityTokenPositions(
                inputIDs: inputIDs.asArray(Int.self),
                imageTokenID: config.imageTokenID,
                videoStartTokenID: config.videoStartTokenID,
                videoEndTokenID: config.videoEndTokenID)
        } else {
            positions = ModalityTokenPositions(images: [], videos: [])
        }
        let dtype = visionModel.patchEmbed.projection.weight.dtype
        var embeddings: MLXArray? = input.image == nil && input.video == nil
            ? nil : languageModel.embedTokens(inputIDs)

        if let image = input.image,
           let imageGrids = image.frames,
           !imageGrids.isEmpty {
            let features = visionModel(
                image.pixels.asType(dtype), gridTHW: imageGrids)
                .asType(embeddings!.dtype)
            embeddings = try Self.merge(
                features: features,
                embeddings: embeddings!,
                tokenPositions: positions.images,
                modality: "image")
        }
        if let video = input.video,
           let videoGrids = video.frames,
           !videoGrids.isEmpty {
            let frameGrids = videoGrids.flatMap { grid in
                Array(repeating: THW(1, grid.h, grid.w), count: grid.t)
            }
            let features = visionModel(
                video.pixels.asType(dtype), gridTHW: frameGrids)
                .asType(embeddings!.dtype)
            embeddings = try Self.merge(
                features: features,
                embeddings: embeddings!,
                tokenPositions: positions.videos,
                modality: "video")
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

    public func shouldLoad(weightKey key: String) -> Bool {
        if key.hasPrefix("model.visual.") || key.hasPrefix("vision_model.") {
            return true
        }
        if key.hasPrefix("visual.") || key.hasPrefix("vision_tower.") {
            return false
        }
        return languageModel.shouldLoad(weightKey: key)
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
