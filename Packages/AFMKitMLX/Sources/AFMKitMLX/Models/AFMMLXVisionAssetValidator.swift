import Foundation
import MLXVLM

public final class AFMMLXVisionAssetValidator: @unchecked Sendable {
    private static let maximumQualificationMetadataBytes = 64 * 1_024 * 1_024
    private struct SafetensorEvidence {
        struct TensorMetadata {
            let dtype: String
            let shape: [Int]
        }

        let tensors: [String: TensorMetadata]

        var tensorNames: Set<String> { Set(tensors.keys) }
    }

    private struct SnapshotFingerprint: Hashable {
        struct FileEvidence: Hashable {
            let name: String
            let size: Int64
            let modifiedAt: TimeInterval
            let contentSignature: UInt64
        }

        let directory: String
        let files: [FileEvidence]

        var identity: String {
            let evidence = files.map {
                "\($0.name):\($0.size):\($0.modifiedAt):\($0.contentSignature)"
            }.joined(separator: "|")
            return "\(directory)|\(evidence)"
        }
    }

    private let lock = NSLock()
    private var cache: [SnapshotFingerprint: AFMMLXVisionAssetQualification] = [:]

    public init() {}

    public func qualify(
        modelDirectory: URL,
        architecture: AFMMLXModelArchitecturePreflight
    ) -> AFMMLXVisionAssetQualification {
        guard let fingerprint = snapshotFingerprint(for: modelDirectory) else {
            return Self.inspect(
                modelDirectory: modelDirectory,
                architecture: architecture,
                snapshotIdentity: "\(modelDirectory.standardizedFileURL.path)|uncached"
            )
        }
        if let cached = withLock({ cache[fingerprint] }) { return cached }

        let qualification = Self.inspect(
            modelDirectory: modelDirectory,
            architecture: architecture,
            snapshotIdentity: fingerprint.identity
        )
        return withLock {
            if let cached = cache[fingerprint] {
                return cached
            }
            cache[fingerprint] = qualification
            return qualification
        }
    }

    private static func inspect(
        modelDirectory: URL,
        architecture: AFMMLXModelArchitecturePreflight,
        snapshotIdentity: String
    ) -> AFMMLXVisionAssetQualification {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        let configData = try? Data(contentsOf: configURL)
        let config = jsonObject(at: configURL) ?? [:]
        let isConditionalGeneration = conditionalGenerationArchitecture(in: config)
        let isQwenConditional = isQwenConditionalModelType(
            architecture.canonicalModelType
        ) && isConditionalGeneration
        let hasVisionConfiguration: Bool
        if isQwenConditional {
            hasVisionConfiguration = configData.flatMap {
                try? JSONDecoder().decode(Qwen3_5MoEVLConfiguration.self, from: $0)
            } != nil && hasCoherentQwenVisionDimensions(in: config)
        } else if architecture.canonicalModelType == "glm5_next" {
            hasVisionConfiguration = hasCoherentGLM5NextVisionDimensions(in: config)
        } else {
            hasVisionConfiguration = config["vision_config"] is [String: Any]
        }
        let hasImageTokenIdentifiers: Bool
        if architecture.canonicalModelType == "glm5_next" {
            hasImageTokenIdentifiers = positiveInteger(config["image_token_id"]) != nil
                && positiveInteger(config["video_token_id"]) != nil
                && positiveInteger(config["image_start_token_id"]) != nil
                && positiveInteger(config["image_end_token_id"]) != nil
                && positiveInteger(config["video_start_token_id"]) != nil
                && positiveInteger(config["video_end_token_id"]) != nil
        } else {
            hasImageTokenIdentifiers = integer(config["image_token_id"]) != nil
                && integer(config["vision_start_token_id"]) != nil
                && integer(config["vision_end_token_id"]) != nil
        }
        let processorClass = selectedProcessorClass(
            modelDirectory: modelDirectory,
            canonicalModelType: architecture.canonicalModelType,
            config: config
        )
        let visionTensorNames = visionTensorNames(
            in: modelDirectory,
            config: config,
            requiresCompleteQwenTower: isQwenConditional,
            requiresCompleteGLMTower: architecture.canonicalModelType == "glm5_next"
        )
        let visionTensorCount = visionTensorNames.count

        var missing = Set<AFMMLXVisionAssetIssue>()
        if !isConditionalGeneration {
            missing.insert(.conditionalGenerationArchitecture)
        }
        if !hasVisionConfiguration {
            missing.insert(.visionConfiguration)
        }
        if !hasImageTokenIdentifiers {
            missing.insert(.imageTokenIdentifiers)
        }
        if processorClass == nil {
            missing.insert(.processorConfiguration)
        }
        if visionTensorCount == 0 {
            missing.insert(.visionWeights)
        }

        return AFMMLXVisionAssetQualification(
            snapshotIdentity: snapshotIdentity,
            modelType: architecture.modelType,
            canonicalModelType: architecture.canonicalModelType,
            isConditionalGeneration: isConditionalGeneration,
            declaresVision: architecture.isVisionConfiguration,
            processorClass: processorClass,
            visionTensorCount: visionTensorCount,
            missingAssets: missing
        )
    }

    private static func conditionalGenerationArchitecture(
        in config: [String: Any]
    ) -> Bool {
        guard let architectures = config["architectures"] as? [String] else {
            return false
        }
        return architectures.contains { architecture in
            let normalized = architecture
                .lowercased()
                .replacingOccurrences(of: "_", with: "")
                .replacingOccurrences(of: "-", with: "")
            return (normalized.hasPrefix("qwen35")
                || normalized.hasPrefix("qwen4exp")
                || normalized.hasPrefix("glm5next"))
                && normalized.hasSuffix("forconditionalgeneration")
        }
    }

    private static func isQwenConditionalModelType(_ canonicalModelType: String) -> Bool {
        canonicalModelType == "qwen3_5"
            || canonicalModelType == "qwen3_5_moe"
            || canonicalModelType == "qwen4_exp"
    }

    private static func hasCoherentQwenVisionDimensions(
        in config: [String: Any]
    ) -> Bool {
        guard let text = config["text_config"] as? [String: Any],
              let vision = config["vision_config"] as? [String: Any],
              let textHidden = positiveInteger(text["hidden_size"]),
              let visionHidden = positiveInteger(vision["hidden_size"]),
              let outHidden = positiveInteger(vision["out_hidden_size"]),
              let visionHeads = positiveInteger(vision["num_heads"])
        else { return false }

        guard outHidden == textHidden,
              visionHidden.isMultiple(of: visionHeads)
        else { return false }

        let headWidth = visionHidden / visionHeads
        return headWidth.isMultiple(of: 4)
    }

    private static func hasCoherentGLM5NextVisionDimensions(
        in config: [String: Any]
    ) -> Bool {
        guard let text = config["text_config"] as? [String: Any],
              let vision = config["vision_config"] as? [String: Any],
              let textHidden = positiveInteger(text["hidden_size"]),
              let visionHidden = positiveInteger(vision["hidden_size"]),
              let outHidden = positiveInteger(vision["out_hidden_size"]),
              let heads = positiveInteger(vision["num_heads"]),
              let depth = positiveInteger(vision["depth"]),
              let intermediate = positiveInteger(vision["intermediate_size"]),
              let projectionIntermediate = positiveInteger(
                vision["projection_intermediate_size"]),
              let patchSize = positiveInteger(vision["patch_size"]),
              let temporalPatchSize = positiveInteger(vision["temporal_patch_size"]),
              let mergeSize = positiveInteger(vision["spatial_merge_size"]),
              let inChannels = positiveInteger(vision["in_channels"] ?? 3),
              let rmsNormEps = (vision["rms_norm_eps"] as? NSNumber)?.doubleValue,
              rmsNormEps.isFinite, rmsNormEps > 0,
              let swigluLimit = (vision["swiglu_limit"] as? NSNumber)?.doubleValue,
              swigluLimit.isFinite, swigluLimit > 0,
              (vision["attention_bias"] as? Bool) == true,
              (vision["hidden_act"] as? String)?.lowercased() == "silu",
              depth > 0, intermediate > 0, projectionIntermediate > 0,
              patchSize > 0, temporalPatchSize > 0, mergeSize > 0,
              inChannels == 3,
              outHidden == textHidden,
              visionHidden.isMultiple(of: heads)
        else { return false }
        return (visionHidden / heads).isMultiple(of: 4)
    }

    private static func selectedProcessorClass(
        modelDirectory: URL,
        canonicalModelType: String,
        config: [String: Any]
    ) -> String? {
        let preprocessor = modelDirectory.appendingPathComponent(
            "preprocessor_config.json"
        )
        let processor = modelDirectory.appendingPathComponent("processor_config.json")
        let selectedURL = FileManager.default.fileExists(atPath: preprocessor.path)
            ? preprocessor
            : processor
        guard let data = try? Data(contentsOf: selectedURL),
              let baseConfig = try? JSONDecoder().decode(
                  BaseProcessorConfiguration.self,
                  from: data
              ),
              !baseConfig.processorClass.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return nil }

        if isQwenConditionalModelType(canonicalModelType) {
            guard let qwenProcessor = try? JSONDecoder().decode(
                Qwen3VLProcessorConfiguration.self,
                from: data
            ), isRuntimeCompatibleQwenProcessor(qwenProcessor, config: config)
            else { return nil }
            return "Qwen3VLProcessor"
        }
        if canonicalModelType == "glm5_next" {
            guard let glmProcessor = try? JSONDecoder().decode(
                GLM5NextProcessorConfiguration.self,
                from: data
            ), let rawObject = try? JSONSerialization.jsonObject(with: data),
               let rawProcessor = rawObject as? [String: Any],
               isRuntimeCompatibleGLMProcessor(
                glmProcessor, rawProcessor: rawProcessor, config: config)
            else { return nil }
            return "Glm5NextProcessor"
        }
        return baseConfig.processorClass
    }

    private static func isRuntimeCompatibleGLMProcessor(
        _ processor: GLM5NextProcessorConfiguration,
        rawProcessor: [String: Any],
        config: [String: Any]
    ) -> Bool {
        guard let vision = config["vision_config"] as? [String: Any],
              let inChannels = positiveInteger(vision["in_channels"] ?? 3),
              let patchSize = positiveInteger(vision["patch_size"]),
              let temporalPatchSize = positiveInteger(vision["temporal_patch_size"]),
              let spatialMergeSize = positiveInteger(vision["spatial_merge_size"]),
              inChannels == 3,
              processor.imageProcessor.imageMean.count == inChannels,
              processor.imageProcessor.imageStd.count == inChannels,
              processor.imageProcessor.imageMean.allSatisfy(\.isFinite),
              processor.imageProcessor.imageStd.allSatisfy({ $0.isFinite && $0 > 0 }),
              processor.imageProcessor.patchSize == patchSize,
              processor.imageProcessor.temporalPatchSize == temporalPatchSize,
              processor.imageProcessor.mergeSize == spatialMergeSize,
              processor.imageProcessor.minPixels > 0,
              processor.imageProcessor.maxPixels >= processor.imageProcessor.minPixels,
              multiplied(patchSize, by: spatialMergeSize) != nil,
              processor.processorClass == "Glm5NextProcessor",
              processor.videoProcessor != nil,
              let rawImage = rawProcessor["image_processor"] as? [String: Any],
              let rawVideo = rawProcessor["video_processor"] as? [String: Any],
              integer(rawImage["patch_expand_factor"] ?? 1) == 1,
              integer(rawVideo["patch_expand_factor"] ?? 1) == 1,
              positiveInteger(rawVideo["max_frames"] ?? 2_048) != nil
        else { return false }
        guard let video = processor.videoProcessor else { return false }
        return video.imageMean.count == inChannels
            && video.imageStd.count == inChannels
            && video.imageMean.allSatisfy(\.isFinite)
            && video.imageStd.allSatisfy({ $0.isFinite && $0 > 0 })
            && video.patchSize == patchSize
            && video.temporalPatchSize == temporalPatchSize
            && video.mergeSize == spatialMergeSize
            && video.minPixels > 0
            && video.maxPixels >= video.minPixels
            && (video.fps ?? 2).isFinite
            && (video.fps ?? 2) > 0
    }

    private static func isRuntimeCompatibleQwenProcessor(
        _ processor: Qwen3VLProcessorConfiguration,
        config: [String: Any]
    ) -> Bool {
        guard let vision = config["vision_config"] as? [String: Any],
              let inChannels = positiveInteger(vision["in_channels"] ?? 3),
              let patchSize = positiveInteger(vision["patch_size"]),
              let temporalPatchSize = positiveInteger(vision["temporal_patch_size"]),
              let spatialMergeSize = positiveInteger(vision["spatial_merge_size"]),
              inChannels == 3,
              processor.imageMean.count == inChannels,
              processor.imageStd.count == inChannels,
              processor.imageMean.allSatisfy(\.isFinite),
              processor.imageStd.allSatisfy({ $0.isFinite && $0 > 0 }),
              processor.patchSize == patchSize,
              processor.temporalPatchSize == temporalPatchSize,
              processor.mergeSize == spatialMergeSize,
              processor.minPixels > 0,
              processor.maxPixels >= processor.minPixels,
              multiplied(processor.patchSize, by: processor.mergeSize) != nil
        else { return false }
        return true
    }

    private static func visionTensorNames(
        in modelDirectory: URL,
        config: [String: Any],
        requiresCompleteQwenTower: Bool,
        requiresCompleteGLMTower: Bool
    ) -> Set<String> {
        let indexURL = modelDirectory.appendingPathComponent(
            "model.safetensors.index.json"
        )
        let tensors: [String: SafetensorEvidence.TensorMetadata]
        if FileManager.default.fileExists(atPath: indexURL.path) {
            guard let index = jsonObject(at: indexURL),
                  let rawWeightMap = index["weight_map"] as? [String: Any]
            else { return [] }
            let weightMap = rawWeightMap.compactMapValues { $0 as? String }
            guard weightMap.count == rawWeightMap.count, !weightMap.isEmpty else {
                return []
            }
            var shardEvidence: [String: SafetensorEvidence] = [:]
            for shardName in Set(weightMap.values) {
                let shardURL = modelDirectory.appendingPathComponent(shardName)
                guard shardURL.standardizedFileURL.deletingLastPathComponent()
                        == modelDirectory.standardizedFileURL,
                      let evidence = safetensorEvidence(in: shardURL)
                else { return [] }
                shardEvidence[shardName] = evidence
            }
            for (shardName, evidence) in shardEvidence {
                let indexedNames = Set(
                    weightMap.compactMap { name, mappedShard in
                        mappedShard == shardName ? name : nil
                    }
                )
                guard evidence.tensorNames == indexedNames else { return [] }
            }
            guard weightMap.allSatisfy({ tensorName, shardName in
                shardEvidence[shardName]?.tensors[tensorName] != nil
            }) else { return [] }
            guard !requiresCompleteGLMTower
                    || !weightMap.keys.contains(where: isUnsupportedGLMVisionTensorName)
            else { return [] }
            var discovered: [String: SafetensorEvidence.TensorMetadata] = [:]
            for (tensorName, shardName) in weightMap
            where isVisionTensorName(
                tensorName, requiresCompleteGLMTower: requiresCompleteGLMTower)
            {
                guard let metadata = shardEvidence[shardName]?.tensors[tensorName] else {
                    return []
                }
                let normalizedName = normalizedVisionTensorName(tensorName)
                guard discovered[normalizedName] == nil else { return [] }
                discovered[normalizedName] = metadata
            }
            tensors = discovered
        } else {
            guard let files = try? FileManager.default.contentsOfDirectory(
                at: modelDirectory,
                includingPropertiesForKeys: nil
            ) else { return [] }
            let weightFiles = files.filter {
                $0.pathExtension == "safetensors"
                    && $0.lastPathComponent != "mtp.safetensors"
            }
            guard !weightFiles.isEmpty else { return [] }
            var discovered: [String: SafetensorEvidence.TensorMetadata] = [:]
            for file in weightFiles {
                guard let evidence = safetensorEvidence(in: file) else { return [] }
                guard !requiresCompleteGLMTower
                        || !evidence.tensorNames.contains(
                            where: isUnsupportedGLMVisionTensorName)
                else { return [] }
                for (name, metadata) in evidence.tensors
                where isVisionTensorName(
                    name, requiresCompleteGLMTower: requiresCompleteGLMTower)
                {
                    let normalizedName = normalizedVisionTensorName(name)
                    guard discovered[normalizedName] == nil else { return [] }
                    discovered[normalizedName] = metadata
                }
            }
            tensors = discovered
        }

        if requiresCompleteQwenTower {
            guard let required = requiredQwenVisionTensorNames(config: config),
                  required.isSubset(of: Set(tensors.keys)),
                  hasCompleteQwenQuantizationCompanions(
                    required: required,
                    tensors: tensors,
                    config: config
                  )
            else { return [] }
        }
        if requiresCompleteGLMTower {
            guard let required = requiredGLM5NextVisionTensorNames(config: config),
                  required.isSubset(of: Set(tensors.keys)),
                  hasValidGLM5NextVisionTensorMetadata(
                    required: required,
                    tensors: tensors,
                    config: config)
            else { return [] }
        }
        return Set(tensors.keys)
    }

    private static func requiredGLM5NextVisionTensorNames(
        config: [String: Any]
    ) -> Set<String>? {
        guard let vision = config["vision_config"] as? [String: Any],
              let depth = positiveInteger(vision["depth"])
        else { return nil }
        let blockLeaves = [
            "attn.k_norm.weight", "attn.proj.bias", "attn.proj.weight",
            "attn.q_norm.weight", "attn.qkv.bias", "attn.qkv.weight",
            "mlp.down_proj.bias", "mlp.down_proj.weight", "mlp.gate_proj.bias",
            "mlp.gate_proj.weight", "mlp.up_proj.bias", "mlp.up_proj.weight",
            "norm1.weight", "norm2.weight",
        ]
        var required = Set((0..<depth).flatMap { block in
            blockLeaves.map { "vision_tower.blocks.\(block).\($0)" }
        })
        required.formUnion([
            "vision_tower.downsample.bias", "vision_tower.downsample.weight",
            "vision_tower.merger.down_proj.weight",
            "vision_tower.merger.gate_proj.weight",
            "vision_tower.merger.post_projection_norm.bias",
            "vision_tower.merger.post_projection_norm.weight",
            "vision_tower.merger.proj.weight", "vision_tower.merger.up_proj.weight",
            "vision_tower.patch_embed.proj.bias",
            "vision_tower.patch_embed.proj.weight",
            "vision_tower.post_layernorm.weight",
        ])
        return required
    }

    private static func isVisionTensorName(
        _ name: String,
        requiresCompleteGLMTower: Bool
    ) -> Bool {
        if requiresCompleteGLMTower {
            return name.hasPrefix("vision_model.") || name.hasPrefix("model.visual.")
        }
        return name.hasPrefix("vision_tower.") || name.hasPrefix("vision_model.")
            || name.hasPrefix("model.visual.")
    }

    private static func isUnsupportedGLMVisionTensorName(_ name: String) -> Bool {
        name.hasPrefix("vision_tower.") || name.hasPrefix("visual.")
    }

    private static func normalizedVisionTensorName(_ name: String) -> String {
        if name.hasPrefix("model.visual.") {
            return "vision_tower." + name.dropFirst("model.visual.".count)
        }
        if name.hasPrefix("vision_model.") {
            return "vision_tower." + name.dropFirst("vision_model.".count)
        }
        return name
    }

    private static func hasValidGLM5NextVisionTensorMetadata(
        required: Set<String>,
        tensors: [String: SafetensorEvidence.TensorMetadata],
        config: [String: Any]
    ) -> Bool {
        guard let shapes = expectedGLM5NextVisionTensorShapes(config: config),
              Set(shapes.keys) == required else { return false }
        let floatDTypes: Set<String> = ["BF16", "F16", "F32"]
        for name in required {
            guard let metadata = tensors[name],
                  floatDTypes.contains(metadata.dtype.uppercased()),
                  let alternatives = shapes[name],
                  alternatives.contains(metadata.shape)
            else { return false }
        }
        return true
    }

    private static func expectedGLM5NextVisionTensorShapes(
        config: [String: Any]
    ) -> [String: [[Int]]]? {
        guard let vision = config["vision_config"] as? [String: Any],
              let depth = positiveInteger(vision["depth"]),
              let hidden = positiveInteger(vision["hidden_size"]),
              let intermediate = positiveInteger(vision["intermediate_size"]),
              let outHidden = positiveInteger(vision["out_hidden_size"]),
              let heads = positiveInteger(vision["num_heads"]),
              hidden.isMultiple(of: heads),
              let projectionIntermediate = positiveInteger(
                vision["projection_intermediate_size"]),
              let inChannels = positiveInteger(vision["in_channels"] ?? 3),
              let patch = positiveInteger(vision["patch_size"]),
              let temporal = positiveInteger(vision["temporal_patch_size"]),
              let merge = positiveInteger(vision["spatial_merge_size"]),
              let tripleHidden = multiplied(hidden, by: 3)
        else { return nil }
        let headWidth = hidden / heads
        var shapes: [String: [[Int]]] = [
            "vision_tower.patch_embed.proj.weight": [
                [hidden, temporal, patch, patch, inChannels],
                [hidden, inChannels, temporal, patch, patch],
            ],
            "vision_tower.patch_embed.proj.bias": [[hidden]],
            "vision_tower.downsample.weight": [
                [outHidden, merge, merge, hidden],
                [outHidden, hidden, merge, merge],
            ],
            "vision_tower.downsample.bias": [[outHidden]],
            "vision_tower.post_layernorm.weight": [[hidden]],
            "vision_tower.merger.proj.weight": [[outHidden, outHidden]],
            "vision_tower.merger.post_projection_norm.weight": [[outHidden]],
            "vision_tower.merger.post_projection_norm.bias": [[outHidden]],
            "vision_tower.merger.gate_proj.weight": [
                [projectionIntermediate, outHidden],
            ],
            "vision_tower.merger.up_proj.weight": [
                [projectionIntermediate, outHidden],
            ],
            "vision_tower.merger.down_proj.weight": [
                [outHidden, projectionIntermediate],
            ],
        ]
        for block in 0..<depth {
            let prefix = "vision_tower.blocks.\(block)"
            shapes["\(prefix).attn.q_norm.weight"] = [[headWidth]]
            shapes["\(prefix).attn.k_norm.weight"] = [[headWidth]]
            shapes["\(prefix).attn.qkv.weight"] = [[tripleHidden, hidden]]
            shapes["\(prefix).attn.qkv.bias"] = [[tripleHidden]]
            shapes["\(prefix).attn.proj.weight"] = [[hidden, hidden]]
            shapes["\(prefix).attn.proj.bias"] = [[hidden]]
            shapes["\(prefix).mlp.gate_proj.weight"] = [[intermediate, hidden]]
            shapes["\(prefix).mlp.gate_proj.bias"] = [[intermediate]]
            shapes["\(prefix).mlp.up_proj.weight"] = [[intermediate, hidden]]
            shapes["\(prefix).mlp.up_proj.bias"] = [[intermediate]]
            shapes["\(prefix).mlp.down_proj.weight"] = [[hidden, intermediate]]
            shapes["\(prefix).mlp.down_proj.bias"] = [[hidden]]
            shapes["\(prefix).norm1.weight"] = [[hidden]]
            shapes["\(prefix).norm2.weight"] = [[hidden]]
        }
        return shapes
    }

    private static func requiredQwenVisionTensorNames(
        config: [String: Any]
    ) -> Set<String>? {
        guard let vision = config["vision_config"] as? [String: Any],
              let depth = integer(vision["depth"]), depth > 0
        else { return nil }
        let deepstackIndexes = (vision["deepstack_visual_indexes"] as? [Any])?
            .compactMap(integer) ?? []
        guard deepstackIndexes.allSatisfy({ $0 >= 0 && $0 < depth }) else {
            return nil
        }

        var required: Set<String> = [
            "vision_tower.patch_embed.proj.weight",
            "vision_tower.patch_embed.proj.bias",
            "vision_tower.pos_embed.weight",
        ]
        let blockSuffixes = [
            "attn.proj.bias", "attn.proj.weight", "attn.qkv.bias", "attn.qkv.weight",
            "mlp.linear_fc1.bias", "mlp.linear_fc1.weight",
            "mlp.linear_fc2.bias", "mlp.linear_fc2.weight",
            "norm1.bias", "norm1.weight", "norm2.bias", "norm2.weight",
        ]
        for block in 0..<depth {
            for suffix in blockSuffixes {
                required.insert("vision_tower.blocks.\(block).\(suffix)")
            }
        }
        let mergerSuffixes = [
            "linear_fc1.bias", "linear_fc1.weight",
            "linear_fc2.bias", "linear_fc2.weight", "norm.bias", "norm.weight",
        ]
        for suffix in mergerSuffixes {
            required.insert("vision_tower.merger.\(suffix)")
        }
        for index in deepstackIndexes.indices {
            for suffix in mergerSuffixes {
                required.insert("vision_tower.deepstack_merger_list.\(index).\(suffix)")
            }
        }
        return required
    }

    private static func hasCompleteQwenQuantizationCompanions(
        required: Set<String>,
        tensors: [String: SafetensorEvidence.TensorMetadata],
        config: [String: Any]
    ) -> Bool {
        guard let expectedShapes = expectedQwenVisionTensorShapes(config: config),
              required.allSatisfy({ expectedShapes[$0] != nil })
        else { return false }

        let quantization = (config["quantization_config"] as? [String: Any])
            ?? (config["quantization"] as? [String: Any])
        let mode = (quantization?["mode"] as? String)?.lowercased()
        let isMXFP = mode == "mxfp4" || mode == "mxfp8"
        let patchEmbeddingProvenance = (config["vision_patch_embedding_layout"] as? String)
            .flatMap(Qwen3_5VisionPatchEmbeddingLayout.init(rawValue:))

        for tensorName in required where !tensorName.hasSuffix(".weight") {
            guard tensors[tensorName]?.shape == expectedShapes[tensorName] else {
                return false
            }
        }

        for weightName in required where weightName.hasSuffix(".weight") {
            guard let metadata = tensors[weightName],
                  let logicalShape = expectedShapes[weightName]
            else { return false }
            let base = String(weightName.dropLast(".weight".count))
            let scales = tensors["\(base).scales"]
            let biases = tensors["\(base).biases"]
            let hasScales = scales != nil
            let hasBiases = biases != nil
            let packedWeight = ["U8", "U16", "U32", "I8", "I16", "I32"]
                .contains(metadata.dtype.uppercased())
            let hasQuantizedRepresentation = packedWeight || hasScales || hasBiases
            guard !hasQuantizedRepresentation || quantization != nil else {
                return false
            }
            guard hasQuantizedRepresentation else {
                guard matchesUnquantizedQwenShape(
                    metadata.shape,
                    logicalShape: logicalShape,
                    tensorName: weightName,
                    patchEmbeddingProvenance: patchEmbeddingProvenance
                ) else { return false }
                continue
            }
            guard metadata.dtype.uppercased() == "U32",
                  let bits = integer(quantization?["bits"]),
                  let groupSize = integer(quantization?["group_size"]),
                  let packedShape = quantizedPackedShape(
                    logicalShape: logicalShape,
                    bits: bits
                  ),
                  let companionShape = quantizedCompanionShape(
                    logicalShape: logicalShape,
                    groupSize: groupSize
                  ),
                  metadata.shape == packedShape,
                  scales?.shape == companionShape
            else { return false }
            if isMXFP {
                let expectedBits = mode == "mxfp8" ? 8 : 4
                let scaleDType = scales?.dtype.uppercased()
                guard bits == expectedBits,
                      groupSize == 32,
                      scaleDType == "U8" || scaleDType == "F8_E8M0",
                      !hasBiases
                else { return false }
            } else {
                guard biases?.shape == companionShape else { return false }
            }
        }
        return true
    }

    private static func matchesUnquantizedQwenShape(
        _ shape: [Int],
        logicalShape: [Int],
        tensorName: String,
        patchEmbeddingProvenance: Qwen3_5VisionPatchEmbeddingLayout?
    ) -> Bool {
        guard tensorName == "vision_tower.patch_embed.proj.weight" else {
            return shape == logicalShape
        }
        guard logicalShape.count == 5 else { return false }

        guard logicalShape[2] == logicalShape[3] else { return false }
        return Qwen3_5VisionPatchEmbeddingLayout.classify(
            shape: shape,
            outputChannels: logicalShape[0],
            inputChannels: logicalShape[4],
            temporalPatchSize: logicalShape[1],
            patchSize: logicalShape[2],
            trustedProvenance: patchEmbeddingProvenance
        ) != nil
    }

    private static func expectedQwenVisionTensorShapes(
        config: [String: Any]
    ) -> [String: [Int]]? {
        guard let vision = config["vision_config"] as? [String: Any],
              let depth = positiveInteger(vision["depth"]),
              let hidden = positiveInteger(vision["hidden_size"]),
              let intermediate = positiveInteger(vision["intermediate_size"]),
              let outHidden = positiveInteger(vision["out_hidden_size"]),
              let inChannels = positiveInteger(vision["in_channels"] ?? 3),
              let patchSize = positiveInteger(vision["patch_size"]),
              let temporalPatchSize = positiveInteger(vision["temporal_patch_size"]),
              let positionCount = positiveInteger(vision["num_position_embeddings"]),
              let spatialMergeSize = positiveInteger(vision["spatial_merge_size"]),
              let tripleHidden = multiplied(hidden, by: 3),
              let spatialMergeArea = multiplied(spatialMergeSize, by: spatialMergeSize),
              let mergedHidden = multiplied(hidden, by: spatialMergeArea)
        else { return nil }

        let deepstackIndexes = (vision["deepstack_visual_indexes"] as? [Any])?
            .compactMap(integer) ?? []
        guard deepstackIndexes.allSatisfy({ $0 >= 0 && $0 < depth }) else {
            return nil
        }

        var shapes: [String: [Int]] = [
            "vision_tower.patch_embed.proj.weight": [
                hidden, temporalPatchSize, patchSize, patchSize, inChannels,
            ],
            "vision_tower.patch_embed.proj.bias": [hidden],
            "vision_tower.pos_embed.weight": [positionCount, hidden],
        ]
        for block in 0..<depth {
            let prefix = "vision_tower.blocks.\(block)"
            shapes["\(prefix).attn.proj.weight"] = [hidden, hidden]
            shapes["\(prefix).attn.proj.bias"] = [hidden]
            shapes["\(prefix).attn.qkv.weight"] = [tripleHidden, hidden]
            shapes["\(prefix).attn.qkv.bias"] = [tripleHidden]
            shapes["\(prefix).mlp.linear_fc1.weight"] = [intermediate, hidden]
            shapes["\(prefix).mlp.linear_fc1.bias"] = [intermediate]
            shapes["\(prefix).mlp.linear_fc2.weight"] = [hidden, intermediate]
            shapes["\(prefix).mlp.linear_fc2.bias"] = [hidden]
            shapes["\(prefix).norm1.weight"] = [hidden]
            shapes["\(prefix).norm1.bias"] = [hidden]
            shapes["\(prefix).norm2.weight"] = [hidden]
            shapes["\(prefix).norm2.bias"] = [hidden]
        }

        addQwenMergerShapes(
            prefix: "vision_tower.merger",
            normSize: hidden,
            mergedHidden: mergedHidden,
            outHidden: outHidden,
            to: &shapes
        )
        for index in deepstackIndexes.indices {
            addQwenMergerShapes(
                prefix: "vision_tower.deepstack_merger_list.\(index)",
                normSize: mergedHidden,
                mergedHidden: mergedHidden,
                outHidden: outHidden,
                to: &shapes
            )
        }
        return shapes
    }

    private static func addQwenMergerShapes(
        prefix: String,
        normSize: Int,
        mergedHidden: Int,
        outHidden: Int,
        to shapes: inout [String: [Int]]
    ) {
        shapes["\(prefix).norm.weight"] = [normSize]
        shapes["\(prefix).norm.bias"] = [normSize]
        shapes["\(prefix).linear_fc1.weight"] = [mergedHidden, mergedHidden]
        shapes["\(prefix).linear_fc1.bias"] = [mergedHidden]
        shapes["\(prefix).linear_fc2.weight"] = [outHidden, mergedHidden]
        shapes["\(prefix).linear_fc2.bias"] = [outHidden]
    }

    private static func quantizedPackedShape(
        logicalShape: [Int],
        bits: Int
    ) -> [Int]? {
        guard bits > 0, bits <= 32, 32.isMultiple(of: bits),
              let last = logicalShape.last,
              last.isMultiple(of: 32 / bits)
        else { return nil }
        var shape = logicalShape
        shape[shape.count - 1] = last / (32 / bits)
        return shape
    }

    private static func quantizedCompanionShape(
        logicalShape: [Int],
        groupSize: Int
    ) -> [Int]? {
        guard groupSize > 0,
              let last = logicalShape.last,
              last.isMultiple(of: groupSize)
        else { return nil }
        var shape = logicalShape
        shape[shape.count - 1] = last / groupSize
        return shape
    }

    private static func positiveInteger(_ value: Any?) -> Int? {
        guard let value = integer(value), value > 0 else { return nil }
        return value
    }

    private static func multiplied(_ lhs: Int, by rhs: Int) -> Int? {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        return overflow ? nil : result
    }

    private static func safetensorEvidence(in url: URL) -> SafetensorEvidence? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let prefix = try? handle.read(upToCount: 8),
              prefix.count == 8
        else { return nil }

        let headerSize = prefix.enumerated().reduce(UInt64(0)) { result, item in
            result | (UInt64(item.element) << UInt64(item.offset * 8))
        }
        guard headerSize <= UInt64(maximumQualificationMetadataBytes),
              let header = try? handle.read(upToCount: Int(headerSize)),
              header.count == Int(headerSize),
              let object = try? JSONSerialization.jsonObject(with: header) as? [String: Any],
              let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let fileSize = (attributes[.size] as? NSNumber)?.uint64Value
        else { return nil }
        let payloadStart = UInt64(8) + headerSize
        guard payloadStart <= fileSize else { return nil }

        var tensors: [String: SafetensorEvidence.TensorMetadata] = [:]
        var ranges: [(start: UInt64, end: UInt64)] = []
        for (name, rawMetadata) in object where name != "__metadata__" {
            guard let metadata = rawMetadata as? [String: Any],
                  let dtype = metadata["dtype"] as? String,
                  let rawShape = metadata["shape"] as? [Any],
                  let shape = integerShape(rawShape),
                  let offsets = metadata["data_offsets"] as? [Any],
                  offsets.count == 2,
                  let start = unsignedInteger(offsets[0]),
                  let end = unsignedInteger(offsets[1]),
                  start <= end,
                  let expectedBytes = tensorByteCount(dtype: dtype, shape: shape),
                  end - start == expectedBytes
            else { return nil }
            ranges.append((start, end))
            tensors[name] = SafetensorEvidence.TensorMetadata(
                dtype: dtype,
                shape: shape
            )
        }
        let sortedRanges = ranges.sorted {
            $0.start == $1.start ? $0.end < $1.end : $0.start < $1.start
        }
        guard !tensors.isEmpty, sortedRanges.first?.start == 0 else { return nil }
        for (previous, next) in zip(sortedRanges, sortedRanges.dropFirst()) {
            guard previous.end == next.start else { return nil }
        }
        guard sortedRanges.last?.end == fileSize - payloadStart else { return nil }
        return SafetensorEvidence(tensors: tensors)
    }

    private static func jsonObject(at url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url),
              data.count <= maximumQualificationMetadataBytes else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    private static func integer(_ value: Any?) -> Int? {
        if let value = value as? Int { return value }
        return (value as? NSNumber)?.intValue
    }

    private static func unsignedInteger(_ value: Any?) -> UInt64? {
        guard let number = value as? NSNumber, number.int64Value >= 0 else {
            return nil
        }
        return number.uint64Value
    }

    private static func integerShape(_ values: [Any]) -> [Int]? {
        let shape = values.compactMap(integer)
        guard shape.count == values.count,
              shape.allSatisfy({ $0 > 0 })
        else { return nil }
        return shape
    }

    private static func tensorByteCount(dtype: String, shape: [Int]) -> UInt64? {
        let byteWidth: UInt64
        switch dtype.uppercased() {
        case "BOOL", "I8", "U8", "F8_E4M3", "F8_E5M2", "F8_E8M0":
            byteWidth = 1
        case "I16", "U16", "F16", "BF16":
            byteWidth = 2
        case "I32", "U32", "F32":
            byteWidth = 4
        case "I64", "U64", "F64":
            byteWidth = 8
        default:
            return nil
        }
        var elements: UInt64 = 1
        for dimension in shape {
            let (next, overflow) = elements.multipliedReportingOverflow(by: UInt64(dimension))
            guard !overflow else { return nil }
            elements = next
        }
        let (bytes, overflow) = elements.multipliedReportingOverflow(by: byteWidth)
        return overflow ? nil : bytes
    }

    private func snapshotFingerprint(for modelDirectory: URL) -> SnapshotFingerprint? {
        let directory = modelDirectory.standardizedFileURL.path
        let keys: Set<URLResourceKey> = [.fileSizeKey, .contentModificationDateKey]
        let files = (try? FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: Array(keys),
            options: [.skipsHiddenFiles]
        )) ?? []
        var evidence: [SnapshotFingerprint.FileEvidence] = []
        for url in files.filter(Self.isQualificationInput) {
            guard let values = try? url.resourceValues(forKeys: keys),
                  let signature = Self.qualificationContentSignature(for: url)
            else { return nil }
            evidence.append(
                SnapshotFingerprint.FileEvidence(
                    name: url.lastPathComponent,
                    size: Int64(values.fileSize ?? 0),
                    modifiedAt: values.contentModificationDate?.timeIntervalSince1970 ?? 0,
                    contentSignature: signature
                )
            )
        }
        evidence.sort { $0.name < $1.name }
        return SnapshotFingerprint(directory: directory, files: evidence)
    }

    private static func isQualificationInput(_ url: URL) -> Bool {
        switch url.lastPathComponent {
        case "config.json", "preprocessor_config.json", "processor_config.json",
             "model.safetensors.index.json":
            return true
        default:
            return url.pathExtension == "safetensors"
        }
    }

    /// Hash only metadata that qualification reads: complete small JSON files
    /// and the bounded safetensor header. Weight payloads remain untouched.
    private static func qualificationContentSignature(for url: URL) -> UInt64? {
        let data: Data
        if url.pathExtension == "safetensors" {
            guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
            defer { try? handle.close() }
            guard let prefix = try? handle.read(upToCount: 8),
                  prefix.count == 8 else { return nil }
            let headerLength = prefix.enumerated().reduce(UInt64(0)) { value, element in
                value | (UInt64(element.element) << UInt64(element.offset * 8))
            }
            guard headerLength <= UInt64(maximumQualificationMetadataBytes),
                  let header = try? handle.read(upToCount: Int(headerLength)),
                  header.count == Int(headerLength) else { return nil }
            data = prefix + header
        } else {
            guard let candidate = try? Data(contentsOf: url),
                  candidate.count <= maximumQualificationMetadataBytes else { return nil }
            data = candidate
        }
        return data.reduce(UInt64(14_695_981_039_346_656_037)) { hash, byte in
            (hash ^ UInt64(byte)) &* 1_099_511_628_211
        }
    }

    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }
}
