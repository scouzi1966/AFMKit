import CryptoKit
import Foundation
import MLX

/// Converts the official `zai-org/GLM-5.3-Flash` raw FP8 checkpoint into the
/// affine-quantized, multimodal MLX layout consumed by AFMKit.
///
/// The converter is deliberately local-only. It never downloads model data,
/// and it requires a pinned 40-character source revision so conversion output
/// cannot silently mix checkpoint revisions.
public struct GLM5NextCheckpointConverter {
    public typealias ProgressHandler = (String) -> Void

    public static let officialModelID = "zai-org/GLM-5.3-Flash"
    public static let minimumDestinationFreeBytes: Int64 = 600_000_000_000
    private static let currentFormatVersion = 2
    private static let fp8BlockRows = 128
    private static let fp8BlockColumns = 128
    private static let affineGroupSize = 64
    private static let affineBits = 4
    private static let requiredSupportFiles = [
        "chat_template.jinja", "processor_config.json", "tokenizer.json",
        "tokenizer_config.json",
    ]

    public enum Profile: String, Codable, CaseIterable, Sendable {
        case mlxAffine4 = "mlx-affine-4"
    }

    public struct Inspection: Sendable, Equatable {
        public let modelType: String
        public let sourceRevision: String
        public let sourceBytes: Int64
        public let estimatedOutputBytes: Int64
        public let requiredDestinationFreeBytes: Int64
        public let shardCount: Int
        public let tensorCount: Int
        public let visionTensorCount: Int
        public let fp8TensorCount: Int
        public let omittedMTPTensorCount: Int
    }

    /// Provider-owned validation of bytes that an interrupted conversion can
    /// safely credit during destination-capacity preflight.
    public struct ResumeInspection: Sendable, Equatable {
        public let sourceRevision: String
        public let verifiedCompletedOutputBytes: Int64
    }

    public enum ConversionError: LocalizedError, Equatable {
        case invalidSource(String)
        case unsafeOutput(String)
        case outputExists(String)
        case sourceMismatch(String)
        case unsupportedTensor(String)

        public var errorDescription: String? {
            switch self {
            case .invalidSource(let message), .unsafeOutput(let message),
                 .outputExists(let message), .sourceMismatch(let message),
                 .unsupportedTensor(let message):
                message
            }
        }
    }

    private struct Quantization: Codable, Equatable {
        let groupSize: Int
        let bits: Int
        let mode: String

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
            case mode
        }
    }

    private struct SourceShardFingerprint: Codable, Equatable {
        let size: Int64
        let modificationTime: TimeInterval
        let contentSHA256: String?
    }

    private struct CompletedUnit: Codable {
        let outputFile: String
        let outputSize: Int64
        let outputSHA256: String
        let outputKeys: [String]
    }

    private struct State: Codable {
        var formatVersion = GLM5NextCheckpointConverter.currentFormatVersion
        var profile = Profile.mlxAffine4.rawValue
        var sourceModelID = GLM5NextCheckpointConverter.officialModelID
        var sourceRevision: String
        var configSHA256: String
        var indexSHA256: String
        var sourceShards: [String: SourceShardFingerprint]
        var sourceAssets: [String: String]
        var completed: [String: CompletedUnit] = [:]
        var weightMap: [String: String] = [:]
        var quantization: [String: Quantization] = [:]
        var skippedQuantization: [String] = []
        var omittedMTP: [String] = []
    }

    private struct Configuration {
        let object: [String: Any]
        let hiddenLayers: Int
        let routedExperts: Int
        let kvLoraRank: Int
        let attentionHeads: Int
        let qkNopeHeadDim: Int
        let vHeadDim: Int
        let sourceMTPCount: Int
    }

    struct TensorReference: Equatable {
        let name: String
        let shard: String
        let dtype: AFMSafetensorHeader.DType
        let shape: [Int]
        let byteCount: Int64
    }

    private struct SourceCheckpoint {
        let root: URL
        let configURL: URL
        let indexURL: URL
        let configData: Data
        let indexData: Data
        let config: Configuration
        let revision: String
        let tensors: [String: TensorReference]
        let shardURLs: [String: URL]
        let shardFingerprints: [String: SourceShardFingerprint]
        let assetSHA256: [String: String]
        let sourceBytes: Int64
    }

    private enum ConversionUnit {
        case ordinary(id: String, outputFile: String, tensors: [TensorReference])
        case experts(
            id: String,
            outputFile: String,
            layer: Int,
            projection: String,
            tensors: [TensorReference]
        )

        var id: String {
            switch self {
            case .ordinary(let id, _, _), .experts(let id, _, _, _, _): id
            }
        }

        var outputFile: String {
            switch self {
            case .ordinary(_, let outputFile, _),
                 .experts(_, let outputFile, _, _, _): outputFile
            }
        }
    }

    private struct ConvertedUnit {
        var arrays: [String: MLXArray]
        var quantization: [String: Quantization]
        var skippedQuantization: Set<String>
    }

    private final class TensorLoader {
        let root: URL
        private var loadedShard: String?
        private var arrays: [String: MLXArray] = [:]

        init(root: URL) {
            self.root = root
        }

        func load(_ reference: TensorReference) throws -> MLXArray {
            if loadedShard != reference.shard {
                arrays = try loadArrays(url: root.appendingPathComponent(reference.shard))
                loadedShard = reference.shard
            }
            guard let value = arrays[reference.name] else {
                throw ConversionError.invalidSource(
                    "Tensor \(reference.name) is missing from \(reference.shard).")
            }
            return value
        }

        func release() {
            arrays.removeAll(keepingCapacity: false)
            loadedShard = nil
            Memory.clearCache()
        }
    }

    let source: URL
    let output: URL
    let overwrite: Bool
    let profile: Profile
    let sourceRevision: String?
    let progress: ProgressHandler?

    private var stateURL: URL {
        output.standardizedFileURL.appendingPathComponent(".afm-mlx-conversion.json")
    }

    public init(
        source: URL,
        output: URL,
        overwrite: Bool = false,
        profile: Profile = .mlxAffine4,
        sourceRevision: String? = nil,
        progress: ProgressHandler? = nil
    ) {
        self.source = source
        self.output = output
        self.overwrite = overwrite
        self.profile = profile
        self.sourceRevision = sourceRevision
        self.progress = progress
    }

    public static func inspect(
        source: URL,
        sourceRevision: String? = nil
    ) throws -> Inspection {
        let checkpoint = try loadSource(
            source, explicitRevision: sourceRevision, fingerprintContents: false)
        return inspection(for: checkpoint)
    }

    /// Validates the complete source identity, conversion plan, private resume
    /// manifest, and completed SafeTensor outputs before returning resumable
    /// bytes. Source shards are hashed in a streaming pass; no tensor payload is
    /// materialized by this inspection.
    public static func inspectResume(
        source: URL,
        output: URL,
        profile: Profile = .mlxAffine4,
        sourceRevision: String? = nil
    ) throws -> ResumeInspection {
        let paths = try validatedPaths(source: source, output: output)
        let stateURL = paths.output.appendingPathComponent(".afm-mlx-conversion.json")
        guard FileManager.default.fileExists(atPath: stateURL.path) else {
            // A new destination has nothing to credit. Validate/infer source
            // provenance without hashing the complete ~328 GB checkpoint; run()
            // performs the authoritative content fingerprint before writing.
            let checkpoint = try loadSource(
                paths.source, explicitRevision: sourceRevision, fingerprintContents: false)
            return ResumeInspection(
                sourceRevision: checkpoint.revision,
                verifiedCompletedOutputBytes: 0)
        }
        let checkpoint = try loadSource(
            paths.source, explicitRevision: sourceRevision, fingerprintContents: true)
        let state: State
        do {
            state = try JSONDecoder().decode(State.self, from: Data(contentsOf: stateURL))
        } catch {
            throw ConversionError.sourceMismatch(
                "The existing conversion resume manifest is invalid; use --overwrite after verifying the destination.")
        }
        let plan = try makePlan(checkpoint: checkpoint)
        try validate(
            state: state,
            checkpoint: checkpoint,
            configHash: sha256(checkpoint.configData),
            indexHash: sha256(checkpoint.indexData),
            profile: profile,
            plan: plan)
        return ResumeInspection(
            sourceRevision: checkpoint.revision,
            verifiedCompletedOutputBytes: try verifiedCompletedBytes(
                state: state, plan: plan, checkpoint: checkpoint, output: paths.output))
    }

    public func run() throws {
        let fm = FileManager.default
        let paths = try Self.validatedPaths(source: source, output: output)
        let sourceURL = paths.source
        let outputURL = paths.output

        try MLXMetalLibrary.ensureAvailable(verbose: true)

        let checkpoint = try Self.loadSource(
            sourceURL, explicitRevision: sourceRevision, fingerprintContents: true)
        let inspection = Self.inspection(for: checkpoint)
        report("Converting \(Self.officialModelID) at \(inspection.sourceRevision)")
        report("  source: \(sourceURL.path)")
        report("  output: \(outputURL.path)")
        report("  profile: \(profile.rawValue)")
        report("  source shards: \(inspection.shardCount)")
        report("  MTP tensors omitted: \(inspection.omittedMTPTensorCount)")

        if overwrite, fm.fileExists(atPath: outputURL.path) {
            try fm.removeItem(at: outputURL)
        }
        if fm.fileExists(atPath: outputURL.path), !fm.fileExists(atPath: stateURL.path) {
            let contents = try fm.contentsOfDirectory(atPath: outputURL.path)
            guard contents.isEmpty else {
                throw ConversionError.outputExists(
                    "Output exists but is not a resumable AFM conversion. Use --overwrite to replace it.")
            }
        }
        try fm.createDirectory(at: outputURL, withIntermediateDirectories: true)

        let configHash = Self.sha256(checkpoint.configData)
        let indexHash = Self.sha256(checkpoint.indexData)
        let plan = try Self.makePlan(checkpoint: checkpoint)
        var state: State
        if fm.fileExists(atPath: stateURL.path) {
            state = try JSONDecoder().decode(State.self, from: Data(contentsOf: stateURL))
            try Self.validate(
                state: state,
                checkpoint: checkpoint,
                configHash: configHash,
                indexHash: indexHash,
                profile: profile,
                plan: plan)
        } else {
            state = State(
                profile: profile.rawValue,
                sourceRevision: checkpoint.revision,
                configSHA256: configHash,
                indexSHA256: indexHash,
                sourceShards: checkpoint.shardFingerprints,
                sourceAssets: checkpoint.assetSHA256)
            try saveState(state)
        }

        let loader = TensorLoader(root: checkpoint.root)
        for (position, unit) in plan.enumerated() {
            let destination = outputURL.appendingPathComponent(unit.outputFile)
            if let completed = state.completed[unit.id],
               completed.outputFile == unit.outputFile,
               try Self.isCompletedOutputValid(completed, at: destination)
            {
                report("[\(position + 1)/\(plan.count)] \(unit.id): already converted")
                continue
            }

            report("[\(position + 1)/\(plan.count)] \(unit.id): converting")
            let converted: ConvertedUnit
            switch unit {
            case .ordinary(_, _, let references):
                converted = try convertOrdinary(
                    references, checkpoint: checkpoint, loader: loader)
            case .experts(_, _, let layer, let projection, let references):
                converted = try convertExperts(
                    layer: layer,
                    projection: projection,
                    references: references,
                    checkpoint: checkpoint,
                    loader: loader)
            }
            guard !converted.arrays.isEmpty else {
                throw ConversionError.unsupportedTensor(
                    "Conversion unit \(unit.id) produced no output tensors.")
            }

            let partial = outputURL.appendingPathComponent(
                ".\(unit.outputFile).partial.safetensors")
            try? fm.removeItem(at: partial)
            try save(
                arrays: converted.arrays,
                metadata: [
                    "format": "mlx",
                    "afm_source_revision": checkpoint.revision,
                    "afm_conversion_profile": profile.rawValue,
                ],
                url: partial)
            try replaceOrMove(partial, to: destination)

            let outputSize = Int64(
                try destination.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
            let outputKeys = converted.arrays.keys.sorted()
            for key in outputKeys {
                state.weightMap[key] = unit.outputFile
            }
            state.quantization.merge(converted.quantization) { _, new in new }
            state.skippedQuantization = Array(
                Set(state.skippedQuantization).union(converted.skippedQuantization))
                .sorted()
            state.completed[unit.id] = CompletedUnit(
                outputFile: unit.outputFile,
                outputSize: outputSize,
                outputSHA256: try Self.sha256File(destination),
                outputKeys: outputKeys)
            try saveState(state)
            loader.release()
        }

        state.omittedMTP = checkpoint.tensors.keys.filter {
            Self.layerIndex(in: $0).map { $0 >= checkpoint.config.hiddenLayers } ?? false
        }.sorted()
        try copySupportFiles(from: checkpoint.root, to: outputURL)
        try writeOutputConfiguration(checkpoint: checkpoint, state: state)
        try writeOutputIndex(state: state, output: outputURL)
        try saveState(state)

        report("Conversion complete: \(outputURL.path)")
        report("Run: afm mlx -m \(outputURL.path)")
    }

    static func dequantizeFP8(
        raw: MLXArray,
        scaleInverse: MLXArray,
        weightShape: [Int],
        scaleShape: [Int],
        blockRows: Int = fp8BlockRows,
        blockColumns: Int = fp8BlockColumns
    ) throws -> MLXArray {
        guard raw.dtype == .uint8 else {
            throw ConversionError.unsupportedTensor(
                "FP8 storage must be exposed by MLX as UInt8.")
        }
        guard scaleInverse.dtype == .float32 else {
            throw ConversionError.unsupportedTensor(
                "FP8 inverse scales must use F32 storage.")
        }
        guard weightShape.count == 2, scaleShape.count == 2,
              raw.shape == weightShape, scaleInverse.shape == scaleShape
        else {
            throw ConversionError.unsupportedTensor(
                "FP8 block conversion requires matching two-dimensional weight and scale shapes.")
        }
        let rows = weightShape[0]
        let columns = weightShape[1]
        let expectedScaleShape = [
            (rows + blockRows - 1) / blockRows,
            (columns + blockColumns - 1) / blockColumns,
        ]
        guard scaleShape == expectedScaleShape else {
            throw ConversionError.unsupportedTensor(
                "FP8 inverse-scale shape \(scaleShape) does not match 128x128 blocks for \(weightShape).")
        }

        let finiteScales = isFinite(scaleInverse).all()
        MLX.eval(finiteScales)
        guard finiteScales.item(Bool.self) else {
            throw ConversionError.unsupportedTensor(
                "FP8 inverse scales contain non-finite values.")
        }

        var expanded = repeated(scaleInverse, count: blockRows, axis: 0)
        expanded = repeated(expanded, count: blockColumns, axis: 1)
        expanded = expanded[0..<rows, 0..<columns]
        let decoded = MLX.fromFP8(raw, dtype: .float32)
        return decoded * expanded.asType(.float32)
    }

    static func affineQuantize(
        _ weight: MLXArray,
        groupSize: Int = affineGroupSize,
        bits: Int = affineBits
    ) throws -> (weight: MLXArray, scales: MLXArray, biases: MLXArray) {
        guard weight.ndim >= 2, let columns = weight.shape.last,
              columns.isMultiple(of: groupSize)
        else {
            throw ConversionError.unsupportedTensor(
                "Affine quantization requires a rank >= 2 weight whose last dimension is divisible by \(groupSize).")
        }
        let leading = weight.shape.dropLast().reduce(1, *)
        let source = weight.asType(.bfloat16).reshaped([leading, columns])
        let quantized = MLX.quantized(
            source, groupSize: groupSize, bits: bits, mode: .affine)
        guard let biases = quantized.biases else {
            throw ConversionError.unsupportedTensor(
                "MLX affine quantization did not produce bias sidecars.")
        }
        let prefix = Array(weight.shape.dropLast())
        let output = quantized.wq.reshaped(prefix + [quantized.wq.dim(-1)])
        let scales = quantized.scales.reshaped(prefix + [quantized.scales.dim(-1)])
        let reshapedBiases = biases.reshaped(prefix + [biases.dim(-1)])
        MLX.eval(output, scales, reshapedBiases)
        return (output, scales, reshapedBiases)
    }

    static func mappedName(_ sourceName: String) -> String {
        var key = sourceName
        if key.hasPrefix("model.language_model.") {
            key = "language_model.model." + key.dropFirst("model.language_model.".count)
        } else if key.hasPrefix("lm_head.") {
            key = "language_model.lm_head." + key.dropFirst("lm_head.".count)
        } else if key.hasPrefix("model.visual.") {
            key = "vision_model." + key.dropFirst("model.visual.".count)
        }
        key = key
            .replacingOccurrences(of: ".hc_attn_", with: ".attn_hc.")
            .replacingOccurrences(of: ".hc_ffn_", with: ".ffn_hc.")
        for name in ["A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight"] {
            let suffix = ".self_attn.\(name)"
            if key.hasSuffix(suffix) {
                key = String(key.dropLast(name.count)) + "forget_gate." + name
                break
            }
        }
        return key
    }

    static func visionWeightPermutation(
        sourceName: String,
        sourceShape: [Int]
    ) throws -> [Int]? {
        switch sourceName {
        case "model.visual.patch_embed.proj.weight":
            guard sourceShape == [1024, 3, 2, 14, 14] else {
                throw ConversionError.unsupportedTensor(
                    "Unexpected GLM-5.3 patch embedding shape \(sourceShape).")
            }
            return [0, 2, 3, 4, 1]
        case "model.visual.downsample.weight":
            guard sourceShape == [4096, 1024, 2, 2] else {
                throw ConversionError.unsupportedTensor(
                    "Unexpected GLM-5.3 vision downsample shape \(sourceShape).")
            }
            return [0, 2, 3, 1]
        default:
            return nil
        }
    }

    static func convertedVisionWeight(
        _ value: MLXArray,
        sourceName: String,
        sourceShape: [Int]
    ) throws -> MLXArray {
        guard let permutation = try visionWeightPermutation(
            sourceName: sourceName, sourceShape: sourceShape)
        else { return value }
        return contiguous(value.transposed(axes: permutation))
    }

    static func validatedPaths(
        source: URL,
        output: URL
    ) throws -> (source: URL, output: URL) {
        do {
            let paths = try CheckpointConversionPathSafety.validate(
                source: source, output: output)
            return (paths.source, paths.output)
        } catch CheckpointConversionPathSafety.PathError.nonLocal {
            throw ConversionError.invalidSource(
                "GLM-5.3 conversion requires local filesystem paths; remote download is not supported.")
        } catch {
            throw ConversionError.unsafeOutput(
                "Conversion output cannot be a filesystem or volume root, and source/output must be separate directories with neither containing the other, including through symlinks.")
        }
    }

    private static func loadSource(
        _ source: URL,
        explicitRevision: String?,
        fingerprintContents: Bool
    ) throws -> SourceCheckpoint {
        let fm = FileManager.default
        let root = source.standardizedFileURL
        var isDirectory: ObjCBool = false
        guard root.isFileURL,
              fm.fileExists(atPath: root.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            throw ConversionError.invalidSource(
                "GLM-5.3 conversion requires an existing local source directory.")
        }
        let configURL = root.appendingPathComponent("config.json")
        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
        guard fm.fileExists(atPath: configURL.path), fm.fileExists(atPath: indexURL.path) else {
            throw ConversionError.invalidSource(
                "Source must contain config.json and model.safetensors.index.json.")
        }
        let configData = try Data(contentsOf: configURL)
        let indexData = try Data(contentsOf: indexURL)
        let config = try parseConfiguration(configData)
        let revision = try resolveRevision(root: root, explicit: explicitRevision)
        var assetSHA256 = [String: String]()
        for name in requiredSupportFiles {
            let url = root.appendingPathComponent(name)
            guard fm.fileExists(atPath: url.path) else {
                throw ConversionError.invalidSource(
                    "Required multimodal/tokenizer asset \(name) is missing.")
            }
            assetSHA256[name] = try sha256File(url)
        }

        guard let index = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
              let rawWeightMap = index["weight_map"] as? [String: Any]
        else {
            throw ConversionError.invalidSource("Invalid SafeTensor index.")
        }
        var weightMap = [String: String]()
        for (name, value) in rawWeightMap {
            guard let shard = value as? String,
                  URL(fileURLWithPath: shard).lastPathComponent == shard,
                  shard.hasSuffix(".safetensors")
            else {
                throw ConversionError.invalidSource(
                    "Invalid shard mapping for tensor \(name).")
            }
            weightMap[name] = shard
        }
        guard !weightMap.isEmpty else {
            throw ConversionError.invalidSource("SafeTensor index has no tensors.")
        }

        let shardNames = Set(weightMap.values).sorted()
        var shardURLs = [String: URL]()
        var fingerprints = [String: SourceShardFingerprint]()
        var sourceBytes: Int64 = 0
        var tensors = [String: TensorReference]()
        for shard in shardNames {
            let url = root.appendingPathComponent(shard)
            guard fm.fileExists(atPath: url.path) else {
                throw ConversionError.invalidSource("Missing source shard \(shard).")
            }
            let values = try url.resourceValues(
                forKeys: [.fileSizeKey, .contentModificationDateKey])
            let size = Int64(values.fileSize ?? 0)
            fingerprints[shard] = SourceShardFingerprint(
                size: size,
                modificationTime: values.contentModificationDate?.timeIntervalSince1970 ?? 0,
                contentSHA256: fingerprintContents ? try sha256File(url) : nil)
            sourceBytes += size
            shardURLs[shard] = url
            let header = try AFMSafetensorHeader(url: url)
            for tensor in header.tensors {
                guard weightMap[tensor.name] == shard else {
                    throw ConversionError.invalidSource(
                        "Tensor \(tensor.name) header/index shard mismatch.")
                }
                tensors[tensor.name] = TensorReference(
                    name: tensor.name,
                    shard: shard,
                    dtype: tensor.dtype,
                    shape: tensor.shape,
                    byteCount: Int64(tensor.byteCount))
            }
        }
        let missing = Set(weightMap.keys).subtracting(tensors.keys)
        guard missing.isEmpty else {
            throw ConversionError.invalidSource(
                "SafeTensor headers are missing indexed tensors: \(missing.sorted().prefix(5).joined(separator: ", ")).")
        }
        try validateTensorFormats(tensors, config: config)
        return SourceCheckpoint(
            root: root,
            configURL: configURL,
            indexURL: indexURL,
            configData: configData,
            indexData: indexData,
            config: config,
            revision: revision,
            tensors: tensors,
            shardURLs: shardURLs,
            shardFingerprints: fingerprints,
            assetSHA256: assetSHA256,
            sourceBytes: sourceBytes)
    }

    private static func parseConfiguration(_ data: Data) throws -> Configuration {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              object["model_type"] as? String == "glm5_next",
              let text = object["text_config"] as? [String: Any],
              text["model_type"] as? String == "glm5_next_text",
              let hiddenLayers = text["num_hidden_layers"] as? Int,
              let routedExperts = text["n_routed_experts"] as? Int,
              let kvLoraRank = text["kv_lora_rank"] as? Int,
              let attentionHeads = text["num_attention_heads"] as? Int,
              let qkNopeHeadDim = text["qk_nope_head_dim"] as? Int,
              let vHeadDim = text["v_head_dim"] as? Int,
              let vision = object["vision_config"] as? [String: Any],
              vision["model_type"] as? String == "glm5_next_vision",
              let quantization = object["quantization_config"] as? [String: Any],
              (quantization["fmt"] as? String)?.lowercased() == "e4m3",
              let blockSize = quantization["weight_block_size"] as? [Int],
              blockSize == [fp8BlockRows, fp8BlockColumns]
        else {
            throw ConversionError.invalidSource(
                "Source must be the official multimodal glm5_next raw E4M3 checkpoint with 128x128 block scales.")
        }
        return Configuration(
            object: object,
            hiddenLayers: hiddenLayers,
            routedExperts: routedExperts,
            kvLoraRank: kvLoraRank,
            attentionHeads: attentionHeads,
            qkNopeHeadDim: qkNopeHeadDim,
            vHeadDim: vHeadDim,
            sourceMTPCount: text["num_nextn_predict_layers"] as? Int ?? 0)
    }

    private static func validateTensorFormats(
        _ tensors: [String: TensorReference],
        config: Configuration
    ) throws {
        var consumedScales = Set<String>()
        for reference in tensors.values {
            if reference.dtype == .uint8, reference.name.hasSuffix(".weight") {
                throw ConversionError.unsupportedTensor(
                    "Tensor \(reference.name) is U8, not header-declared F8_E4M3; AFM never infers FP8 from UInt8.")
            }
            guard reference.dtype == .float8E4M3 else { continue }
            guard reference.name.hasSuffix(".weight") else {
                throw ConversionError.unsupportedTensor(
                    "Only weight tensors may use raw F8_E4M3: \(reference.name).")
            }
            let scaleName = scaleName(for: reference.name)
            guard let scale = tensors[scaleName], scale.dtype == .float32,
                  reference.shape.count == 2, scale.shape.count == 2
            else {
                throw ConversionError.unsupportedTensor(
                    "Raw FP8 tensor \(reference.name) requires a paired F32 \(scaleName).")
            }
            let expected = [
                (reference.shape[0] + fp8BlockRows - 1) / fp8BlockRows,
                (reference.shape[1] + fp8BlockColumns - 1) / fp8BlockColumns,
            ]
            guard scale.shape == expected else {
                throw ConversionError.unsupportedTensor(
                    "Inverse-scale shape \(scale.shape) does not match \(reference.name) \(reference.shape).")
            }
            consumedScales.insert(scaleName)
        }
        let orphanScales = tensors.keys.filter {
            $0.hasSuffix(".weight_scale_inv") && !consumedScales.contains($0)
        }
        guard orphanScales.isEmpty else {
            throw ConversionError.unsupportedTensor(
                "Orphan FP8 inverse scales: \(orphanScales.sorted().prefix(5).joined(separator: ", ")).")
        }
    }

    private static func inspection(for checkpoint: SourceCheckpoint) -> Inspection {
        var estimated: Int64 = 0
        var vision = 0
        var fp8 = 0
        var omittedMTP = 0
        for reference in checkpoint.tensors.values {
            if let layer = layerIndex(in: reference.name),
               layer >= checkpoint.config.hiddenLayers
            {
                omittedMTP += 1
                continue
            }
            if reference.name.hasSuffix(".weight_scale_inv") { continue }
            if reference.name.hasPrefix("model.visual.") { vision += 1 }
            if reference.dtype == .float8E4M3 {
                fp8 += 1
                estimated += affineOutputBytes(shape: reference.shape)
            } else if shouldQuantize(reference) {
                estimated += affineOutputBytes(shape: reference.shape)
            } else {
                estimated += reference.byteCount
            }
        }
        let margin = max(Int64(16_000_000_000), estimated / 10)
        return Inspection(
            modelType: "glm5_next",
            sourceRevision: checkpoint.revision,
            sourceBytes: checkpoint.sourceBytes,
            estimatedOutputBytes: estimated,
            requiredDestinationFreeBytes: max(minimumDestinationFreeBytes, estimated + margin),
            shardCount: checkpoint.shardURLs.count,
            tensorCount: checkpoint.tensors.count,
            visionTensorCount: vision,
            fp8TensorCount: fp8,
            omittedMTPTensorCount: omittedMTP)
    }

    private static func makePlan(checkpoint: SourceCheckpoint) throws -> [ConversionUnit] {
        var ordinary = [String: [TensorReference]]()
        var experts = [String: [TensorReference]]()
        for reference in checkpoint.tensors.values.sorted(by: { $0.name < $1.name }) {
            if let layer = layerIndex(in: reference.name),
               layer >= checkpoint.config.hiddenLayers
            {
                continue
            }
            if reference.name.hasSuffix(".weight_scale_inv") { continue }
            if let expert = expertIdentity(reference.name) {
                let key = "\(expert.layer):\(expert.projection)"
                experts[key, default: []].append(reference)
                continue
            }
            let unit: String
            if let layer = layerIndex(in: reference.name) {
                unit = String(format: "language-layer-%03d", layer)
            } else if reference.name.hasPrefix("model.visual.blocks."),
                      let block = visionBlockIndex(in: reference.name)
            {
                unit = String(format: "vision-block-%03d", block)
            } else if reference.name.hasPrefix("model.visual.") {
                unit = "vision-globals"
            } else {
                unit = "language-globals"
            }
            ordinary[unit, default: []].append(reference)
        }

        var plan = ordinary.keys.sorted().map { id in
            ConversionUnit.ordinary(
                id: id,
                outputFile: "model-\(id).safetensors",
                tensors: ordinary[id]!.sorted { $0.name < $1.name })
        }
        for key in experts.keys.sorted() {
            let parts = key.split(separator: ":")
            guard parts.count == 2, let layer = Int(parts[0]) else {
                throw ConversionError.invalidSource("Invalid expert conversion key \(key).")
            }
            let projection = String(parts[1])
            let references = experts[key]!.sorted {
                (expertIdentity($0.name)?.expert ?? -1) < (expertIdentity($1.name)?.expert ?? -1)
            }
            let identities = references.compactMap { expertIdentity($0.name)?.expert }
            guard identities == Array(0..<checkpoint.config.routedExperts) else {
                throw ConversionError.invalidSource(
                    "Layer \(layer) \(projection) has \(identities.count) experts; expected \(checkpoint.config.routedExperts).")
            }
            let id = String(format: "language-layer-%03d-experts-%@", layer, projection)
            plan.append(.experts(
                id: id,
                outputFile: "model-\(id).safetensors",
                layer: layer,
                projection: projection,
                tensors: references))
        }
        return plan.sorted { $0.id < $1.id }
    }

    private func convertOrdinary(
        _ references: [TensorReference],
        checkpoint: SourceCheckpoint,
        loader: TensorLoader
    ) throws -> ConvertedUnit {
        let byName = Dictionary(uniqueKeysWithValues: references.map { ($0.name, $0) })
        var handled = Set<String>()
        var converted = ConvertedUnit(
            arrays: [:], quantization: [:], skippedQuantization: [])

        for reference in references where reference.name.hasSuffix(".q_conv1d.weight") {
            let prefix = String(reference.name.dropLast("q_conv1d.weight".count))
            let names = ["q", "k", "v"].map { "\(prefix)\($0)_conv1d.weight" }
            guard names.allSatisfy({ byName[$0] != nil }) else {
                throw ConversionError.invalidSource(
                    "Incomplete q/k/v convolution group at \(prefix).")
            }
            let values = try names.map { name -> MLXArray in
                let item = byName[name]!
                handled.insert(name)
                return try decodedWeight(item, checkpoint: checkpoint, loader: loader)
            }
            var convolution = concatenated(values, axis: 0)
            if convolution.ndim == 3, convolution.dim(-1) != 1 {
                convolution = convolution.movedAxis(source: 2, destination: 1)
            }
            converted.arrays[Self.mappedName(prefix + "conv1d.weight")] = convolution
        }

        for reference in references where reference.name.hasSuffix(".kv_b_proj.weight") {
            handled.insert(reference.name)
            let value = try decodedWeight(
                reference, checkpoint: checkpoint, loader: loader)
            let c = checkpoint.config
            let expectedRows = c.attentionHeads * (c.qkNopeHeadDim + c.vHeadDim)
            guard reference.shape == [expectedRows, c.kvLoraRank] else {
                throw ConversionError.unsupportedTensor(
                    "Unexpected kv_b_proj shape \(reference.shape) for \(reference.name).")
            }
            let shaped = value.reshaped(
                c.attentionHeads, c.qkNopeHeadDim + c.vHeadDim, c.kvLoraRank)
            let query = contiguous(
                shaped[0..., 0..<c.qkNopeHeadDim, 0...].swappedAxes(-1, -2))
            let output = contiguous(shaped[0..., c.qkNopeHeadDim..., 0...])
            let prefix = String(Self.mappedName(reference.name).dropLast("kv_b_proj.weight".count))
            try addQuantized(query, base: prefix + "embed_q", to: &converted)
            try addQuantized(output, base: prefix + "unembed_out", to: &converted)
        }

        for reference in references where !handled.contains(reference.name) {
            let mapped = Self.mappedName(reference.name)
            if reference.dtype == .float8E4M3 || Self.shouldQuantize(reference) {
                let value = try decodedWeight(
                    reference, checkpoint: checkpoint, loader: loader)
                let base = String(mapped.dropLast(".weight".count))
                try addQuantized(value, base: base, to: &converted)
            } else {
                let loaded = try loader.load(reference)
                let value = try Self.convertedVisionWeight(
                    loaded,
                    sourceName: reference.name,
                    sourceShape: reference.shape)
                converted.arrays[mapped] = value
                if mapped.hasSuffix(".weight") {
                    converted.skippedQuantization.insert(
                        String(mapped.dropLast(".weight".count)))
                }
            }
        }
        return converted
    }

    private func convertExperts(
        layer: Int,
        projection: String,
        references: [TensorReference],
        checkpoint: SourceCheckpoint,
        loader: TensorLoader
    ) throws -> ConvertedUnit {
        var weights = [MLXArray?](repeating: nil, count: checkpoint.config.routedExperts)
        var scales = [MLXArray?](repeating: nil, count: checkpoint.config.routedExperts)
        var biases = [MLXArray?](repeating: nil, count: checkpoint.config.routedExperts)
        let grouped = Dictionary(grouping: references, by: \.shard)
        for shard in grouped.keys.sorted() {
            for reference in grouped[shard]!.sorted(by: {
                (Self.expertIdentity($0.name)?.expert ?? -1)
                    < (Self.expertIdentity($1.name)?.expert ?? -1)
            }) {
                guard let identity = Self.expertIdentity(reference.name) else { continue }
                let decoded = try decodedWeight(
                    reference, checkpoint: checkpoint, loader: loader)
                let quantized = try Self.affineQuantize(decoded)
                weights[identity.expert] = quantized.weight
                scales[identity.expert] = quantized.scales
                biases[identity.expert] = quantized.biases
            }
            loader.release()
        }
        guard weights.allSatisfy({ $0 != nil }), scales.allSatisfy({ $0 != nil }),
              biases.allSatisfy({ $0 != nil })
        else {
            throw ConversionError.invalidSource(
                "Cross-shard expert reconstruction is incomplete for layer \(layer) \(projection).")
        }
        let base = "language_model.model.layers.\(layer).mlp.switch_mlp.\(projection)"
        let stackedWeights = stacked(weights.map { $0! })
        let stackedScales = stacked(scales.map { $0! })
        let stackedBiases = stacked(biases.map { $0! })
        MLX.eval(stackedWeights, stackedScales, stackedBiases)
        return ConvertedUnit(
            arrays: [
                "\(base).weight": stackedWeights,
                "\(base).scales": stackedScales,
                "\(base).biases": stackedBiases,
            ],
            quantization: [base: Self.defaultQuantization],
            skippedQuantization: [])
    }

    private func decodedWeight(
        _ reference: TensorReference,
        checkpoint: SourceCheckpoint,
        loader: TensorLoader
    ) throws -> MLXArray {
        let raw = try loader.load(reference)
        guard reference.dtype == .float8E4M3 else { return raw }
        let scaleReference = checkpoint.tensors[Self.scaleName(for: reference.name)]!
        let scale = try loader.load(scaleReference)
        return try Self.dequantizeFP8(
            raw: raw,
            scaleInverse: scale,
            weightShape: reference.shape,
            scaleShape: scaleReference.shape)
    }

    private func addQuantized(
        _ value: MLXArray,
        base: String,
        to converted: inout ConvertedUnit
    ) throws {
        let quantized = try Self.affineQuantize(value)
        converted.arrays["\(base).weight"] = quantized.weight
        converted.arrays["\(base).scales"] = quantized.scales
        converted.arrays["\(base).biases"] = quantized.biases
        converted.quantization[base] = Self.defaultQuantization
    }

    private static var defaultQuantization: Quantization {
        Quantization(groupSize: affineGroupSize, bits: affineBits, mode: "affine")
    }

    private static func shouldQuantize(_ reference: TensorReference) -> Bool {
        guard reference.name.hasSuffix(".weight"),
              !reference.name.hasPrefix("model.visual."),
              reference.dtype != .float8E4M3,
              reference.dtype.isFloatingPoint,
              reference.shape.count >= 2,
              let columns = reference.shape.last,
              columns.isMultiple(of: affineGroupSize)
        else { return false }
        return true
    }

    private static func affineOutputBytes(shape: [Int]) -> Int64 {
        guard shape.count >= 2, let columns = shape.last else { return 0 }
        let rows = shape.dropLast().reduce(1, *)
        let packedColumns = (columns * affineBits + 31) / 32
        let groups = (columns + affineGroupSize - 1) / affineGroupSize
        return Int64(rows * packedColumns * 4 + rows * groups * 2 * 2)
    }

    private static func scaleName(for weightName: String) -> String {
        String(weightName.dropLast(".weight".count)) + ".weight_scale_inv"
    }

    private static func expertIdentity(
        _ name: String
    ) -> (layer: Int, expert: Int, projection: String)? {
        let parts = name.split(separator: ".")
        guard let layers = parts.firstIndex(of: "layers"), layers + 6 < parts.count,
              let layer = Int(parts[layers + 1]),
              parts[layers + 2] == "mlp", parts[layers + 3] == "experts",
              let expert = Int(parts[layers + 4]),
              ["gate_proj", "up_proj", "down_proj"].contains(String(parts[layers + 5])),
              parts[layers + 6] == "weight"
        else { return nil }
        return (layer, expert, String(parts[layers + 5]))
    }

    private static func layerIndex(in name: String) -> Int? {
        let parts = name.split(separator: ".")
        guard let index = parts.firstIndex(of: "layers"), index + 1 < parts.count else {
            return nil
        }
        return Int(parts[index + 1])
    }

    private static func visionBlockIndex(in name: String) -> Int? {
        let parts = name.split(separator: ".")
        guard let index = parts.firstIndex(of: "blocks"), index + 1 < parts.count else {
            return nil
        }
        return Int(parts[index + 1])
    }

    private static func validate(
        state: State,
        checkpoint: SourceCheckpoint,
        configHash: String,
        indexHash: String,
        profile: Profile,
        plan: [ConversionUnit]
    ) throws {
        guard state.formatVersion == currentFormatVersion else {
            throw ConversionError.sourceMismatch(
                "Conversion format changed; use --overwrite to rebuild the output.")
        }
        guard state.profile == profile.rawValue else {
            throw ConversionError.sourceMismatch(
                "Conversion profile differs from the resumable output.")
        }
        guard state.sourceRevision == checkpoint.revision else {
            throw ConversionError.sourceMismatch(
                "Source revision changed from \(state.sourceRevision) to \(checkpoint.revision).")
        }
        guard state.configSHA256 == configHash, state.indexSHA256 == indexHash,
              state.sourceShards == checkpoint.shardFingerprints,
              state.sourceAssets == checkpoint.assetSHA256
        else {
            throw ConversionError.sourceMismatch(
                "Source config, index, shard contents, or support assets changed; use --overwrite after verifying the checkpoint.")
        }
        let planned = Dictionary(uniqueKeysWithValues: plan.map { ($0.id, $0) })
        guard state.completed.keys.allSatisfy({ planned[$0] != nil }) else {
            throw ConversionError.sourceMismatch(
                "The conversion resume manifest contains units outside the current conversion plan.")
        }
        for (id, completed) in state.completed {
            guard let unit = planned[id], completed.outputFile == unit.outputFile,
                  URL(fileURLWithPath: completed.outputFile).lastPathComponent
                    == completed.outputFile,
                  completed.outputSize >= 0,
                  isSHA256(completed.outputSHA256),
                  !completed.outputKeys.isEmpty,
                  Set(completed.outputKeys).count == completed.outputKeys.count,
                  Set(completed.outputKeys) == expectedOutputKeys(
                    for: unit, checkpoint: checkpoint),
                  completed.outputSize <= maximumOutputBytes(
                    for: unit, checkpoint: checkpoint),
                  completed.outputKeys.allSatisfy({ key in
                      state.weightMap[key] == completed.outputFile
                  })
            else {
                throw ConversionError.sourceMismatch(
                    "The conversion resume manifest does not match planned unit \(id).")
            }
        }
        let completedKeys = Set(state.completed.values.flatMap(\.outputKeys))
        guard Set(state.weightMap.keys) == completedKeys,
              state.weightMap.values.allSatisfy({ outputFile in
                  state.completed.values.contains(where: {
                      $0.outputFile == outputFile
                  })
              })
        else {
            throw ConversionError.sourceMismatch(
                "The conversion resume manifest weight map is inconsistent with completed units.")
        }
    }

    private static func verifiedCompletedBytes(
        state: State,
        plan: [ConversionUnit],
        checkpoint: SourceCheckpoint,
        output: URL
    ) throws -> Int64 {
        let planned = Dictionary(uniqueKeysWithValues: plan.map { ($0.id, $0) })
        var total: Int64 = 0
        for (id, completed) in state.completed {
            guard let unit = planned[id] else { continue }
            let url = output.appendingPathComponent(unit.outputFile)
            guard try isCompletedOutputValid(completed, at: url)
            else { continue }
            let sum = total.addingReportingOverflow(completed.outputSize)
            guard !sum.overflow else {
                throw ConversionError.sourceMismatch(
                    "Completed conversion output size overflowed Int64.")
            }
            total = sum.partialValue
        }
        return total
    }

    private static func isCompletedOutputValid(
        _ completed: CompletedUnit,
        at url: URL
    ) throws -> Bool {
        guard FileManager.default.fileExists(atPath: url.path),
              Int64(try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? -1)
                == completed.outputSize,
              try sha256File(url) == completed.outputSHA256,
              let header = try? AFMSafetensorHeader(url: url),
              Set(header.tensors.map(\.name)) == Set(completed.outputKeys)
        else { return false }
        return true
    }

    private static func expectedOutputKeys(
        for unit: ConversionUnit,
        checkpoint: SourceCheckpoint
    ) -> Set<String> {
        switch unit {
        case .experts(_, _, let layer, let projection, _):
            let base = "language_model.model.layers.\(layer).mlp.switch_mlp.\(projection)"
            return ["\(base).weight", "\(base).scales", "\(base).biases"]
        case .ordinary(_, _, let references):
            let byName = Dictionary(uniqueKeysWithValues: references.map { ($0.name, $0) })
            var handled = Set<String>()
            var result = Set<String>()
            for reference in references where reference.name.hasSuffix(".q_conv1d.weight") {
                let prefix = String(reference.name.dropLast("q_conv1d.weight".count))
                let names = ["q", "k", "v"].map { "\(prefix)\($0)_conv1d.weight" }
                guard names.allSatisfy({ byName[$0] != nil }) else { continue }
                handled.formUnion(names)
                result.insert(mappedName(prefix + "conv1d.weight"))
            }
            for reference in references where reference.name.hasSuffix(".kv_b_proj.weight") {
                handled.insert(reference.name)
                let prefix = String(
                    mappedName(reference.name).dropLast("kv_b_proj.weight".count))
                for base in [prefix + "embed_q", prefix + "unembed_out"] {
                    result.formUnion([
                        "\(base).weight", "\(base).scales", "\(base).biases",
                    ])
                }
            }
            for reference in references where !handled.contains(reference.name) {
                let mapped = mappedName(reference.name)
                if reference.dtype == .float8E4M3 || shouldQuantize(reference) {
                    let base = String(mapped.dropLast(".weight".count))
                    result.formUnion([
                        "\(base).weight", "\(base).scales", "\(base).biases",
                    ])
                } else {
                    result.insert(mapped)
                }
            }
            return result
        }
    }

    private static func maximumOutputBytes(
        for unit: ConversionUnit,
        checkpoint: SourceCheckpoint
    ) -> Int64 {
        let payload: Int64
        switch unit {
        case .experts(_, _, _, _, let references):
            payload = references.reduce(0) { $0 + affineOutputBytes(shape: $1.shape) }
        case .ordinary(_, _, let references):
            var handled = Set<String>()
            var bytes: Int64 = 0
            let byName = Dictionary(uniqueKeysWithValues: references.map { ($0.name, $0) })
            for reference in references where reference.name.hasSuffix(".q_conv1d.weight") {
                let prefix = String(reference.name.dropLast("q_conv1d.weight".count))
                let names = ["q", "k", "v"].map { "\(prefix)\($0)_conv1d.weight" }
                guard names.allSatisfy({ byName[$0] != nil }) else { continue }
                handled.formUnion(names)
                for name in names {
                    let item = byName[name]!
                    bytes += item.dtype == .float8E4M3
                        ? Int64(item.shape.reduce(1, *)) * 4 : item.byteCount
                }
            }
            for reference in references where reference.name.hasSuffix(".kv_b_proj.weight") {
                handled.insert(reference.name)
                bytes += affineOutputBytes(shape: [
                    checkpoint.config.attentionHeads,
                    checkpoint.config.kvLoraRank,
                    checkpoint.config.qkNopeHeadDim,
                ])
                bytes += affineOutputBytes(shape: [
                    checkpoint.config.attentionHeads,
                    checkpoint.config.vHeadDim,
                    checkpoint.config.kvLoraRank,
                ])
            }
            for reference in references where !handled.contains(reference.name) {
                bytes += reference.dtype == .float8E4M3 || shouldQuantize(reference)
                    ? affineOutputBytes(shape: reference.shape) : reference.byteCount
            }
            payload = bytes
        }
        let bounded = payload.addingReportingOverflow(4_000_000)
        return bounded.overflow ? Int64.max : bounded.partialValue
    }

    private static func isSHA256(_ value: String) -> Bool {
        value.count == 64 && value.allSatisfy(\.isHexDigit)
    }

    private static func resolveRevision(root: URL, explicit: String?) throws -> String {
        let inferred = try inferredRevision(root: root)
        if let explicit {
            guard isCommitRevision(explicit) else {
                throw ConversionError.invalidSource(
                    "--source-revision must be a full 40-character hexadecimal commit revision.")
            }
            let normalized = explicit.lowercased()
            if let inferred, inferred != normalized {
                throw ConversionError.invalidSource(
                    "--source-revision \(normalized) conflicts with locally provable checkpoint revision \(inferred).")
            }
            return normalized
        }
        if let inferred { return inferred }
        throw ConversionError.invalidSource(
            "Cannot prove the local checkpoint revision. Pass --source-revision with the official 40-character Hugging Face commit.")
    }

    private static func inferredRevision(root: URL) throws -> String? {
        var revisions = Set<String>()
        let components = root.pathComponents
        if let snapshots = components.lastIndex(of: "snapshots"), snapshots + 1 < components.count,
           isCommitRevision(components[snapshots + 1])
        {
            revisions.insert(components[snapshots + 1].lowercased())
        }
        let metadataCandidates = [
            ".cache/huggingface/download/config.json.metadata",
            ".cache/huggingface/download/model.safetensors.index.json.metadata",
        ]
        for relative in metadataCandidates {
            let url = root.appendingPathComponent(relative)
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
            if let revision = text.split(whereSeparator: \.isWhitespace)
                .map(String.init).first(where: isCommitRevision)
            {
                revisions.insert(revision.lowercased())
            }
        }
        guard revisions.count <= 1 else {
            throw ConversionError.invalidSource(
                "Local Hugging Face metadata contains conflicting checkpoint revisions.")
        }
        return revisions.first
    }

    private static func isCommitRevision(_ value: String) -> Bool {
        value.count == 40 && value.allSatisfy { $0.isHexDigit }
    }

    private func writeOutputConfiguration(
        checkpoint: SourceCheckpoint,
        state: State
    ) throws {
        var object = checkpoint.config.object
        object.removeValue(forKey: "quantization_config")
        if var text = object["text_config"] as? [String: Any] {
            text["num_nextn_predict_layers"] = 0
            object["text_config"] = text
        }
        var quantization: [String: Any] = [
            "group_size": Self.affineGroupSize,
            "bits": Self.affineBits,
            "mode": "affine",
        ]
        for base in state.skippedQuantization { quantization[base] = false }
        for (base, value) in state.quantization where value != Self.defaultQuantization {
            quantization[base] = [
                "group_size": value.groupSize,
                "bits": value.bits,
                "mode": value.mode,
            ]
        }
        object["quantization"] = quantization
        object["afm_conversion"] = [
            "format_version": Self.currentFormatVersion,
            "profile": profile.rawValue,
            "source_model": Self.officialModelID,
            "source_revision": checkpoint.revision,
            "source_format": "safetensors-f8-e4m3-block",
            "source_weight_block_size": [Self.fp8BlockRows, Self.fp8BlockColumns],
            "inverse_scale_compute_dtype": "float32",
            "output_group_size": Self.affineGroupSize,
            "output_bits": Self.affineBits,
            "vision_preserved": true,
            "processor_assets_copied": true,
            "source_num_nextn_predict_layers": checkpoint.config.sourceMTPCount,
            "mtp_omitted": true,
            "omitted_mtp_tensor_count": state.omittedMTP.count,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try writeAtomically(data, to: output.standardizedFileURL.appendingPathComponent("config.json"))
    }

    private func writeOutputIndex(state: State, output: URL) throws {
        let files = Set(state.weightMap.values)
        let totalSize = try files.reduce(Int64(0)) { partial, name in
            partial + Int64(
                try output.appendingPathComponent(name)
                    .resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
        }
        let object: [String: Any] = [
            "metadata": [
                "total_size": totalSize,
                "afm_source_revision": state.sourceRevision,
            ],
            "weight_map": state.weightMap,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try writeAtomically(
            data, to: output.appendingPathComponent("model.safetensors.index.json"))
    }

    private func copySupportFiles(from source: URL, to output: URL) throws {
        let fm = FileManager.default
        let names = [
            "LICENSE", "README.md", "chat_template.jinja", "generation_config.json",
            "processor_config.json", "tokenizer.json", "tokenizer_config.json",
        ]
        for name in names {
            let from = source.appendingPathComponent(name)
            let to = output.appendingPathComponent(name)
            guard fm.fileExists(atPath: from.path) else {
                if Self.requiredSupportFiles.contains(name) {
                    throw ConversionError.invalidSource(
                        "Required multimodal/tokenizer asset \(name) is missing.")
                }
                continue
            }
            try writeAtomically(try Data(contentsOf: from), to: to)
        }
    }

    private func report(_ message: String) {
        progress?(message)
    }

    private func saveState(_ state: State) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try writeAtomically(try encoder.encode(state), to: stateURL)
    }

    private func writeAtomically(_ data: Data, to url: URL) throws {
        let partial = url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.lastPathComponent).partial")
        try data.write(to: partial, options: .atomic)
        try replaceOrMove(partial, to: url)
    }

    private func replaceOrMove(_ source: URL, to destination: URL) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) {
            _ = try fm.replaceItemAt(destination, withItemAt: source)
        } else {
            try fm.moveItem(at: source, to: destination)
        }
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func sha256File(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var digest = SHA256()
        while let chunk = try handle.read(upToCount: 4 * 1024 * 1024), !chunk.isEmpty {
            digest.update(data: chunk)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// Minimal semantic SafeTensor header reader used before MLX loads tensor
/// payloads. MLX represents F8 payloads as UInt8 arrays; this header is the
/// authority that distinguishes genuine E4M3 data from an ordinary U8 tensor.
struct AFMSafetensorHeader {
    enum DType: String, Equatable {
        case bool = "BOOL"
        case uint8 = "U8"
        case int8 = "I8"
        case uint16 = "U16"
        case int16 = "I16"
        case uint32 = "U32"
        case int32 = "I32"
        case uint64 = "U64"
        case int64 = "I64"
        case float16 = "F16"
        case bfloat16 = "BF16"
        case float32 = "F32"
        case float64 = "F64"
        case float8E4M3 = "F8_E4M3"

        var byteWidth: Int {
            switch self {
            case .bool, .uint8, .int8, .float8E4M3: 1
            case .uint16, .int16, .float16, .bfloat16: 2
            case .uint32, .int32, .float32: 4
            case .uint64, .int64, .float64: 8
            }
        }

        var isFloatingPoint: Bool {
            switch self {
            case .float16, .bfloat16, .float32, .float64, .float8E4M3: true
            default: false
            }
        }
    }

    struct Tensor: Equatable {
        let name: String
        let dtype: DType
        let shape: [Int]
        let dataOffsets: [Int]
        var byteCount: Int { dataOffsets[1] - dataOffsets[0] }
    }

    let tensors: [Tensor]

    init(url: URL) throws {
        let fileSize = Int64(
            try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
            throw GLM5NextCheckpointConverter.ConversionError.invalidSource(
                "Truncated SafeTensor header in \(url.lastPathComponent).")
        }
        let headerSize = prefix.enumerated().reduce(UInt64(0)) { result, item in
            result | (UInt64(item.element) << UInt64(item.offset * 8))
        }
        guard fileSize >= 8,
              headerSize <= UInt64(fileSize - 8),
              headerSize <= 512 * 1024 * 1024,
              headerSize <= UInt64(Int.max),
              let data = try handle.read(upToCount: Int(headerSize)),
              data.count == Int(headerSize),
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw GLM5NextCheckpointConverter.ConversionError.invalidSource(
                "Invalid SafeTensor header in \(url.lastPathComponent).")
        }
        let payloadBytes = fileSize - 8 - Int64(headerSize)
        var parsed = [Tensor]()
        for (name, raw) in object where name != "__metadata__" {
            guard let entry = raw as? [String: Any],
                  let rawDType = entry["dtype"] as? String,
                  let dtype = DType(rawValue: rawDType),
                  let shape = entry["shape"] as? [Int],
                  let offsets = entry["data_offsets"] as? [Int], offsets.count == 2,
                  offsets[0] >= 0, offsets[1] >= offsets[0],
                  Int64(offsets[1]) <= payloadBytes,
                  let elementCount = Self.elementCount(shape),
                  let expectedBytes = Self.byteCount(
                    elements: elementCount, width: dtype.byteWidth),
                  expectedBytes == offsets[1] - offsets[0]
            else {
                throw GLM5NextCheckpointConverter.ConversionError.invalidSource(
                    "Invalid or unsupported SafeTensor entry \(name) in \(url.lastPathComponent).")
            }
            parsed.append(Tensor(
                name: name, dtype: dtype, shape: shape, dataOffsets: offsets))
        }
        tensors = parsed.sorted { $0.dataOffsets[0] < $1.dataOffsets[0] }
    }

    private static func elementCount(_ shape: [Int]) -> Int? {
        var count = 1
        for dimension in shape {
            guard dimension >= 0 else { return nil }
            let product = count.multipliedReportingOverflow(by: dimension)
            guard !product.overflow else { return nil }
            count = product.partialValue
        }
        return count
    }

    private static func byteCount(elements: Int, width: Int) -> Int? {
        let product = elements.multipliedReportingOverflow(by: width)
        return product.overflow ? nil : product.partialValue
    }
}
