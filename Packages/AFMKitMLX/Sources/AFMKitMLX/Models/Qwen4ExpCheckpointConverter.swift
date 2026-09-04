import CryptoKit
import Foundation
import MLX

/// Converts the official Qwen3.8 Flash Next BF16 checkpoint into AFMKit's
/// affine-quantized runtime layout with a host-mapped n-gram table.
///
/// The mapped-sidecar layout is based on the MIT-licensed reference converter
/// in ddalcu/mlx-serve at commit
/// 805807669565d359188b329c659f9f45d6358cd7. This is a native Swift
/// implementation using AFMKit and MLX primitives; it does not invoke Python
/// or depend on the reference project at runtime.
public struct Qwen4ExpCheckpointConverter {
    public typealias ProgressHandler = (String) -> Void

    public static let officialModelID = "Qwen/Qwen3.8-Flash-Next"
    private static let currentFormatVersion = 3
    private static let groupSize = 64
    private static let bits = 4
    private static let lmHeadBits = 8
    private static let ngramGroupSize = 32
    private static let ngramBits = 4
    private static let ngramDimension = 160
    private static let ngramSidecarFilename = "ngram_table.ngram"
    private static let minimumDestinationFreeBytes: Int64 = 140_000_000_000
    private static let ngramMarker = ".ple.ple_embedding.ngram_embedding.shard_"
    private static let requiredSupportFiles = [
        "chat_template.jinja", "preprocessor_config.json", "tokenizer.json",
        "tokenizer_config.json",
    ]
    private static let copiedSupportFiles = [
        "LICENSE", "README.md", "chat_template.jinja", "generation_config.json",
        "merges.txt", "preprocessor_config.json", "processor_config.json",
        "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json",
        "vocab.json",
    ]
    private static let foldedNormSuffixes = [
        "hc_norm.weight", "q_norm.weight", "k_norm.weight",
        "q_layernorm.weight", "k_layernorm.weight",
        "ple.norm_key.weight", "ple.norm_query.weight", "ple.norm_conv.weight",
        "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
    ]

    public enum Profile: String, Codable, CaseIterable, Sendable {
        case afmMapped4 = "afm-mapped-4"
    }

    public struct Inspection: Sendable, Equatable {
        public let sourceRevision: String
        public let sourceBytes: Int64
        public let estimatedOutputBytes: Int64
        public let requiredDestinationFreeBytes: Int64
        public let shardCount: Int
        public let tensorCount: Int
        public let ngramShardCount: Int
    }

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

    private struct TensorReference: Equatable {
        let name: String
        let shard: String
        let dtype: AFMSafetensorHeader.DType
        let shape: [Int]
        let byteCount: Int64
    }

    private struct SourceShardFingerprint: Codable, Equatable {
        let size: Int64
        let contentSHA256: String
    }

    private struct SourceCheckpoint {
        let root: URL
        let revision: String
        let configData: Data
        let indexData: Data
        let config: [String: Any]
        let tensors: [String: TensorReference]
        let shards: [String]
        let shardFingerprints: [String: SourceShardFingerprint]
        let assetSHA256: [String: String]
        let sourceBytes: Int64
    }

    fileprivate struct CompletedShard: Codable {
        let outputFile: String?
        let outputSize: Int64
        let outputSHA256: String?
        let outputKeys: [String]
        let ngramRanges: [NGramRange]
        let ngramSHA256: String?
    }

    private struct State: Codable {
        var formatVersion = Qwen4ExpCheckpointConverter.currentFormatVersion
        var profile = Profile.afmMapped4.rawValue
        var sourceModelID = Qwen4ExpCheckpointConverter.officialModelID
        var sourceRevision: String
        var configSHA256: String
        var indexSHA256: String
        var sourceShards: [String: SourceShardFingerprint]
        var sourceAssets: [String: String]
        var completed: [String: CompletedShard] = [:]
        var weightMap: [String: String] = [:]
    }

    fileprivate struct NGramRange: Codable, Equatable {
        let rowOffset: Int
        let rowCount: Int
    }

    private typealias NGramLocation = NGramRange

    private let source: URL
    private let output: URL
    private let overwrite: Bool
    private let profile: Profile
    private let sourceRevision: String?
    private let progress: ProgressHandler?

    private var stateURL: URL {
        output.standardizedFileURL.appendingPathComponent(".afm-mlx-conversion.json")
    }

    public init(
        source: URL,
        output: URL,
        overwrite: Bool = false,
        profile: Profile = .afmMapped4,
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
        let checkpoint = try loadSource(source, explicitRevision: sourceRevision)
        return inspection(checkpoint)
    }

    public static func inspectResume(
        source: URL,
        output: URL,
        profile: Profile = .afmMapped4,
        sourceRevision: String? = nil
    ) throws -> ResumeInspection {
        let paths = try validatedPaths(source: source, output: output)
        let checkpoint = try loadSource(paths.source, explicitRevision: sourceRevision)
        let stateURL = paths.output.appendingPathComponent(".afm-mlx-conversion.json")
        guard FileManager.default.fileExists(atPath: stateURL.path) else {
            return ResumeInspection(
                sourceRevision: checkpoint.revision,
                verifiedCompletedOutputBytes: 0)
        }
        let state = try decodeState(at: stateURL)
        try validate(state: state, checkpoint: checkpoint, profile: profile)
        var bytes: Int64 = 0
        for completed in state.completed.values {
            guard let file = completed.outputFile,
                  URL(fileURLWithPath: file).lastPathComponent == file
            else { continue }
            let url = paths.output.appendingPathComponent(file)
            guard try completedOutputIsValid(completed, at: url) else { continue }
            bytes += completed.outputSize
        }
        return ResumeInspection(
            sourceRevision: checkpoint.revision,
            verifiedCompletedOutputBytes: bytes)
    }

    public func run() throws {
        let fm = FileManager.default
        let paths = try Self.validatedPaths(source: source, output: output)
        try MLXMetalLibrary.ensureAvailable(verbose: true)
        let checkpoint = try Self.loadSource(
            paths.source, explicitRevision: sourceRevision)
        let details = Self.inspection(checkpoint)
        report("Converting \(Self.officialModelID) at \(details.sourceRevision)")
        report("  source: \(paths.source.path)")
        report("  output: \(paths.output.path)")
        report("  profile: \(profile.rawValue)")
        report("  source shards: \(details.shardCount)")
        report("  n-gram shards: \(details.ngramShardCount)")

        if overwrite, fm.fileExists(atPath: paths.output.path) {
            try fm.removeItem(at: paths.output)
        }
        if fm.fileExists(atPath: paths.output.path),
           !fm.fileExists(atPath: stateURL.path)
        {
            let contents = try fm.contentsOfDirectory(atPath: paths.output.path)
            guard contents.isEmpty else {
                throw ConversionError.outputExists(
                    "Output exists but is not a resumable AFM conversion. Use --overwrite to replace it.")
            }
        }
        try fm.createDirectory(at: paths.output, withIntermediateDirectories: true)

        let configHash = Self.sha256(checkpoint.configData)
        let indexHash = Self.sha256(checkpoint.indexData)
        var state: State
        if fm.fileExists(atPath: stateURL.path) {
            state = try Self.decodeState(at: stateURL)
            try Self.validate(state: state, checkpoint: checkpoint, profile: profile)
        } else {
            state = State(
                sourceRevision: checkpoint.revision,
                configSHA256: configHash,
                indexSHA256: indexHash,
                sourceShards: checkpoint.shardFingerprints,
                sourceAssets: checkpoint.assetSHA256)
            try saveState(state)
        }

        let ngramReferences = checkpoint.tensors.values
            .filter { Self.isNGram($0.name) }
            .sorted { Self.ngramIndex($0.name) < Self.ngramIndex($1.name) }
        let ngramLocations = try Self.makeNGramLocations(ngramReferences)
        let ngramRows = ngramReferences.reduce(0) { $0 + $1.shape[0] }
        let sidecar = try NGramSidecar(
            url: paths.output.appendingPathComponent(Self.ngramSidecarFilename),
            rows: ngramRows,
            dimension: Self.ngramDimension,
            bits: Self.ngramBits,
            groupSize: Self.ngramGroupSize,
            resume: !state.completed.isEmpty)

        for (position, shard) in checkpoint.shards.enumerated() {
            if let completed = state.completed[shard] {
                let outputIsValid: Bool
                if let file = completed.outputFile {
                    outputIsValid = try Self.completedOutputIsValid(
                        completed, at: paths.output.appendingPathComponent(file))
                } else {
                    outputIsValid = completed.outputSize == 0
                        && completed.outputSHA256 == nil
                        && completed.outputKeys.isEmpty
                }
                if outputIsValid,
                   try sidecar.completedRangesAreValid(completed)
                {
                    report("[\(position + 1)/\(checkpoint.shards.count)] \(shard): already converted")
                    continue
                }
                for key in completed.outputKeys { state.weightMap.removeValue(forKey: key) }
                report("[\(position + 1)/\(checkpoint.shards.count)] \(shard): resume data invalid; regenerating")
            }

            report("[\(position + 1)/\(checkpoint.shards.count)] \(shard): converting")
            let sourceURL = checkpoint.root.appendingPathComponent(shard)
            let loaded = try loadArrays(url: sourceURL)
            var converted = [String: MLXArray]()
            var writtenNGramRanges = [NGramRange]()
            for name in loaded.keys.sorted() {
                guard let value = loaded[name] else { continue }
                if Self.isNGram(name) {
                    guard let location = ngramLocations[name] else {
                        throw ConversionError.invalidSource(
                            "No n-gram row location was planned for \(name).")
                    }
                    let quantized = try Self.quantize(
                        value, groupSize: Self.ngramGroupSize, bits: Self.ngramBits)
                    try sidecar.write(
                        rowOffset: location.rowOffset,
                        rowCount: location.rowCount,
                        weight: quantized.weight,
                        scales: quantized.scales,
                        biases: quantized.biases)
                    writtenNGramRanges.append(location)
                    continue
                }
                try Self.convert(name: name, value: value, into: &converted)
            }

            let outputName: String?
            let outputSize: Int64
            let outputHash: String?
            if converted.isEmpty {
                outputName = nil
                outputSize = 0
                outputHash = nil
            } else {
                outputName = String(format: "model-%05d.safetensors", position + 1)
                let destination = paths.output.appendingPathComponent(outputName!)
                let partial = paths.output.appendingPathComponent(
                    ".\(outputName!).partial.safetensors")
                try? fm.removeItem(at: partial)
                try save(
                    arrays: converted,
                    metadata: [
                        "format": "mlx",
                        "afm_source_revision": checkpoint.revision,
                        "afm_conversion_profile": profile.rawValue,
                    ],
                    url: partial)
                try Self.replaceOrMove(partial, to: destination)
                outputSize = Int64(
                    try destination.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
                outputHash = try Self.sha256File(destination)
                for key in converted.keys { state.weightMap[key] = outputName! }
            }
            writtenNGramRanges.sort { $0.rowOffset < $1.rowOffset }
            state.completed[shard] = CompletedShard(
                outputFile: outputName,
                outputSize: outputSize,
                outputSHA256: outputHash,
                outputKeys: converted.keys.sorted(),
                ngramRanges: writtenNGramRanges,
                ngramSHA256: try sidecar.sha256(ranges: writtenNGramRanges))
            try saveState(state)
            Memory.clearCache()
        }

        try copySupportFiles(from: checkpoint.root, to: paths.output)
        try writeConfiguration(checkpoint: checkpoint, output: paths.output)
        try Self.writeIndex(state: state, output: paths.output)
        try saveState(state)
        report("Conversion complete: \(paths.output.path)")
        report("Run: afm mlx -m \(paths.output.path)")
    }

    private static func inspection(_ checkpoint: SourceCheckpoint) -> Inspection {
        let ngramBytes = checkpoint.tensors.values.filter { isNGram($0.name) }
            .reduce(Int64(0)) { partial, reference in
                let rows = Int64(reference.shape.first ?? 0)
                return partial + rows * 100
            }
        let ordinarySource = checkpoint.tensors.values.filter { !isNGram($0.name) }
            .reduce(Int64(0)) { $0 + $1.byteCount }
        let estimated = ordinarySource / 4 + ngramBytes + 5_000_000_000
        return Inspection(
            sourceRevision: checkpoint.revision,
            sourceBytes: checkpoint.sourceBytes,
            estimatedOutputBytes: estimated,
            requiredDestinationFreeBytes: max(
                minimumDestinationFreeBytes, estimated + estimated / 4),
            shardCount: checkpoint.shards.count,
            tensorCount: checkpoint.tensors.count,
            ngramShardCount: checkpoint.tensors.values.filter { isNGram($0.name) }.count)
    }

    private static func loadSource(
        _ source: URL,
        explicitRevision: String?
    ) throws -> SourceCheckpoint {
        let root = source.standardizedFileURL
        let configURL = root.appendingPathComponent("config.json")
        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: configURL.path),
              FileManager.default.fileExists(atPath: indexURL.path)
        else {
            throw ConversionError.invalidSource(
                "Source must contain config.json and model.safetensors.index.json.")
        }
        let configData = try Data(contentsOf: configURL)
        let indexData = try Data(contentsOf: indexURL)
        guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
              config["model_type"] as? String == "qwen4_exp",
              let index = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
              let weightMap = index["weight_map"] as? [String: String]
        else {
            throw ConversionError.invalidSource(
                "Source must be an indexed qwen4_exp checkpoint.")
        }
        let shards = Set(weightMap.values).sorted()
        var references = [String: TensorReference]()
        var shardFingerprints = [String: SourceShardFingerprint]()
        var assetSHA256 = [String: String]()
        var sourceBytes: Int64 = 0
        for name in requiredSupportFiles {
            let url = root.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ConversionError.invalidSource(
                    "Required tokenizer or multimodal asset \(name) is missing.")
            }
            assetSHA256[name] = try sha256File(url)
        }
        for shard in shards {
            guard URL(fileURLWithPath: shard).lastPathComponent == shard else {
                throw ConversionError.invalidSource("Unsafe source shard path \(shard).")
            }
            let url = root.appendingPathComponent(shard)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ConversionError.invalidSource("Source shard is missing: \(shard).")
            }
            let size = Int64(
                try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
            sourceBytes += size
            shardFingerprints[shard] = SourceShardFingerprint(
                size: size, contentSHA256: try sha256File(url))
            let header = try AFMSafetensorHeader(url: url)
            for tensor in header.tensors {
                guard weightMap[tensor.name] == shard else { continue }
                references[tensor.name] = TensorReference(
                    name: tensor.name,
                    shard: shard,
                    dtype: tensor.dtype,
                    shape: tensor.shape,
                    byteCount: Int64(tensor.byteCount))
            }
        }
        guard references.count == weightMap.count else {
            throw ConversionError.invalidSource(
                "The source index and SafeTensor headers disagree (\(weightMap.count) indexed, \(references.count) found).")
        }
        let ngrams = references.values.filter { isNGram($0.name) }
        guard ngrams.count == 128,
              ngrams.allSatisfy({ $0.dtype == .bfloat16 && $0.shape.count == 2 && $0.shape[1] == ngramDimension })
        else {
            throw ConversionError.invalidSource(
                "Expected 128 BF16 n-gram shards with row width \(ngramDimension).")
        }
        return SourceCheckpoint(
            root: root,
            revision: try resolveRevision(root: root, explicit: explicitRevision),
            configData: configData,
            indexData: indexData,
            config: config,
            tensors: references,
            shards: shards,
            shardFingerprints: shardFingerprints,
            assetSHA256: assetSHA256,
            sourceBytes: sourceBytes)
    }

    private static func convert(
        name: String,
        value: MLXArray,
        into output: inout [String: MLXArray]
    ) throws {
        let mapped = mappedName(name)
        // The official checkpoint carries three small I64 PLE routing tables.
        // They are runtime metadata, not quantizable parameters, and the
        // reference conversion preserves them byte-for-byte.
        if value.dtype == .int64 {
            output[mapped] = contiguous(value)
            return
        }
        guard value.dtype == .bfloat16 || value.dtype == .float16 || value.dtype == .float32 else {
            throw ConversionError.unsupportedTensor(
                "Unsupported source dtype \(value.dtype) for \(name).")
        }
        // Preserve the multimodal tower exactly as the reference pack does.
        // Quantizing these matrices changes vision behavior and is outside the
        // fast Qwen Next text-runtime profile.
        if mapped.hasPrefix("model.visual.") {
            output[mapped] = contiguous(value)
            return
        }
        if mapped.hasSuffix(".mlp.experts.gate_up_proj") {
            guard value.ndim == 3, value.shape[1].isMultiple(of: 2) else {
                throw ConversionError.unsupportedTensor("Invalid fused expert shape for \(name).")
            }
            let half = value.shape[1] / 2
            let base = String(mapped.dropLast("experts.gate_up_proj".count)) + "switch_mlp."
            try emitQuantized(
                base: base + "gate_proj", value: value[0..., 0..<half, 0...], into: &output)
            try emitQuantized(
                base: base + "up_proj", value: value[0..., half..., 0...], into: &output)
            return
        }
        if mapped.hasSuffix(".mlp.experts.down_proj") {
            let base = String(mapped.dropLast("experts.down_proj".count))
                + "switch_mlp.down_proj"
            try emitQuantized(base: base, value: value, into: &output)
            return
        }

        var transformed = value
        if mapped.hasSuffix("conv1d.weight"), value.ndim == 3 {
            transformed = value.swappedAxes(1, 2)
        }
        if foldedNormSuffixes.contains(where: { mapped.hasSuffix($0) }) {
            transformed = transformed.asType(.float32) + 1.0
            transformed = transformed.asType(.bfloat16)
        }
        if shouldQuantize(name: mapped, value: transformed) {
            let quantizationBits = mapped == "language_model.lm_head.weight"
                ? lmHeadBits : bits
            let base = mapped.hasSuffix(".weight")
                ? String(mapped.dropLast(".weight".count)) : mapped
            try emitQuantized(
                base: base,
                value: transformed,
                bits: quantizationBits,
                into: &output)
        } else {
            output[mapped] = contiguous(transformed)
        }
    }

    private static func shouldQuantize(name: String, value: MLXArray) -> Bool {
        guard value.ndim == 2, value.shape[0] >= 32,
              value.shape[1].isMultiple(of: groupSize)
        else { return false }
        return name.hasSuffix(".weight")
    }

    private static func emitQuantized(
        base: String,
        value: MLXArray,
        bits: Int = bits,
        into output: inout [String: MLXArray]
    ) throws {
        let q = try quantize(value, groupSize: groupSize, bits: bits)
        output[base + ".weight"] = q.weight
        output[base + ".scales"] = q.scales
        output[base + ".biases"] = q.biases
    }

    private static func quantize(
        _ value: MLXArray,
        groupSize: Int,
        bits: Int
    ) throws -> (weight: MLXArray, scales: MLXArray, biases: MLXArray) {
        try GLM5NextCheckpointConverter.affineQuantize(
            value, groupSize: groupSize, bits: bits)
    }

    static func mappedName(_ name: String) -> String {
        if name.hasPrefix("model.language_model.") {
            return "language_model.model." + name.dropFirst("model.language_model.".count)
        }
        if name.hasPrefix("mtp.") {
            return "language_model.mtp." + name.dropFirst("mtp.".count)
        }
        if name == "lm_head.weight" {
            return "language_model.lm_head.weight"
        }
        return name
    }

    private static func isNGram(_ name: String) -> Bool {
        name.contains(ngramMarker)
    }

    private static func ngramIndex(_ name: String) -> Int {
        guard let range = name.range(of: ngramMarker) else { return .max }
        let suffix = name[range.upperBound...]
        return Int(suffix.prefix(while: \Character.isNumber)) ?? .max
    }

    private static func makeNGramLocations(
        _ references: [TensorReference]
    ) throws -> [String: NGramLocation] {
        var result = [String: NGramLocation]()
        var offset = 0
        for (expected, reference) in references.enumerated() {
            guard ngramIndex(reference.name) == expected else {
                throw ConversionError.invalidSource(
                    "N-gram shard indices must be contiguous from zero; expected \(expected) at \(reference.name).")
            }
            let rows = reference.shape[0]
            result[reference.name] = NGramLocation(rowOffset: offset, rowCount: rows)
            offset += rows
        }
        return result
    }

    private static func validatedPaths(
        source: URL,
        output: URL
    ) throws -> (source: URL, output: URL) {
        do {
            let paths = try CheckpointConversionPathSafety.validate(
                source: source, output: output)
            return (paths.source, paths.output)
        } catch CheckpointConversionPathSafety.PathError.nonLocal {
            throw ConversionError.invalidSource(
                "Qwen Next conversion requires local filesystem paths; remote download is not supported.")
        } catch {
            throw ConversionError.unsafeOutput(
                "Conversion output cannot be a filesystem or volume root, and source/output must be separate directories with neither containing the other, including through symlinks.")
        }
    }

    private static func resolveRevision(root: URL, explicit: String?) throws -> String {
        let inferred = inferredRevision(root: root)
        if let explicit {
            guard isCommitRevision(explicit) else {
                throw ConversionError.invalidSource(
                    "--source-revision must be a full 40-character hexadecimal commit revision.")
            }
            let normalized = explicit.lowercased()
            if let inferred, inferred != normalized {
                throw ConversionError.invalidSource(
                    "--source-revision conflicts with locally recorded Hugging Face provenance.")
            }
            return normalized
        }
        guard let inferred else {
            throw ConversionError.invalidSource(
                "Cannot prove the local checkpoint revision. Pass --source-revision with the official 40-character Hugging Face commit.")
        }
        return inferred
    }

    private static func inferredRevision(root: URL) -> String? {
        let components = root.pathComponents
        if let snapshots = components.lastIndex(of: "snapshots"),
           snapshots + 1 < components.count,
           isCommitRevision(components[snapshots + 1])
        {
            return components[snapshots + 1].lowercased()
        }
        for relative in [
            ".cache/huggingface/download/config.json.metadata",
            ".cache/huggingface/download/model.safetensors.index.json.metadata",
        ] {
            let url = root.appendingPathComponent(relative)
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
            if let revision = text.split(whereSeparator: \Character.isWhitespace)
                .map(String.init).first(where: isCommitRevision)
            {
                return revision.lowercased()
            }
        }
        return nil
    }

    private static func isCommitRevision(_ value: String) -> Bool {
        value.count == 40 && value.allSatisfy(\.isHexDigit)
    }

    private func copySupportFiles(from source: URL, to output: URL) throws {
        let fm = FileManager.default
        for name in Self.copiedSupportFiles {
            let from = source.appendingPathComponent(name)
            let to = output.appendingPathComponent(name)
            guard fm.fileExists(atPath: from.path) else {
                if Self.requiredSupportFiles.contains(name) {
                    throw ConversionError.invalidSource(
                        "Required tokenizer or multimodal asset \(name) is missing.")
                }
                continue
            }
            try Self.writeAtomically(try Data(contentsOf: from), to: to)
        }
    }

    private func writeConfiguration(
        checkpoint: SourceCheckpoint,
        output: URL
    ) throws {
        var object = checkpoint.config
        let quantization: [String: Any] = [
            "group_size": Self.groupSize,
            "bits": Self.bits,
            "mode": "affine",
        ]
        object["quantization"] = quantization
        object["quantization_config"] = quantization
        object["ngram_table"] = [
            "file": Self.ngramSidecarFilename,
            "bits": Self.ngramBits,
            "group_size": Self.ngramGroupSize,
        ]
        object["language_model_only"] = false
        object["afm_conversion"] = [
            "format_version": Self.currentFormatVersion,
            "profile": profile.rawValue,
            "source_model": Self.officialModelID,
            "source_revision": checkpoint.revision,
            "source_format": "safetensors-bf16",
            "output_bits": Self.bits,
            "output_group_size": Self.groupSize,
            "lm_head_bits": Self.lmHeadBits,
            "ngram_bits": Self.ngramBits,
            "ngram_group_size": Self.ngramGroupSize,
            "ngram_storage": "mapped-safetensor-sidecar",
            "vision_preserved": true,
            "mtp_preserved": true,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try Self.writeAtomically(data, to: output.appendingPathComponent("config.json"))
    }

    private static func writeIndex(state: State, output: URL) throws {
        let total = try Set(state.weightMap.values).reduce(Int64(0)) { partial, name in
            partial + Int64(
                try output.appendingPathComponent(name)
                    .resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
        }
        let object: [String: Any] = [
            "metadata": [
                "total_size": total,
                "afm_source_revision": state.sourceRevision,
            ],
            "weight_map": state.weightMap,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try writeAtomically(
            data, to: output.appendingPathComponent("model.safetensors.index.json"))
    }

    private func saveState(_ state: State) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try Self.writeAtomically(try encoder.encode(state), to: stateURL)
    }

    private static func decodeState(at url: URL) throws -> State {
        do { return try JSONDecoder().decode(State.self, from: Data(contentsOf: url)) }
        catch {
            throw ConversionError.sourceMismatch(
                "The conversion resume manifest is invalid; use --overwrite after verifying the destination.")
        }
    }

    private static func validate(
        state: State,
        checkpoint: SourceCheckpoint,
        profile: Profile
    ) throws {
        guard state.formatVersion == currentFormatVersion,
              state.profile == profile.rawValue,
              state.sourceModelID == officialModelID,
              state.sourceRevision == checkpoint.revision,
              state.configSHA256 == sha256(checkpoint.configData),
              state.indexSHA256 == sha256(checkpoint.indexData),
              state.sourceShards == checkpoint.shardFingerprints,
              state.sourceAssets == checkpoint.assetSHA256,
              state.completed.keys.allSatisfy({ checkpoint.shards.contains($0) })
        else {
            throw ConversionError.sourceMismatch(
                "Source configuration, index, shards, support assets, revision, profile, or conversion format changed; use --overwrite after verification.")
        }
    }

    private static func completedOutputIsValid(
        _ completed: CompletedShard,
        at url: URL
    ) throws -> Bool {
        guard let expectedHash = completed.outputSHA256,
              FileManager.default.fileExists(atPath: url.path),
              Int64(try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? -1)
                == completed.outputSize,
              try sha256File(url) == expectedHash,
              let header = try? AFMSafetensorHeader(url: url),
              Set(header.tensors.map(\.name)) == Set(completed.outputKeys)
        else { return false }
        return true
    }

    private static func writeAtomically(_ data: Data, to url: URL) throws {
        let partial = url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.lastPathComponent).partial")
        try data.write(to: partial, options: .atomic)
        try replaceOrMove(partial, to: url)
    }

    private static func replaceOrMove(_ source: URL, to destination: URL) throws {
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

    private func report(_ message: String) {
        progress?(message)
    }
}

private final class NGramSidecar {
    private let handle: FileHandle
    private let dataOffset: UInt64
    private let weightOffset: UInt64
    private let scalesOffset: UInt64
    private let biasesOffset: UInt64
    private let weightRowBytes: Int
    private let scaleRowBytes: Int

    init(
        url: URL,
        rows: Int,
        dimension: Int,
        bits: Int,
        groupSize: Int,
        resume: Bool
    ) throws {
        weightRowBytes = dimension * bits / 8
        scaleRowBytes = dimension / groupSize * 2
        let weightBytes = rows * weightRowBytes
        let scaleBytes = rows * scaleRowBytes
        let header: [String: Any] = [
            "__metadata__": [
                // Preserve the established mapped-table wire format consumed
                // by the self-contained MLX Swift loader.
                "format": "mlx-serve-ngram",
                "bits": String(bits),
                "group_size": String(groupSize),
            ],
            "weight": [
                "dtype": "U32", "shape": [rows, dimension * bits / 32],
                "data_offsets": [0, weightBytes],
            ],
            "scales": [
                "dtype": "BF16", "shape": [rows, dimension / groupSize],
                "data_offsets": [weightBytes, weightBytes + scaleBytes],
            ],
            "biases": [
                "dtype": "BF16", "shape": [rows, dimension / groupSize],
                "data_offsets": [weightBytes + scaleBytes, weightBytes + 2 * scaleBytes],
            ],
        ]
        var headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        headerData.append(contentsOf: repeatElement(UInt8(ascii: " "), count: (8 - headerData.count % 8) % 8))
        dataOffset = UInt64(8 + headerData.count)
        weightOffset = dataOffset
        scalesOffset = dataOffset + UInt64(weightBytes)
        biasesOffset = scalesOffset + UInt64(scaleBytes)
        let expectedSize = biasesOffset + UInt64(scaleBytes)

        let fm = FileManager.default
        var reuseExisting = resume && fm.fileExists(atPath: url.path)
        if reuseExisting {
            let size = UInt64(try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
            let input = try FileHandle(forReadingFrom: url)
            let prefix = try input.read(upToCount: Int(dataOffset)) ?? Data()
            try input.close()
            var littleEndian = UInt64(headerData.count).littleEndian
            let expectedPrefix = withUnsafeBytes(of: &littleEndian) {
                Data($0) + headerData
            }
            reuseExisting = size == expectedSize && prefix == expectedPrefix
        }
        if !reuseExisting {
            try? fm.removeItem(at: url)
            fm.createFile(atPath: url.path, contents: nil)
            let output = try FileHandle(forWritingTo: url)
            var littleEndian = UInt64(headerData.count).littleEndian
            try withUnsafeBytes(of: &littleEndian) { try output.write(contentsOf: $0) }
            try output.write(contentsOf: headerData)
            try output.truncate(atOffset: expectedSize)
            try output.close()
        }
        handle = try FileHandle(forUpdating: url)
    }

    deinit { try? handle.close() }

    func write(
        rowOffset: Int,
        rowCount: Int,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray
    ) throws {
        let weightData = weight.asData(access: .copy).data
        let scalesData = scales.asData(access: .copy).data
        let biasesData = biases.asData(access: .copy).data
        guard weightData.count == rowCount * weightRowBytes,
              scalesData.count == rowCount * scaleRowBytes,
              biasesData.count == rowCount * scaleRowBytes
        else {
            throw Qwen4ExpCheckpointConverter.ConversionError.unsupportedTensor(
                "Quantized n-gram shard byte count does not match its planned row range.")
        }
        try handle.seek(toOffset: weightOffset + UInt64(rowOffset * weightRowBytes))
        try handle.write(contentsOf: weightData)
        try handle.seek(toOffset: scalesOffset + UInt64(rowOffset * scaleRowBytes))
        try handle.write(contentsOf: scalesData)
        try handle.seek(toOffset: biasesOffset + UInt64(rowOffset * scaleRowBytes))
        try handle.write(contentsOf: biasesData)
        try handle.synchronize()
    }

    func completedRangesAreValid(
        _ completed: Qwen4ExpCheckpointConverter.CompletedShard
    ) throws -> Bool {
        guard !completed.ngramRanges.isEmpty else {
            return completed.ngramSHA256 == nil
        }
        guard let expected = completed.ngramSHA256 else { return false }
        return try sha256(ranges: completed.ngramRanges) == expected
    }

    func sha256(
        ranges: [Qwen4ExpCheckpointConverter.NGramRange]
    ) throws -> String? {
        guard !ranges.isEmpty else { return nil }
        try handle.synchronize()
        var digest = SHA256()
        for range in ranges.sorted(by: { $0.rowOffset < $1.rowOffset }) {
            try update(
                &digest,
                offset: weightOffset + UInt64(range.rowOffset * weightRowBytes),
                byteCount: range.rowCount * weightRowBytes)
            try update(
                &digest,
                offset: scalesOffset + UInt64(range.rowOffset * scaleRowBytes),
                byteCount: range.rowCount * scaleRowBytes)
            try update(
                &digest,
                offset: biasesOffset + UInt64(range.rowOffset * scaleRowBytes),
                byteCount: range.rowCount * scaleRowBytes)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func update(
        _ digest: inout SHA256,
        offset: UInt64,
        byteCount: Int
    ) throws {
        try handle.seek(toOffset: offset)
        var remaining = byteCount
        while remaining > 0 {
            let requested = min(remaining, 4 * 1024 * 1024)
            guard let data = try handle.read(upToCount: requested), !data.isEmpty else {
                throw Qwen4ExpCheckpointConverter.ConversionError.sourceMismatch(
                    "The n-gram sidecar ended inside a completed range.")
            }
            digest.update(data: data)
            remaining -= data.count
        }
    }
}
