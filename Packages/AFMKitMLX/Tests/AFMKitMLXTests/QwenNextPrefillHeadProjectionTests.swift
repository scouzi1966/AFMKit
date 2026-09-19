import CryptoKit
import Darwin
import Foundation
import MLX
import MLXNN
import XCTest
@testable import AFMKitMLX

/// Test-only final-prefill projection experiment. This never loads the model,
/// converts weights, exercises MTP, or measures end-to-end prefill/TTFT/quality.
final class QwenNextPrefillHeadProjectionTests: XCTestCase {
    private enum Fixture {
        static let checkpoint = "/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit"
        static let prefix = "language_model.lm_head."
        static let keys = ["weight", "scales", "biases"].map { prefix + $0 }
        static let widths = [493, 864, 998, 2112, 4150]
        static let warmupPairs = 2
        static let measuredPairs = 6
        static let hiddenSize = 2560
        static let vocabularySize = 248320
        static let groupSize = 64
        static let headBits = 8
        static let gib = 1024 * 1024 * 1024
        static let memoryCap = 8 * gib
        static let cacheCap = 3 * gib
        static let maximumHeaderBytes = 1024 * 1024
    }

    private enum Arm: String, CaseIterable, Hashable {
        case projectAllThenSlice = "project_all_then_slice"
        case sliceThenProject = "slice_then_project"
    }

    private struct TensorHeader: Decodable {
        let dtype: String
        let shape: [Int]
        let data_offsets: [Int]
    }

    private struct Quantization: Decodable {
        let group_size: Int
        let bits: Int
        let mode: String
    }

    private struct Configuration: Decodable {
        struct Text: Decodable {
            let hidden_size: Int
            let vocab_size: Int
            let dtype: String
            let tie_word_embeddings: Bool
        }
        let model_type: String
        let text_config: Text
        let quantization: Quantization
        let quantization_config: Quantization
    }

    private func require(_ condition: Bool, _ message: String) throws {
        guard condition else {
            throw NSError(domain: "QwenNextPrefillHeadProjectionTests", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: message])
        }
    }

    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func fileSHA256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while let data = try handle.read(upToCount: Fixture.maximumHeaderBytes), !data.isEmpty {
            hash.update(data: data)
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func header(_ url: URL) throws -> [String: TensorHeader] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let prefix = try XCTUnwrap(try handle.read(upToCount: 8))
        try require(prefix.count == 8, "Truncated safetensors header length")
        let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * $1.offset) }
        try require(length > 0 && length <= Fixture.maximumHeaderBytes, "Unsafe safetensors header length")
        let data = try XCTUnwrap(try handle.read(upToCount: Int(length)))
        try require(data.count == length, "Truncated safetensors header")
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        object.removeValue(forKey: "__metadata__")
        return try JSONDecoder().decode([String: TensorHeader].self,
            from: JSONSerialization.data(withJSONObject: object))
    }

    /// Conservative snapshot of free + inactive pages, excluding compressed,
    /// speculative and purgeable counts to avoid counting overlapping categories.
    private func reclaimableMemoryBytes() throws -> UInt64 {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size
            / MemoryLayout<integer_t>.size)
        let host = mach_host_self()
        defer { mach_port_deallocate(mach_task_self_, host) }
        let status = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(host, HOST_VM_INFO64, $0, &count)
            }
        }
        try require(status == KERN_SUCCESS, "Cannot establish available-memory budget")
        return (UInt64(stats.free_count) + UInt64(stats.inactive_count)) * UInt64(getpagesize())
    }

    private func freshOutput(_ path: String, checkpoint: URL) throws -> URL {
        let output = URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL.resolvingSymlinksInPath()
        try require(path.hasPrefix("/Volumes/") && output.path.hasPrefix("/Volumes/"),
                    "Evidence must be on an external /Volumes directory, never /tmp or a RAM disk")
        try require(!output.path.hasPrefix(checkpoint.path + "/") && output != checkpoint,
                    "Evidence must not be written inside the checkpoint")
        var ancestor = output.deletingLastPathComponent()
        var isDirectory: ObjCBool = false
        try require(FileManager.default.fileExists(atPath: ancestor.path, isDirectory: &isDirectory)
                    && isDirectory.boolValue, "Create the external evidence parent directory first")
        while ancestor.path != "/" {
            try require(!FileManager.default.fileExists(atPath: ancestor.appendingPathComponent(".git").path),
                        "Evidence must be outside repositories")
            ancestor.deleteLastPathComponent()
        }
        // mkdir fails on EEXIST, including a directory created after the checks.
        try require(mkdir(output.path, S_IRWXU) == 0,
                    "Output must be a fresh directory; mkdir failed for \(output.path), errno=\(errno)")
        return output
    }

    /// No random state, model data, or quality fixture: dyadic FP32 values are
    /// rounded once to BF16, then the SAME materialized array feeds both arms.
    private func syntheticHidden(width: Int, dimensions: Int, dtype: DType = .bfloat16) -> MLXArray {
        let values = (0..<(width * dimensions)).map { index in
            Float((index * 17 + 11) % 257 - 128) / 64
        }
        return MLXArray(values).reshaped(1, width, dimensions).asType(dtype)
    }

    private func project(_ arm: Arm, head: QuantizedLinear, hidden: MLXArray) -> MLXArray {
        let last = (hidden.dim(1) - 1)..<hidden.dim(1)
        switch arm {
        case .projectAllThenSlice:
            // Match ordinary Qwen4Exp forward + the scheduler's final-row slice.
            // Earlier discarded chunk logits can be lazy-pruned; this final
            // remainder's projection is a dependency of the evaluated slice.
            return head(hidden)[0..., last, 0...]
        case .sliceThenProject:
            return head(hidden[0..., last, 0...])
        }
    }

    private func memoryRecord(_ snapshot: Memory.Snapshot) -> [String: Int] {
        ["active_bytes": snapshot.activeMemory, "cache_bytes": snapshot.cacheMemory,
         "peak_active_bytes": snapshot.peakMemory]
    }

    private func sample(_ arm: Arm, head: QuantizedLinear, hidden: MLXArray,
                        stream: MLX.Stream) throws -> (record: [String: Any], logits: [Float]) {
        stream.synchronize() // Drain prior validation/copies before resetting peak or starting the timer.
        Memory.peakMemory = 0
        let before = Memory.snapshot()
        let start = DispatchTime.now().uptimeNanoseconds
        let output = project(arm, head: head, hidden: hidden) // Fresh graph for EVERY sample.
        eval(output)
        stream.synchronize() // Completion is part of latency, not deferred until the CPU read.
        let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
        let after = Memory.snapshot()
        try require(output.shape == [1, 1, head.shape.0] && output.dtype == .bfloat16,
                    "Projection output shape/dtype changed")
        // Conversion and full-row CPU transfer deliberately occur after timing/peak capture.
        let logits = output.asType(.float32).asArray(Float.self)
        let finite = logits.allSatisfy(\.isFinite)
        let argmax = finite ? logits.indices.max(by: { logits[$0] < logits[$1] }) : nil
        return (["arm": arm.rawValue, "milliseconds": milliseconds, "finite": finite,
                 "argmax": argmax.map { $0 as Any } ?? NSNull(),
                 "output_shape": output.shape, "output_dtype": String(describing: output.dtype),
                 "memory_before": memoryRecord(before), "memory_after": memoryRecord(after)], logits)
    }

    func testSmallProjectionShapesAndRowSelection() throws {
        // Tiny FP32 oracle checks slice placement independently of checkpoint BF16 drift.
        Device.withDefaultDevice(.cpu) {
            let dimensions = 64
            let vocabulary = 32
            let width = 3
            let head = QuantizedLinear(
                weight: MLXArray(Array(repeating: UInt32(0x01010101), count: vocabulary * dimensions / 4))
                    .reshaped(vocabulary, dimensions / 4),
                scales: MLXArray.ones([vocabulary, 1]), biases: MLXArray.zeros([vocabulary, 1]),
                groupSize: dimensions, bits: 8)
            let hidden = syntheticHidden(width: width, dimensions: dimensions, dtype: .float32)
            eval(head, hidden)
            let full = head(hidden)
            XCTAssertEqual(full.shape, [1, width, vocabulary])
            let expected = hidden[0, width - 1].sum().item(Float.self)
            for arm in Arm.allCases {
                let logits = project(arm, head: head, hidden: hidden)
                XCTAssertEqual(logits.shape, [1, 1, vocabulary])
                XCTAssertLessThan(abs(logits - expected).max().item(Float.self), 1e-5)
            }
        }
    }

    func testOptionalExactCheckpointFinalPrefillHeadProjection() throws {
        let env = ProcessInfo.processInfo.environment
        guard let checkpointPath = env["AFM_QWEN_PREFILL_HEAD_MODEL"], !checkpointPath.isEmpty,
              let outputPath = env["AFM_QWEN_PREFILL_HEAD_OUT"], !outputPath.isEmpty else {
            throw XCTSkip("Test-only benchmark requires AFM_QWEN_PREFILL_HEAD_MODEL and AFM_QWEN_PREFILL_HEAD_OUT")
        }
        #if DEBUG
        throw XCTSkip("Latency experiment requires a Release test build through Scripts/swiftpm-reliable.sh")
        #else
        let checkpoint = URL(fileURLWithPath: checkpointPath, isDirectory: true)
            .standardizedFileURL.resolvingSymlinksInPath()
        let expectedCheckpoint = URL(fileURLWithPath: Fixture.checkpoint, isDirectory: true).resolvingSymlinksInPath()
        try require(checkpoint == expectedCheckpoint, "This benchmark is restricted to \(Fixture.checkpoint)")
        let configData = try Data(contentsOf: checkpoint.appendingPathComponent("config.json"))
        let indexData = try Data(contentsOf: checkpoint.appendingPathComponent("model.safetensors.index.json"))
        let config = try JSONDecoder().decode(Configuration.self, from: configData)
        try require(config.model_type == "qwen4_exp" && config.text_config.dtype == "bfloat16"
                    && config.text_config.hidden_size == Fixture.hiddenSize
                    && config.text_config.vocab_size == Fixture.vocabularySize
                    && !config.text_config.tie_word_embeddings, "Unexpected checkpoint architecture")
        for quantization in [config.quantization, config.quantization_config] {
            try require(quantization.bits == 4 && quantization.group_size == Fixture.groupSize
                        && quantization.mode == "affine", "Unexpected checkpoint quantization configuration")
        }
        struct Index: Decodable { let weight_map: [String: String] }
        let index = try JSONDecoder().decode(Index.self, from: indexData)
        let shardNames = Set(try Fixture.keys.map { try XCTUnwrap(index.weight_map[$0]) })
        var descriptors = [String: TensorHeader]()
        var shardURLs = [URL]()
        var shardIdentities = [[String: Any]]()
        var checkpointBytes = 0
        for name in shardNames.sorted() {
            try require(name == URL(fileURLWithPath: name).lastPathComponent && name.hasSuffix(".safetensors"),
                        "Index shard must be a simple safetensors filename")
            let url = checkpoint.appendingPathComponent(name).resolvingSymlinksInPath()
            try require(url.deletingLastPathComponent() == checkpoint, "Shard escapes checkpoint directory")
            let tensors = try header(url)
            let indexedHeadKeys = Set(Fixture.keys.filter { index.weight_map[$0] == name })
            try require(Set(tensors.keys) == indexedHeadKeys,
                        "Refusing to load non-head tensors or an inconsistent head shard")
            let bytes = try XCTUnwrap(try url.resourceValues(forKeys: [.fileSizeKey]).fileSize)
            try require(bytes > 0 && bytes <= Fixture.gib, "Unexpectedly large head shard")
            checkpointBytes += bytes
            descriptors.merge(tensors) { first, _ in first }
            shardURLs.append(url)
            shardIdentities.append(["file": name, "bytes": bytes, "sha256": try fileSHA256(url),
                                    "tensor_keys": tensors.keys.sorted()])
        }
        try require(checkpointBytes <= Fixture.gib, "Head-only checkpoint budget exceeded")
        let weight = try XCTUnwrap(descriptors[Fixture.prefix + "weight"])
        let scales = try XCTUnwrap(descriptors[Fixture.prefix + "scales"])
        let biases = try XCTUnwrap(descriptors[Fixture.prefix + "biases"])
        try require(weight.dtype == "U32" && weight.shape == [Fixture.vocabularySize, 640]
                    && scales.dtype == "BF16" && scales.shape == [Fixture.vocabularySize, 40]
                    && biases.dtype == "BF16" && biases.shape == scales.shape, "Head tensor geometry changed")
        // Match loadWeights' affine inference; the global 4-bit config is NOT the head's precision.
        let actualGroupSize = config.text_config.hidden_size / scales.shape[1]
        let actualBits = weight.shape[1] * 32 / config.text_config.hidden_size
        try require(actualGroupSize == Fixture.groupSize && actualBits == Fixture.headBits,
                    "Original mixed-precision head must be 8-bit/group64")
        for (key, tensor) in descriptors {
            let elementBytes = tensor.dtype == "U32" ? 4 : 2
            try require(tensor.data_offsets.count == 2 && tensor.data_offsets[0] >= 0
                        && tensor.data_offsets[1] - tensor.data_offsets[0] == tensor.shape.reduce(1, *) * elementBytes,
                        "Malformed tensor byte range: \(key)")
        }

        let available = try reclaimableMemoryBytes()
        let physical = ProcessInfo.processInfo.physicalMemory
        let budget = min(UInt64(Fixture.memoryCap), physical / 8, available / 4)
        let maximumWidth = try XCTUnwrap(Fixture.widths.max())
        let fullLogitBytes = maximumWidth * Fixture.vocabularySize * 2
        let hiddenBytes = maximumWidth * Fixture.hiddenSize * 2
        // Two full logit buffers cover live + cached storage; two checkpoint
        // copies cover CPU loading; extra hidden temporaries and 512 MiB slack.
        let estimatedBytes = 2 * fullLogitBytes + 2 * checkpointBytes + 4 * hiddenBytes + Fixture.gib / 2
        try require(UInt64(estimatedBytes) < budget, "Insufficient bounded head-only memory budget")
        let output = try freshOutput(outputPath, checkpoint: checkpoint)
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let originalLimit = Memory.memoryLimit
        let originalCacheLimit = Memory.cacheLimit
        Memory.memoryLimit = Int(budget)
        Memory.cacheLimit = min(Fixture.cacheCap, Int(budget) / 2)
        defer {
            Stream.gpu.synchronize()
            Memory.clearCache()
            Memory.cacheLimit = originalCacheLimit
            Memory.memoryLimit = originalLimit
        }

        var report: [String: Any] = [
            "schema": "qwen-next-final-prefill-head-projection-v1",
            "started_at": ISO8601DateFormatter().string(from: Date()), "checkpoint": checkpoint.path,
            "config_sha256": sha256(configData), "index_sha256": sha256(indexData), "shards": shardIdentities,
            "test_source_sha256": try fileSHA256(URL(fileURLWithPath: #filePath)),
            "source_revision_label": env["AFM_QWEN_PREFILL_HEAD_REVISION"] ?? "unspecified",
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "processor_count": ProcessInfo.processInfo.processorCount,
            "physical_memory_bytes": physical, "free_plus_inactive_bytes_at_start": available,
            "memory_budget_bytes": budget, "estimated_maximum_bytes": estimatedBytes,
            "mlx_cache_limit_bytes": Memory.cacheLimit, "maximum_full_logits_bytes": fullLogitBytes,
            "configured_default_bits": config.quantization.bits, "actual_head_bits": actualBits,
            "actual_head_group_size": actualGroupSize, "head_mode": "affine",
            "warmup_pairs_per_width": Fixture.warmupPairs, "measured_pairs_per_width": Fixture.measuredPairs,
            "hidden_fixture": "SYNTHETIC BF16: Float((flat_index * 17 + 11) % 257 - 128) / 64; materialized once per width",
            "timing_scope": "Fresh graph construction + eval(final logits) + stream completion; prior drain, hidden/head preparation, CPU logits conversion, comparison and JSON excluded",
            "memory_scope": "Process-local MLX active/cache allocations; peak reset before each arm and captured before FP32/CPU validation. Includes resident head+hidden, excludes other processes, RSS and mapped-file pages; allocator limits are restored. Cache cleared before each width, outside timers and before warmups; retained within width up to stated limit. Each arm releases all MLX outputs before the next arm",
            "interpretation": "Synthetic head-only diagnostic; no model-quality, end-to-end prefill/TTFT, MTP, or exact-BF16 parity claim. Earlier discarded chunk logits can be lazy-pruned; the final remainder is required. Timing and argmax/logit differences are observations, not pass thresholds."
        ]
        var rows = [[String: Any]]()
        try Device.withDefaultDevice(.gpu) {
            var arrays = [String: MLXArray]()
            for url in shardURLs { arrays.merge(try loadArrays(url: url)) { first, _ in first } }
            try require(Set(arrays.keys) == Set(Fixture.keys), "Loaded tensors differ from indexed head-only inventory")
            let head = QuantizedLinear(weight: try XCTUnwrap(arrays[Fixture.prefix + "weight"]),
                scales: try XCTUnwrap(arrays[Fixture.prefix + "scales"]),
                biases: try XCTUnwrap(arrays[Fixture.prefix + "biases"]),
                groupSize: actualGroupSize, bits: actualBits)
            try require(head.shape == (Fixture.vocabularySize, Fixture.hiddenSize) && head.bias == nil,
                        "Loaded head dimensions or bias changed")
            eval(head)
            let stream = StreamOrDevice.default.stream
            stream.synchronize()
            for width in Fixture.widths {
                stream.synchronize()
                Memory.clearCache()
                let hidden = syntheticHidden(width: width, dimensions: Fixture.hiddenSize)
                eval(hidden)
                stream.synchronize()
                try require(hidden.shape == [1, width, Fixture.hiddenSize] && hidden.dtype == .bfloat16,
                            "Synthetic hidden shape/dtype changed")
                let lastBefore = hidden[0, width - 1].asType(.float32).asArray(Float.self)
                var samples = [[String: Any]]()
                var timings = [Arm.projectAllThenSlice: [Double](), Arm.sliceThenProject: [Double]()]
                for pair in 0..<(Fixture.warmupPairs + Fixture.measuredPairs) {
                    let order: [Arm] = pair.isMultiple(of: 2)
                        ? [.projectAllThenSlice, .sliceThenProject] : [.sliceThenProject, .projectAllThenSlice]
                    var pairResults = [Arm: (record: [String: Any], logits: [Float])]()
                    for arm in order { pairResults[arm] = try sample(arm, head: head, hidden: hidden, stream: stream) }
                    let full = try XCTUnwrap(pairResults[.projectAllThenSlice])
                    let last = try XCTUnwrap(pairResults[.sliceThenProject])
                    let finite = full.logits.allSatisfy(\.isFinite) && last.logits.allSatisfy(\.isFinite)
                    var deltaMaximum = 0.0
                    var deltaSum = 0.0
                    if finite {
                        for (a, b) in zip(full.logits, last.logits) {
                            let delta = abs(Double(a) - Double(b))
                            deltaMaximum = max(deltaMaximum, delta)
                            deltaSum += delta
                        }
                    }
                    let warmup = pair < Fixture.warmupPairs
                    samples.append(["pair": pair, "warmup": warmup, "order": order.map(\.rawValue),
                        "arms": order.compactMap { pairResults[$0]?.record }, "both_finite": finite,
                        "max_absolute_logit_delta": finite ? deltaMaximum as Any : NSNull(),
                        "mean_absolute_logit_delta": finite ? deltaSum / Double(full.logits.count) as Any : NSNull(),
                        "argmax_agreement": finite && (full.record["argmax"] as? Int) == (last.record["argmax"] as? Int)])
                    XCTAssertTrue(finite, "Non-finite head output at width=\(width), pair=\(pair)")
                    if !warmup {
                        for arm in order { timings[arm, default: []].append(try XCTUnwrap(pairResults[arm]?.record["milliseconds"] as? Double)) }
                    }
                }
                // No timing/parity assertion: GEMM/GEMV BF16 rounding may differ.
                let summaries: [[String: Any]] = Arm.allCases.map { arm in
                    let sorted = timings[arm, default: []].sorted()
                    return ["arm": arm.rawValue, "count": sorted.count, "minimum_ms": sorted.first!,
                            "median_ms": (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2,
                            "maximum_ms": sorted.last!]
                }
                XCTAssertEqual(lastBefore, hidden[0, width - 1].asType(.float32).asArray(Float.self),
                               "Both arms must use the same unchanged materialized hidden row")
                rows.append(["width": width, "hidden_shape": hidden.shape, "last_row_unchanged": true,
                             "samples": samples, "latency_summary": summaries])
                report["widths"] = rows
                let name = "width-\(width).json"
                try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                    .write(to: output.appendingPathComponent(name), options: .withoutOverwriting)
                print("PREFILL_HEAD_PROJECTION width=\(width) evidence=\(output.appendingPathComponent(name).path)")
            }
        }
        report["completed_at"] = ISO8601DateFormatter().string(from: Date())
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("summary.json"), options: .withoutOverwriting)
        #endif
    }
}
