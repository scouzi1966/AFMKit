import Foundation
import CryptoKit
import MLX
import MLXNN
import MLXLMCommon
import XCTest
@testable import AFMKitMLX
@testable import MLXLLM

/// Proposal-only experiment. The target model is not loaded or changed here.
/// Does not establish proposal acceptance, final output quality, or API speed.
final class QwenDraftHeadPrecisionScreenTests: XCTestCase {
    private static let steps = 16
    private static let warmRounds = 4
    private static let measuredRounds = 20

    func testOptionalNativeDraftHeadQ8Screen() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["AFM_TEST_QWEN_COMMUNITY_DIRECTORY"],
              let reportPath = environment["AFM_TEST_DRAFT_HEAD_PRECISION_REPORT"] else {
            throw XCTSkip("Explicit checkpoint and external screen output required")
        }
        #if DEBUG
        throw XCTSkip("Release-only timing")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let directory = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
        let report = URL(fileURLWithPath: reportPath).standardizedFileURL.resolvingSymlinksInPath()
        guard report.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: report.path) else {
            throw NSError(domain: "DraftHeadScreen", code: 1)
        }
        let configuration = try JSONDecoder().decode(Qwen4ExpConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let config = configuration.textConfig
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf:
            directory.appendingPathComponent("model.safetensors.index.json"))) as? [String: Any]
        let index = try XCTUnwrap(object?["weight_map"] as? [String: String])
        let selected = index.filter { $0.key.hasPrefix("mtp.") }
        XCTAssertEqual(selected.count, 54, "Frozen native checkpoint contract")
        var raw = [String: MLXArray]()
        for name in Set(selected.values).sorted() {
            let url = directory.appendingPathComponent(name).resolvingSymlinksInPath()
            guard !name.hasPrefix("/"), !name.split(separator: "/").contains(".."),
                  url.path.hasPrefix(directory.path + "/") else {
                throw NSError(domain: "DraftHeadScreen", code: 2)
            }
            for (key, value) in try MLX.loadArrays(url: url) where selected[key] != nil {
                XCTAssertNil(raw.updateValue(value, forKey: key))
            }
        }
        XCTAssertEqual(Set(raw.keys), Set(selected.keys))
        let prepared = Qwen4ExpMTPHead.prepareCheckpointWeights(raw)
        func loadHead() throws -> Qwen4ExpMTPHead {
            let head = Qwen4ExpMTPHead(config)
            quantize(model: head, filter: { path, _ in
                prepared["\(path).scales"] == nil ? nil : (32, 4, .affine)
            })
            try head.update(parameters: ModuleParameters.unflattened(prepared), verify: [.all])
            eval(head)
            return head
        }
        let native = try loadHead()
        let candidate = try loadHead()
        let before = Dictionary(uniqueKeysWithValues: native.parameters().flattened())
        let originalHashes = before.mapValues { SHA256.hash(data: $0.asData().data) }
        let densePaths = Set(candidate.leafModules().flattened().compactMap { path, module -> String? in
            guard !(module is Quantized), let linear = module as? Linear,
                  linear.weight.dtype == .bfloat16, linear.weight.ndim == 2,
                  linear.weight.dim(1).isMultiple(of: 64) else { return nil }
            return path
        })
        XCTAssertEqual(densePaths.count, 12)
        // Only the still-dense proposal matrices. Existing affine triples,
        // normalization, and all target-model tensors remain untouched.
        quantize(model: candidate, groupSize: 64, bits: 8,
            filter: { path, _ in densePaths.contains(path) })
        eval(candidate)
        let after = Dictionary(uniqueKeysWithValues: candidate.parameters().flattened())
        let nativeAfter = Dictionary(uniqueKeysWithValues: native.parameters().flattened())
        XCTAssertEqual(Set(before.keys), Set(nativeAfter.keys))
        for (key, value) in nativeAfter {
            XCTAssertEqual(SHA256.hash(data: value.asData().data), originalHashes[key], key)
        }
        for (path, module) in candidate.leafModules().flattened() where densePaths.contains(path) {
            let layer = try XCTUnwrap(module as? Quantized, path)
            XCTAssertEqual(layer.bits, 8, path)
            XCTAssertEqual(layer.groupSize, 64, path)
            XCTAssertEqual(layer.mode, .affine, path)
        }
        for (key, value) in before {
            XCTAssertTrue(arrayEqual(value, try XCTUnwrap(nativeAfter[key])).item(Bool.self), key)
            if !densePaths.contains(String(key.dropLast(".weight".count))) || !key.hasSuffix(".weight") {
                XCTAssertTrue(arrayEqual(value, try XCTUnwrap(after[key])).item(Bool.self), key)
            }
        }
        let inventory = densePaths.sorted().map { path -> [String: Any] in
            let weight = before[path + ".weight"]!
            return ["path": path, "shape": weight.shape, "original_bytes": weight.nbytes,
                    "q8_bytes": ["weight", "scales", "biases"].reduce(0) {
                        $0 + (after[path + "." + $1]?.nbytes ?? 0)
                    }]
        }
        MLXRandom.seed(927)
        var samples = [[String: Any]]()
        var differences = [[String: Any]]()
        let heads = [native, candidate]
        for prefix in [512, 1024, 2132, 4096] {
            let hidden = MLXRandom.normal([1, 1, config.hiddenSize * config.hcCount]).asType(.bfloat16)
            let embedding = MLXRandom.normal([1, 1, config.hiddenSize]).asType(.bfloat16)
            let keys = (MLXRandom.normal([1, config.kvHeads, prefix, config.headDim]) * 0.1).asType(.bfloat16)
            let values = (MLXRandom.normal(keys.shape) * 0.1).asType(.bfloat16)
            let indexKeys = (MLXRandom.normal([1, prefix, config.indexerHeadDim]) * 0.1).asType(.bfloat16)
            let positions = MLX.arange(prefix, dtype: .int32).reshaped(1, prefix)
            let token = MLXArray([Int32(1)]).reshaped(1, 1)
            eval(hidden, embedding, keys, values, indexKeys, positions, token)
            func cache() -> Qwen4ExpAttentionCache {
                let cache = Qwen4ExpAttentionCache(indexerCompressRatio: config.indexerCompressRatio)
                _ = cache.updateIndexKeys(indexKeys, positionIDs: positions)
                _ = cache.update(keys: keys, values: values)
                eval(cache.state)
                return cache
            }
            func chain(_ arm: Int, _ cache: Qwen4ExpAttentionCache) -> [MLXArray] {
                var stream = hidden
                var allHidden = [MLXArray]()
                for step in 0..<Self.steps {
                    let result = heads[arm](hiddenStream: stream, tokenEmbeddings: embedding,
                        tokenIDs: token, positionIDs: MLXArray([Int32(prefix + step)]).reshaped(1, 1),
                        cache: [cache])
                    stream = result.stream
                    // Real drafting consumes every combined hidden in its
                    // vocabulary projection. Do not let lazy evaluation drop
                    // the first15 final mixers from this head-only screen.
                    allHidden.append(result.hidden)
                }
                return [stream] + allHidden + cache.state
            }
            let nativeResult = chain(0, cache())
            let q8Result = chain(1, cache())
            eval(nativeResult + q8Result)
            for arrays in [nativeResult, q8Result] {
                XCTAssertTrue(arrays.allSatisfy { MLX.isFinite($0).all().item(Bool.self) })
            }
            differences.append(["prefix": prefix,
                "final_hidden_max_error": abs(nativeResult[Self.steps].asType(.float32) - q8Result[Self.steps].asType(.float32)).max().item(Float.self),
                "final_stream_max_error": abs(nativeResult[0].asType(.float32) - q8Result[0].asType(.float32)).max().item(Float.self)])
            for round in 0..<(Self.warmRounds + Self.measuredRounds) {
                let order = round.isMultiple(of: 2) ? [0, 1] : [1, 0]
                for (position, arm) in order.enumerated() {
                    let state = cache()
                    Stream.gpu.synchronize()
                    let start = DispatchTime.now().uptimeNanoseconds
                    let result = chain(arm, state)
                    let built = DispatchTime.now().uptimeNanoseconds
                    eval(result)
                    Stream.gpu.synchronize()
                    let end = DispatchTime.now().uptimeNanoseconds
                    XCTAssertTrue(result.allSatisfy { MLX.isFinite($0).all().item(Bool.self) })
                    let expected = arm == 0 ? nativeResult : q8Result
                    XCTAssertEqual(result.count, expected.count)
                    for (actual, oracle) in zip(result, expected) {
                        XCTAssertTrue(arrayEqual(actual, oracle).item(Bool.self),
                            "same-arm replay prefix=\(prefix) arm=\(arm) round=\(round)")
                    }
                    if round >= Self.warmRounds {
                        samples.append(["prefix": prefix, "arm": arm, "round": round, "position": position,
                            "milliseconds": Double(end - start) / 1e6,
                            "construction_ms": Double(built - start) / 1e6,
                            "evaluation_ms": Double(end - built) / 1e6])
                    }
                }
            }
        }
        XCTAssertEqual(samples.count, 160)
        let evidence: [String: Any] = ["samples": samples, "differences": differences,
            "inventory": inventory, "steps": Self.steps, "seed": 927,
            "checkpoint": directory.path,
            "per_step_hidden_outputs_evaluated": true,
            "same_arm_complete_state_exact_runs": 192,
            "scope": "Real predictor weights, synthetic hidden/cache inputs, dependent head-only chain. No vocabulary projection, actual draft tokens, target verification, or API/quality/acceptance claim."]
        try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
            .write(to: report, options: [.atomic])
        print("DRAFT_HEAD_PRECISION dense_matrices=\(densePaths.count) measured_samples=160")
        #endif
    }
}
