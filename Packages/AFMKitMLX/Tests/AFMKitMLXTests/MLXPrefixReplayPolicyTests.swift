import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
@testable import AFMKitMLX
import XCTest

final class MLXPrefixReplayPolicyTests: XCTestCase {
    func testRestoredSuffixAlwaysHasBatchAndSequenceDimensions() {
        XCTAssertEqual(
            MLXPrefixReplayPolicy.batchedReplayTokens([42]).shape,
            [1, 1]
        )
        XCTAssertEqual(
            MLXPrefixReplayPolicy.batchedReplayTokens([42, 43, 44]).shape,
            [1, 3]
        )
    }

    func testScopedMLXErrorFromWorkerThreadThrowsWithoutTerminatingProcess() {
        let workerFinished = DispatchSemaphore(value: 0)

        XCTAssertThrowsError(
            try withError { errorBox in
                DispatchQueue.global().async {
                    let lhs = MLXArray(0 ..< 10, [2, 5])
                    let rhs = MLXArray(0 ..< 15, [3, 5])
                    MLX.eval(lhs + rhs)
                    workerFinished.signal()
                }
                XCTAssertEqual(workerFinished.wait(timeout: .now() + 5), .success)
                try errorBox.check()
            }
        ) { error in
            XCTAssertTrue(error is MLXError)
        }
    }

    func testDeepseekV4CacheRequiresExactBoundaryRestore() {
        let cache = DeepseekV4Cache(
            slidingWindow: 128,
            compressRatio: 4,
            poolQuantizationEnabled: false
        )

        XCTAssertTrue(MLXPrefixReplayPolicy.requiresExactBoundaryRestore([cache]))
    }

    func testOrdinaryKVCacheAllowsTrimmedDescendantRestore() {
        XCTAssertFalse(
            MLXPrefixReplayPolicy.requiresExactBoundaryRestore([KVCacheSimple()])
        )
    }

    func testQwenStyleMixedArrayCacheSupportsSerialBoundaryCapture() {
        let cache: [KVCache] = [MambaCache(), KVCacheSimple()]
        XCTAssertTrue(MLXPrefixReplayPolicy.supportsSerialBoundaryCapture(cache))
        XCTAssertTrue(MLXPrefixReplayPolicy.supportsExactSnapshotReferenceReuse(cache))
    }

    func testOrdinaryKVAndCacheListDoNotUseScopedSerialBoundaryCapture() {
        XCTAssertFalse(
            MLXPrefixReplayPolicy.supportsSerialBoundaryCapture([KVCacheSimple()])
        )
        XCTAssertFalse(
            MLXPrefixReplayPolicy.supportsSerialBoundaryCapture([
                CacheList(MambaCache(), KVCacheSimple())
            ])
        )
        XCTAssertFalse(
            MLXPrefixReplayPolicy.supportsExactSnapshotReferenceReuse([
                CacheList(MambaCache(), KVCacheSimple())
            ])
        )
    }

    func testPromptMinusOneEntrySupportsExactAndGrowingReplay() {
        let radix = RadixTreeCache(modelID: "recurrent-test")
        radix.insert(tokens: [10, 20, 30], layerStates: [[]])

        let identical = radix.findExactBoundaryMatch([10, 20, 30, 40])
        XCTAssertEqual(identical.prefixLen, 3)
        XCTAssertEqual(identical.sourceTokenCount, 3)
        XCTAssertEqual(
            MLXPrefixReplayPolicy.effectivePrefixLength(
                matchedPrefix: identical.prefixLen,
                inputTokenCount: 4,
                requiresExactBoundary: true,
                forcedSuffix: nil,
                sourceTokenCount: identical.sourceTokenCount
            ),
            3
        )

        let growing = radix.findExactBoundaryMatch([10, 20, 30, 41, 50])
        XCTAssertEqual(growing.prefixLen, 3)
        XCTAssertEqual(growing.sourceTokenCount, 3)
    }

    func testRecurrentCacheRejectsLongerDescendantState() {
        XCTAssertEqual(
            MLXPrefixReplayPolicy.effectivePrefixLength(
                matchedPrefix: 3,
                inputTokenCount: 218,
                requiresExactBoundary: true,
                forcedSuffix: nil,
                sourceTokenCount: 13
            ),
            0
        )
    }

    func testRecurrentCacheAcceptsStateCapturedAtMatchedBoundary() {
        XCTAssertEqual(
            MLXPrefixReplayPolicy.effectivePrefixLength(
                matchedPrefix: 13,
                inputTokenCount: 218,
                requiresExactBoundary: true,
                forcedSuffix: nil,
                sourceTokenCount: 13
            ),
            13
        )
    }

    func testRecurrentExactReplayFallsBackToColdPrefill() {
        XCTAssertEqual(
            MLXPrefixReplayPolicy.effectivePrefixLength(
                matchedPrefix: 218,
                inputTokenCount: 218,
                requiresExactBoundary: true,
                forcedSuffix: nil,
                sourceTokenCount: 218
            ),
            0
        )
    }

    func testUnsafeExactReplayOverrideStillRetainsSuffixToken() {
        XCTAssertEqual(
            MLXPrefixReplayPolicy.effectivePrefixLength(
                matchedPrefix: 218,
                inputTokenCount: 218,
                requiresExactBoundary: true,
                forcedSuffix: 1,
                sourceTokenCount: 218
            ),
            217
        )
    }

    func testQwenRecurrentSnapshotReplaysPromptMinusOneBoundary() throws {
        let model = try makeTinyQwenRecurrentModel()
        let tokens = Array(1 ... 17)

        let coldCache = model.newCache(parameters: nil)
        let coldLogits = model(
            MLXArray(tokens)[.newAxis],
            cache: coldCache
        )
        MLX.eval(coldLogits)

        let captureCache = model.newCache(parameters: nil)
        let prefixLogits = model(
            MLXArray(Array(tokens.dropLast()))[.newAxis],
            cache: captureCache
        )
        let snapshot = captureCache.map {
            MLXPrefixReplayPolicy.snapshotCacheState($0.state)
        }
        MLX.eval(prefixLogits, snapshot.flatMap { $0 })

        var restoredCache = model.newCache(parameters: nil)
        for index in restoredCache.indices {
            restoredCache[index].state = snapshot[index]
        }
        let replayLogits = model(
            MLXArray([tokens.last!])[.newAxis],
            cache: restoredCache
        )
        MLX.eval(replayLogits)

        let coldLast = coldLogits[0, -1, 0...].asArray(Float.self)
        let replayLast = replayLogits[0, -1, 0...].asArray(Float.self)
        XCTAssertEqual(coldLast.count, replayLast.count)
        for (cold, replay) in zip(coldLast, replayLast) {
            XCTAssertEqual(cold, replay, accuracy: 1e-4)
        }

        // Exercise the same transition TokenIterator performs after preparing
        // the restored suffix: feed the sampled token through both advanced
        // caches and compare the next-token logits.
        let sampled = argMax(replayLogits[0, -1, 0...]).item(Int.self)
        let coldTransition = model(MLXArray([sampled])[.newAxis], cache: coldCache)
        let replayTransition = model(MLXArray([sampled])[.newAxis], cache: restoredCache)
        MLX.eval(coldTransition, replayTransition)

        let coldTransitionLast = coldTransition[0, -1, 0...].asArray(Float.self)
        let replayTransitionLast = replayTransition[0, -1, 0...].asArray(Float.self)
        for (cold, replay) in zip(coldTransitionLast, replayTransitionLast) {
            XCTAssertEqual(cold, replay, accuracy: 1e-4)
        }
    }

    private func makeTinyQwenRecurrentModel() throws -> Qwen3_5MoEVL {
        var config = Qwen38PublishedConfigFixture.mxfp8
        config.removeValue(forKey: "quantization")

        var text = config["text_config"] as! [String: Any]
        text["hidden_size"] = 32
        text["num_hidden_layers"] = 4
        text["intermediate_size"] = 64
        text["num_attention_heads"] = 4
        text["num_key_value_heads"] = 2
        text["head_dim"] = 8
        text["linear_num_value_heads"] = 4
        text["linear_num_key_heads"] = 2
        // The Qwen Metal recurrent kernel uses one SIMD group across Dk.
        text["linear_key_head_dim"] = 32
        text["linear_value_head_dim"] = 8
        text["vocab_size"] = 128
        config["text_config"] = text

        var vision = config["vision_config"] as! [String: Any]
        vision["depth"] = 1
        vision["hidden_size"] = 16
        vision["intermediate_size"] = 32
        vision["out_hidden_size"] = 32
        vision["num_heads"] = 4
        vision["patch_size"] = 2
        vision["num_position_embeddings"] = 16
        config["vision_config"] = vision

        let data = try JSONSerialization.data(withJSONObject: config)
        return Qwen3_5MoEVL(
            try JSONDecoder().decode(Qwen3_5MoEVLConfiguration.self, from: data)
        )
    }
}
