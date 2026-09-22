import MLX
import MLXLMCommon
@testable import AFMKitMLX
import XCTest

final class MLXCompositePrefixRestoreTests: XCTestCase {
    func testFreshCompositeRestoresBothAttentionHistoriesIncludingKeyOnlyValues() {
        Device.withDefaultDevice(.cpu) {
            let attention = KVCacheSimple()
            attention.state = [
                MLXArray([Float(1), 2, 3, 4, 5, 6]).reshaped(1, 1, 3, 2),
                MLXArray([Float(7), 8, 9, 10, 11, 12]).reshaped(1, 1, 3, 2)
            ]
            let indexer = KVCacheSimple()
            indexer.state = [
                MLXArray([Float(13), 14, 15]).reshaped(1, 1, 3, 1),
                MLXArray.zeros([1, 1, 3, 0])
            ]
            let donor = CacheList(attention, indexer)
            let saved = MLXPrefixReplayPolicy.snapshotLayerStates([donor])[0]
            var recipient: KVCache = CacheList(KVCacheSimple(), KVCacheSimple())
            let independent = MLXPrefixReplayPolicy.restoredLayerStates([saved], cache: [recipient])[0]
            MLXPrefixReplayPolicy.installLayerState(independent, into: &recipient, sourceBoundary: 3)

            XCTAssertEqual(recipient.offset, 3)
            XCTAssertEqual(recipient.state.count, 4)
            guard recipient.state.count == saved.count else { return }
            for (restored, original) in zip(recipient.state, saved) {
                XCTAssertEqual(restored.shape, original.shape)
                XCTAssertEqual(restored.asArray(Float.self), original.asArray(Float.self))
            }
            let children = (recipient as! CacheList).caches
            XCTAssertEqual(children.map(\.offset), [3, 3])

            let savedBytes = saved.map { $0.asArray(Float.self) }
            // Advance beyond the saved first-token logits. Both histories must
            // remain present when the next token reaches sparse attention.
            for composite in [donor, recipient as! CacheList] {
                _ = composite[0].update(
                    keys: MLXArray([Float(16), 17]).reshaped(1, 1, 1, 2),
                    values: MLXArray([Float(18), 19]).reshaped(1, 1, 1, 2))
                _ = composite[1].update(
                    keys: MLXArray([Float(20)]).reshaped(1, 1, 1, 1),
                    values: MLXArray.zeros([1, 1, 1, 0]))
            }
            XCTAssertEqual(donor.offset, 4)
            XCTAssertEqual(recipient.offset, 4)
            for (restored, original) in zip(recipient.state, donor.state) {
                XCTAssertEqual(restored.shape, original.shape)
                XCTAssertEqual(restored.asArray(Float.self), original.asArray(Float.self))
            }
            XCTAssertEqual(saved.map { $0.asArray(Float.self) }, savedBytes)
        }
    }

    func testMalformedCompositeStateDiscardsSavedLogitsAndFallsBackToCold() {
        Device.withDefaultDevice(.cpu) {
            let tensor = MLXArray.zeros([1, 1, 3, 2])
            let shorter = MLXArray.zeros([1, 1, 2, 2])
            let cache: [KVCache] = [CacheList(KVCacheSimple(), KVCacheSimple())]
            for state in [[tensor, tensor], [tensor, tensor, tensor],
                          [tensor, tensor, tensor, MLXArray.zeros([1, 1, 2, 0])],
                          [shorter, shorter, shorter, shorter],
                          [tensor, tensor, shorter, shorter]] {
                let match = RadixPrefixMatch(prefixLen: 3, sourceTokenCount: 3,
                    layerStates: [state], layerMetaStates: [[]],
                    promptLogits: MLXArray.zeros([1, 1, 4]))
                let validated = MLXPrefixReplayPolicy.validatedRestoreMatch(match, cache: cache)
                XCTAssertEqual(validated.prefixLen, 0)
                XCTAssertNil(validated.layerStates)
                XCTAssertNil(validated.promptLogits)
                XCTAssertEqual(cache[0].state.count, 0)
            }
        }
    }

    func testRecurrentOffsetUsesTheSavedBoundaryAndOrdinaryKVRestoreIsUnchanged() {
        Device.withDefaultDevice(.cpu) {
            var recurrent: KVCache = ArraysCache(size: 2)
            MLXPrefixReplayPolicy.installLayerState(
                [MLXArray([Float(1)]), MLXArray([Float(2)])],
                into: &recurrent, sourceBoundary: 23)
            XCTAssertEqual(recurrent.offset, 23)
            XCTAssertEqual(recurrent.state.map { $0.asArray(Float.self) }, [[1], [2]])

            var ordinary: KVCache = KVCacheSimple()
            let state = [MLXArray.zeros([1, 1, 4, 2]), MLXArray.ones([1, 1, 4, 2])]
            MLXPrefixReplayPolicy.installLayerState(state, into: &ordinary, sourceBoundary: 4)
            XCTAssertEqual(ordinary.offset, 4)
            XCTAssertEqual(ordinary.state.map { $0.asArray(Float.self) }, state.map { $0.asArray(Float.self) })
        }
    }

    func testRecurrentSubclassKeepsItsOwnOffsetContract() {
        Device.withDefaultDevice(.cpu) {
            var recurrent: KVCache = MambaCache()
            MLXPrefixReplayPolicy.installLayerState(
                [MLXArray([Float(1)]), MLXArray([Float(2)])],
                into: &recurrent, sourceBoundary: 23)
            XCTAssertEqual(recurrent.offset, 0)
            XCTAssertEqual(recurrent.state.map { $0.asArray(Float.self) }, [[1], [2]])
        }
    }
}
