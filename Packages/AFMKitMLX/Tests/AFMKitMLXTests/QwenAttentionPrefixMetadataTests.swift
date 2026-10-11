import MLX
import MLXLMCommon
@testable import MLXLLM
import XCTest
@testable import AFMKitMLX

final class QwenAttentionPrefixMetadataTests: XCTestCase {
    /// Sparse text attention materializes positions without becoming a vision
    /// request. Radix persistence must retain that model-owned provenance.
    func testRadixRestorePreservesImplicitTextPositionsAfterMaterialization() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
            _ = source.update(keys: MLXArray.zeros([1, 1, 8, 4]),
                              values: MLXArray.zeros([1, 1, 8, 4]))
            _ = source.updateIndexKeys(MLXArray.zeros([1, 8, 4]), positionIDs: nil)
            _ = source.ensureSequentialIndexPositionIDs(batchSize: 1)
            _ = source.appendPooledIndexKeys(MLXArray.zeros([1, 2, 4]))
            XCTAssertTrue(source.hasOnlyImplicitIndexPositions)
            let snapshots = MLXPrefixReplayPolicy.snapshotLayerStates([source])
            eval(snapshots.flatMap { $0 })
            let metadata = source.metaState
            var restored: KVCache = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
            MLXPrefixReplayPolicy.installLayerState(snapshots[0], into: &restored, sourceBoundary: 8)
            restored.metaState = metadata
            let attention = try XCTUnwrap(restored as? Qwen4ExpAttentionCache)
            XCTAssertTrue(attention.hasOnlyImplicitIndexPositions,
                          "A radix hit must not silently change the QSA execution path")
            XCTAssertEqual(attention.offset, source.offset)
            XCTAssertEqual(attention.state.count, source.state.count)
        }
    }

    func testRadixRestoreNeverPromotesExplicitPositionsToImplicitText() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
            _ = source.update(keys: MLXArray.zeros([1, 1, 8, 4]),
                              values: MLXArray.zeros([1, 1, 8, 4]))
            _ = source.updateIndexKeys(MLXArray.zeros([1, 8, 4]),
                positionIDs: MLXArray((40..<48).map(Int32.init)).reshaped(1, 8))
            let snapshots = MLXPrefixReplayPolicy.snapshotLayerStates([source])
            eval(snapshots.flatMap { $0 })
            var restored: KVCache = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
            MLXPrefixReplayPolicy.installLayerState(snapshots[0], into: &restored, sourceBoundary: 8)
            restored.metaState = source.metaState
            XCTAssertFalse(try XCTUnwrap(restored as? Qwen4ExpAttentionCache).hasOnlyImplicitIndexPositions)
        }
    }

    func testLegacyTensorOnlyRestoreRemainsConservativeEvenForSequentialPositions() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
            _ = source.update(keys: MLXArray.zeros([1, 1, 8, 4]),
                              values: MLXArray.zeros([1, 1, 8, 4]))
            _ = source.updateIndexKeys(MLXArray.zeros([1, 8, 4]), positionIDs: nil)
            _ = source.ensureSequentialIndexPositionIDs(batchSize: 1)
            let restored = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
            restored.state = source.state
            restored.metaState = []
            XCTAssertFalse(restored.hasOnlyImplicitIndexPositions,
                           "Do not infer text provenance from sequential tensor values")
        }
    }
}
