import MLX
import MLXNN
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

final class QwenHCMixDynamicRowsTests: XCTestCase {
    func testChangingRowCountsPreservesNativeArithmeticAndOutputExtent() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let hidden = 256, streams = 4
        // Reuse the same compiled kernel across changing, non-power-of-two
        // extents. Return to a previous extent to catch stale captured counts.
        for rows in [128, 129, 257, 128] {
            let index = MLXArray(0..<(rows * streams * hidden)).asType(.float32)
            let up = sin(index * 0.07).asType(.bfloat16).reshaped(1, rows, streams * hidden)
            let normalized = cos(index * 0.13).asType(.bfloat16).reshaped(up.shape)
            let actual = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.mixGroupedPrefill(
                up: up, normalized: normalized, groupSize: hidden, forceEnabledForTesting: true))
            let expected = (sigmoid(up).reshaped(1, rows, streams, hidden)
                * normalized.reshaped(1, rows, streams, hidden)).mean(axis: -2)
            XCTAssertEqual(actual.shape, [1, rows, hidden])
            XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
        }
    }
}
