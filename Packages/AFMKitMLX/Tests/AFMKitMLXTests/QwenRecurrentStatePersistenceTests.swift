import Foundation
import MLX
import MLXRandom
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

final class QwenRecurrentStatePersistenceTests: XCTestCase {
    private func inputs(width: Int = 1) -> [MLXArray] {
        MLXRandom.seed(20261025)
        return [
            MLXRandom.normal([1, width, 16, 128]).asType(.bfloat16) * 0.05,
            MLXRandom.normal([1, width, 16, 128]).asType(.bfloat16) * 0.05,
            MLXRandom.normal([1, width, 48, 128]).asType(.bfloat16),
            MLXArray.full([1, width, 48], values: MLXArray(Float(0.95))),
            MLXArray.full([1, width, 48], values: MLXArray(Float(0.4))).asType(.bfloat16),
            MLXRandom.normal([1, 48, 128, 128]),
        ]
    }

    private func update(_ arrays: [MLXArray], fused: Bool) -> (MLXArray, MLXArray) {
        let pair = gatedDeltaKernel(q: arrays[0], k: arrays[1], v: arrays[2],
            g: arrays[3], beta: arrays[4], state: arrays[5], persistStateAsBF16: fused)
        return (pair.0, fused ? pair.1 : pair.1.asType(.bfloat16).asType(.float32))
    }

    func testFusedStoreMatchesMaterializedPersistenceAcrossDecodeSteps() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        var control = inputs(), candidate = control
        XCTAssertEqual(control[0].dtype, .bfloat16)
        XCTAssertEqual(control[5].dtype, .float32)
        for _ in 0..<16 {
            let expected = update(control, fused: false)
            let actual = update(candidate, fused: true)
            eval(expected.0, expected.1, actual.0, actual.1)
            XCTAssertEqual(expected.0.asData(access: .copy).data, actual.0.asData(access: .copy).data)
            XCTAssertEqual(expected.1.asData(access: .copy).data, actual.1.asData(access: .copy).data)
            control[5] = expected.1; candidate[5] = actual.1
        }
    }

    func testMultiTokenFallbackPreservesExistingRecurrenceAndRounding() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for width in [2, 17] {
            let arrays = inputs(width: width)
            let expected = update(arrays, fused: false)
            let actual = update(arrays, fused: true)
            eval(expected.0, expected.1, actual.0, actual.1)
            XCTAssertEqual(expected.0.asData(access: .copy).data, actual.0.asData(access: .copy).data)
            XCTAssertEqual(expected.1.asData(access: .copy).data, actual.1.asData(access: .copy).data)
        }
    }

    func testOptionalAlternatingRecurrenceTiming() throws {
        guard let path = ProcessInfo.processInfo.environment["AFM_QWEN_STATE_PERSISTENCE_TIMING"] else {
            throw XCTSkip("Requires an explicit external timing report")
        }
        #if DEBUG
        throw XCTSkip("Release timing only")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let url = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        XCTAssertTrue(url.path.hasPrefix("/Volumes/"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        let base = inputs()
        let layers = (0..<36).map { i -> [MLXArray] in
            var row = base; row[5] = base[5] + Float(i) * 0.001; return row
        }
        eval(layers.flatMap { $0 })
        var samples = [[String: Any]]()
        for trial in 0..<24 {
            for fused in (trial.isMultiple(of: 2) ? [false, true] : [true, false]) {
                let start = DispatchTime.now().uptimeNanoseconds
                let outputs = layers.flatMap { row -> [MLXArray] in
                    let pair = update(row, fused: fused); return [pair.0, pair.1]
                }
                eval(outputs)
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                if trial >= 4 { samples.append([
                    "trial": trial, "arm": fused ? "fused-store" : "casts", "milliseconds": elapsed]) }
            }
        }
        let data = try JSONSerialization.data(withJSONObject: [
            "scope": "36 independent GDN updates plus persistence; not model decode throughput",
            "samples": samples,
        ], options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .withoutOverwriting)
        #endif
    }
}
