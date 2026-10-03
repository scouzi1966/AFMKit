import Foundation
import MLX
import MLXNN
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Component-only experiment: no serving path or selection flag is installed.
/// The original fused primitives and their arithmetic are unchanged. This
/// screens graph construction/replay, not model quality or API throughput.
final class QwenCompiledHCReadScreenTests: XCTestCase {
    private static let hidden = 2560
    private static let streams = 4
    private static let rank = 320
    private static let epsilon: Float = 1e-6
    private static let widths = [2, 4, 7, 8]

    private final class FrozenRead {
        let norm: MLXArray
        let down: QuantizedLinear
        let up: QuantizedLinear
        let inject: QuantizedLinear
        private(set) var traces = 0

        init(seed: Int, group: Int = 32) {
            let columns = QwenCompiledHCReadScreenTests.hidden * QwenCompiledHCReadScreenTests.streams
            let rank = QwenCompiledHCReadScreenTests.rank
            func values(_ shape: [Int], phase: Int) -> MLXArray {
                let count = shape.reduce(1, *)
                return (sin(MLXArray(0..<count).asType(.float32) * 0.013 + Float(seed + phase))
                    * 0.03).asType(.bfloat16).reshaped(shape)
            }
            norm = values([columns], phase: 1)
            down = QuantizedLinear(weight: values([rank, columns], phase: 2),
                                   bias: nil, groupSize: group, bits: 4)
            up = QuantizedLinear(weight: values([columns, rank], phase: 3),
                                 bias: nil, groupSize: group, bits: 4)
            inject = QuantizedLinear(weight: values([4, columns], phase: 4),
                                     bias: nil, groupSize: group, bits: 4)
            // All captured weights are immutable and materialized before trace.
            eval(norm)
            eval(down, up, inject)
        }

        func eager(_ input: MLXArray) -> [MLXArray] {
            let result = Qwen4ExpHyperConnectionFusion.call(
                input: input, normWeight: norm, down: down, up: up, inject: inject,
                hcCount: QwenCompiledHCReadScreenTests.streams,
                hiddenSize: QwenCompiledHCReadScreenTests.hidden,
                epsilon: QwenCompiledHCReadScreenTests.epsilon)!
            return [result.mixed, input, result.injection]
        }

        lazy var compiled: @Sendable ([MLXArray]) -> [MLXArray] = {
            let body: ([MLXArray]) -> [MLXArray] = { [unowned self] inputs in
                self.traces += 1
                return self.eager(inputs[0])
            }
            return compile(shapeless: false, body)
        }()
    }

    private func input(width: Int, seed: Int, strided: Bool = false) -> MLXArray {
        let columns = Self.hidden * Self.streams
        let features = strided ? columns * 2 : columns
        let count = (width + 1) * features
        let storage = sin(MLXArray(0..<count).asType(.float32) * 0.017 + Float(seed))
            .asType(.bfloat16).reshaped(1, width + 1, features)
        let value = strided ? storage[0..., 1..<(width + 1), .stride(by: 2)]
            : storage[0..., 1..<(width + 1), 0...]
        eval(value)
        return value
    }

    private func exact(_ actual: [MLXArray], _ expected: [MLXArray], _ label: String) {
        XCTAssertEqual(actual.count, 3, label)
        for (index, pair) in zip(actual, expected).enumerated() {
            XCTAssertEqual(pair.0.shape, pair.1.shape, label)
            XCTAssertEqual(pair.0.dtype, pair.1.dtype, label)
            XCTAssertTrue(arrayEqual(pair.0, pair.1).item(Bool.self), "\(label) output=\(index)")
        }
    }

    func testAllOutputsMatchChangedInputsViewsAndOwners() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for group in [32, 64] {
            let owners = [FrozenRead(seed: 7, group: group), FrozenRead(seed: 19, group: group)]
            var pending: [([MLXArray], [MLXArray], String)] = []
            for width in Self.widths {
                for strided in [false, true] {
                    for seed in [2, 11] {
                        let x = input(width: width, seed: seed, strided: strided)
                        for (index, owner) in owners.enumerated() {
                            pending.append((owner.compiled([x]), owner.eager(x),
                                "group=\(group) width=\(width) strided=\(strided) seed=\(seed) owner=\(index)"))
                        }
                    }
                }
            }
            for (actual, expected, label) in pending.reversed() { exact(actual, expected, label) }
            let first = owners.map(\.traces)
            for width in Self.widths {
                for strided in [false, true] {
                    let x = input(width: width, seed: 37, strided: strided)
                    for owner in owners { exact(owner.compiled([x]), owner.eager(x), "replay") }
                }
            }
            XCTAssertEqual(owners.map(\.traces), first, "Repeated geometries must not retrace")
            XCTAssertTrue(first.allSatisfy { $0 >= Self.widths.count && $0 <= 2 * Self.widths.count })
        }
    }

    func testPendingOutputsOutliveOwnerWithoutRetainingIt() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        var owner: FrozenRead? = FrozenRead(seed: 31)
        weak var weakOwner = owner
        let x = input(width: 4, seed: 5)
        let expected = owner!.eager(x).map { $0.asArray(Float.self) }
        let actual = owner!.compiled([x])
        owner = nil
        XCTAssertNil(weakOwner, "Compile closure must not create a model ownership cycle")
        for (value, oracle) in zip(actual, expected) {
            XCTAssertEqual(value.asArray(Float.self), oracle,
                           "Compiled output must retain its own dependencies after owner release")
        }
    }

    func testOptionalBalancedConstructionAndEvaluationTiming() throws {
        #if DEBUG
        throw XCTSkip("Release-only component timing")
        #else
        guard let filename = ProcessInfo.processInfo.environment["AFM_TEST_HC_READ_REPORT"] else {
            throw XCTSkip("Explicit external report path required")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let report = URL(fileURLWithPath: filename)
        XCTAssertFalse(FileManager.default.fileExists(atPath: filename))
        let owners = (0..<8).map { FrozenRead(seed: 3 + $0 * 13) }
        var records: [[String: Any]] = []
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        for width in Self.widths {
            let inputs = (0..<3).map { input(width: width, seed: $0 + 41) }
            let expected = inputs.map { x in owners.map { $0.eager(x) } }
            eval(expected.flatMap { $0.flatMap { $0 } })
            // First compiled calls only. Eager-oracle construction/evaluation
            // and equality checks are not included in this interval. This is
            // not a cold Metal compilation measurement: eager kernels are warm.
            let start = now()
            let first = owners.map { $0.compiled([inputs[0]]) }
            eval(first.flatMap { $0 })
            let firstUse = Double(now() - start) / 1e6
            for (actual, oracle) in zip(first, expected[0]) { exact(actual, oracle, "warm") }
            for round in 0..<12 {
                for compiled in (round.isMultiple(of: 2) ? [false, true] : [true, false]) {
                    let start = now()
                    // Multiple distinct immutable owners; no weights read from
                    // disk in the measured interval, no evaluation during build.
                    let results = owners.map { owner in
                        compiled ? owner.compiled([inputs[round % inputs.count]])
                            : owner.eager(inputs[round % inputs.count])
                    }
                    let constructed = now()
                    eval(results.flatMap { $0 })
                    let finished = now()
                    let divisor = Double(owners.count) * 1e6
                    records.append(["width": width, "round": round, "compiled": compiled,
                        "owners": owners.count, "build_ms_per_owner": Double(constructed - start) / divisor,
                        "total_ms_per_owner": Double(finished - start) / divisor,
                        "first_compiled_call_including_eval_all_owners_ms": firstUse])
                    // Validate the actual timed owners/inputs after stopping
                    // the clock, not just a different correctness fixture.
                    for (actual, oracle) in zip(results, expected[round % inputs.count]) {
                        exact(actual, oracle, "timed graph width=\(width) round=\(round) compiled=\(compiled)")
                    }
                }
            }
        }
        let result: [String: Any] = ["scope": "Synthetic fused HC graph replay only; no serving integration or API claim",
            "group_size": 32, "bits": 4, "dtype": "bfloat16", "records": records]
        try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
            .write(to: report, options: .withoutOverwriting)
        #endif
    }
}
