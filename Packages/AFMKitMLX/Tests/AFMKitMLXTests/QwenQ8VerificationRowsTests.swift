import Foundation
import MLX
import MLXNN
import XCTest
@testable import AFMKitMLX
@testable import MLXLMCommon

final class QwenQ8VerificationRowsTests: XCTestCase {
    private func project(_ head: QuantizedLinear, _ x: MLXArray) -> MLXArray {
        VerifyWidthLinear.singletonRows(x, transform: head.callAsFunction)
    }

    func testRouterAndScalarGatePreserveSingletonArithmetic() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(922)
        for k in [512, 2560, 6144] {
            for n in [1, 512] {
                for withBias in [false, true] {
                    let head = QuantizedLinear(weight: MLXRandom.normal([n, k]).asType(.bfloat16),
                        bias: withBias ? MLXRandom.normal([n]).asType(.bfloat16) : nil,
                        groupSize: 64, bits: 8)
                    eval(head)
                    for rows in [2, 4, 7, 8] {
                        // Noncontiguous feature view must not change the dot product.
                        let storage = MLXRandom.normal([1, rows, k * 2]).asType(.bfloat16)
                        let input = storage[.ellipsis, .stride(by: 2)]
                        let oracle = project(head, input)
                        let actual = try XCTUnwrap(VerifyWidthLinear.independentAffineQ8Rows(
                            head, input, forceEnabledForTesting: true))
                        eval(oracle, actual)
                        XCTAssertTrue(arrayEqual(oracle, actual).item(Bool.self),
                            "K=\(k) N=\(n) rows=\(rows) bias=\(withBias)")
                    }
                }
            }
        }
    }

    func testUnsupportedInputsFallBack() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        func head(_ k: Int = 2560, _ n: Int = 512,
                  group: Int = 64, bits: Int = 8, dtype: DType = .bfloat16) -> QuantizedLinear {
            QuantizedLinear(weight: MLXRandom.normal([n, k]).asType(dtype),
                            bias: nil, groupSize: group, bits: bits)
        }
        let supported = head()
        let input = MLXArray.zeros([1, 4, 2560], dtype: .bfloat16)
        // Establish eligibility first: default QuantizedLinear initializers use
        // FP32 scales, which would mask failures of the individual shape guards.
        XCTAssertNotNil(VerifyWidthLinear.independentAffineQ8Rows(
            supported, input, forceEnabledForTesting: true))
        for bad in [input.asType(.float32), MLXArray.zeros([2, 4, 2560], dtype: .bfloat16),
                    MLXArray.zeros([1, 9, 2560], dtype: .bfloat16),
                    MLXArray.zeros([1, 1, 2560], dtype: .bfloat16),
                    MLXArray.zeros([4, 2560], dtype: .bfloat16),
                    MLXArray.zeros([1, 4, 2304], dtype: .bfloat16)] {
            XCTAssertNil(VerifyWidthLinear.independentAffineQ8Rows(supported, bad, forceEnabledForTesting: true))
        }
        for bad in [head(group: 32), head(bits: 4), head(2560, 513), head(dtype: .float32)] {
            XCTAssertNil(VerifyWidthLinear.independentAffineQ8Rows(bad, input, forceEnabledForTesting: true))
        }
        XCTAssertNil(VerifyWidthLinear.independentAffineQ8Rows(
            head(128), MLXArray.zeros([1, 4, 128], dtype: .bfloat16), forceEnabledForTesting: true))
    }

    func testOptionalCompiledRouterGateLatency() throws {
        guard let reportPath = ProcessInfo.processInfo.environment["AFM_TEST_Q8_ROWS_REPORT"] else {
            throw XCTSkip("Explicit external report path required")
        }
        #if DEBUG
        throw XCTSkip("Release-only latency screen")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let report = URL(fileURLWithPath: reportPath).standardizedFileURL.resolvingSymlinksInPath()
        guard report.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: report.path) else {
            throw NSError(domain: "Q8RowsProbe", code: 1)
        }
        MLXRandom.seed(922)
        let router = QuantizedLinear(weight: MLXRandom.normal([512, 2560]).asType(.bfloat16),
                                     bias: nil, groupSize: 64, bits: 8)
        let gate = QuantizedLinear(weight: MLXRandom.normal([1, 2560]).asType(.bfloat16),
                                   bias: nil, groupSize: 64, bits: 8)
        eval(router, gate)
        var samples = [[String: Any]]()
        for width in [2, 4, 7, 8] {
            let input = MLXRandom.normal([1, width, 2560]).asType(.bfloat16)
            eval(input)
            let functions: [@Sendable ([MLXArray]) -> [MLXArray]] = (0..<2).map { arm in
                let body: ([MLXArray]) -> [MLXArray] = { arguments in
                    let x = arguments[0]
                    let r = arm == 1 ? VerifyWidthLinear.independentAffineQ8Rows(
                        router, x, forceEnabledForTesting: true)!
                        : VerifyWidthLinear.singletonRows(x, transform: router.callAsFunction)
                    let g = arm == 1 ? VerifyWidthLinear.independentAffineQ8Rows(
                        gate, x, forceEnabledForTesting: true)!
                        : VerifyWidthLinear.singletonRows(x, transform: gate.callAsFunction)
                    // Feed results forward to prevent dead-code removal and form
                    // a true dependency chain (not cached repeated eval).
                    return [tanh(x + (r.sum(axis: -1, keepDims: true) + sigmoid(g)) * 0.001)]
                }
                return compile(shapeless: false, body)
            }
            func chain(_ arm: Int) -> MLXArray {
                var value = input
                for _ in 0..<48 { value = functions[arm]([value])[0] }
                return value
            }
            let oracle = chain(0)
            let actual = chain(1)
            eval(oracle, actual)
            XCTAssertTrue(arrayEqual(oracle, actual).item(Bool.self), "chain width=\(width)")
            for trial in 0..<14 {
                for arm in trial.isMultiple(of: 2) ? [0, 1] : [1, 0] {
                    Stream.gpu.synchronize()
                    let start = DispatchTime.now().uptimeNanoseconds
                    let output = chain(arm)
                    eval(output)
                    Stream.gpu.synchronize()
                    if trial >= 2 {
                        samples.append(["width": width, "arm": arm, "trial": trial,
                            "milliseconds": Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6])
                    }
                }
            }
            for arm in 0..<2 {
                let values = samples.filter { ($0["width"] as? Int) == width && ($0["arm"] as? Int) == arm }
                    .compactMap { $0["milliseconds"] as? Double }.sorted()
                print("Q8 rows width=\(width) arm=\(arm) median=\(values[values.count / 2])ms")
            }
        }
        let evidence: [String: Any] = ["samples": samples, "seed": 922, "chain_length": 48,
            "scope": "Synthetic compiled router plus scalar gate; same weight bank reused, not live speed or quality"]
        try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
            .write(to: report, options: .withoutOverwriting)
        #endif
    }
}
