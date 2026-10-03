import Foundation
import MLX
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Test-only attribution of the per-tick concatenate in request-owned GDN
/// batching. Every arm calls the same production recurrence kernel with FP32
/// state; this is not a model-throughput benchmark or a new serving cache.
/// Persistent-group reference: mlx-serve, src/transformer.zig,
/// ssmBindTickState at 1745ffe89e4670f1e0c6de22c75a9875b27399de (MIT).
final class QwenBatchedRecurrentBankTests: XCTestCase {
    private enum Geometry {
        static let keyHeads = 16
        static let valueHeads = 48
        static let headDimension = 128
        static let layers = 36
        static let ticks = 8
        static let trials = 8
        static let warmups = 2
        static let seed: UInt64 = 927
    }

    private enum Mode: String, CaseIterable {
        case concatenate, retainedWithViews, retainedWithoutViews
    }

    private struct Bank {
        var states: [MLXArray]
        var rows: [[MLXArray]] // layer -> request, always current immutable views

        init(_ states: [MLXArray]) {
            self.states = states
            rows = states.map { value in
                (0..<value.dim(0)).map { value[$0..<($0 + 1)] }
            }
        }

        mutating func step(_ inputs: [MLXArray], mode: Mode) -> [MLXArray] {
            var previous = inputs[2]
            var outputs = [MLXArray]()
            for layer in states.indices {
                let state = mode == .concatenate ? concatenated(rows[layer], axis: 0) : states[layer]
                // Serial GPU dependencies across layers and recurrent ticks;
                // nonzero state prevents a constant/zero-state fast path.
                let q = inputs[0] + previous[0..., 0..., 0..<Geometry.keyHeads] * 0.001
                let result = gatedDeltaKernel(q: q, k: inputs[1], v: inputs[2],
                    g: inputs[3], beta: inputs[4], state: state)
                previous = result.0
                states[layer] = result.1
                if mode != .retainedWithoutViews {
                    rows[layer] = (0..<state.dim(0)).map { result.1[$0..<($0 + 1)] }
                }
                outputs.append(result.0)
            }
            return outputs
        }

        mutating func repack(_ membership: [Int]) {
            states = rows.map { layer in concatenated(membership.map { layer[$0] }, axis: 0) }
            rows = states.map { value in membership.indices.map { value[$0..<($0 + 1)] } }
        }
    }

    private func inputs(_ batch: Int) -> [MLXArray] {
        func normalized() -> MLXArray {
            let value = MLXRandom.normal([batch, 1, Geometry.keyHeads, Geometry.headDimension])
            return (value * rsqrt((value * value).sum(axis: -1, keepDims: true) + 1e-6))
                .asType(.bfloat16)
        }
        return [normalized(), normalized(),
            MLXRandom.normal([batch, 1, Geometry.valueHeads, Geometry.headDimension]).asType(.bfloat16),
            MLXRandom.uniform(low: 0.8, high: 0.99, [batch, 1, Geometry.valueHeads]),
            MLXRandom.uniform(low: 0.2, high: 0.8, [batch, 1, Geometry.valueHeads]).asType(.bfloat16)]
    }

    private func states(_ batch: Int, layers: Int) -> [MLXArray] {
        (0..<layers).map { _ in
            MLXRandom.normal([batch, Geometry.valueHeads, Geometry.headDimension, Geometry.headDimension]) * 0.1
        }
    }

    private func assertExact(_ actual: [MLXArray], _ expected: [MLXArray], _ label: String) {
        XCTAssertEqual(actual.count, expected.count, label)
        for (index, pair) in zip(actual, expected).enumerated() {
            XCTAssertEqual(pair.0.shape, pair.1.shape, label)
            XCTAssertEqual(pair.0.dtype, pair.1.dtype, label)
            XCTAssertTrue(MLX.isFinite(pair.0).all().item(Bool.self), "non-finite \(label)")
            XCTAssertTrue(MLX.isFinite(pair.1).all().item(Bool.self), "non-finite oracle \(label)")
            XCTAssertTrue(arrayEqual(pair.0, pair.1).item(Bool.self), "\(label) array=\(index)")
        }
    }

    func testRetainedBankPreservesRowsAcrossTicksRemovalRestoreAndSoloContinuation() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(Geometry.seed)
        let initial = states(15, layers: 2)
        var data = inputs(15)
        eval(initial + data)
        var control = Bank(initial)
        var retained = Bank(initial)
        let frozenInitial = initial.map { $0.asArray(Float.self) }
        var pending: [MLXArray] = []
        var frozenPending: [[Float]] = []
        for tick in 0..<6 {
            if tick > 0 {
                let sizes = [15, 8, 4, 3, 2, 1]
                let members = Array((0..<control.states[0].dim(0)).reversed().prefix(sizes[tick]))
                control.repack(members)
                retained.repack(members)
                data = data.map { $0[MLXArray(members.map(Int32.init))] }
            }
            // A restored request requires repacking before another group tick.
            // No hidden serving owner or revision inference is introduced here.
            if tick == 2 {
                for layer in initial.indices {
                    control.rows[layer][0] = initial[layer][0..<1]
                    retained.rows[layer][0] = initial[layer][0..<1]
                }
                let membership = Array(0..<control.states[0].dim(0))
                control.repack(membership)
                retained.repack(membership)
            }
            for substep in 0..<3 {
                let a = control.step(data, mode: .concatenate)
                let b = retained.step(data, mode: .retainedWithViews)
                eval(a + b + control.states + retained.states)
                assertExact(b, a, "output tick=\(tick) substep=\(substep)")
                assertExact(retained.states, control.states, "state tick=\(tick)")
                assertExact(retained.rows.flatMap { $0 }, control.rows.flatMap { $0 }, "views")
                if tick == 0 && substep == 0 {
                    pending = b + retained.rows.flatMap { $0 }
                    frozenPending = pending.map { $0.asArray(Float.self) }
                }
            }
        }
        XCTAssertEqual(initial.map { $0.asArray(Float.self) }, frozenInitial)
        XCTAssertEqual(pending.map { $0.asArray(Float.self) }, frozenPending,
            "Advancing/removing rows must not overwrite outstanding outputs or snapshots")
    }

    func testOptionalNativeGeometryCopyEliminationScreen() throws {
        guard let path = ProcessInfo.processInfo.environment["AFM_TEST_GDN_BANK_REPORT"] else {
            throw XCTSkip("Explicit external diagnostic report required")
        }
        #if DEBUG
        throw XCTSkip("Release-only diagnostic")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let url = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
        guard url.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: url.path) else {
            throw NSError(domain: "GDNBankScreen", code: 1)
        }
        MLXRandom.seed(Geometry.seed)
        var samples = [[String: Any]]()
        var admissions = [[String: Any]]()
        for batch in [2, 4, 8, 15] {
            let initial = states(batch, layers: Geometry.layers)
            let data = inputs(batch)
            eval(initial + data)
            var oracle = Bank(initial)
            var oracleOutput: [MLXArray] = []
            for _ in 0..<Geometry.ticks {
                oracleOutput = oracle.step(data, mode: .concatenate)
                eval(oracleOutput + oracle.states)
            }
            for trial in 0..<Geometry.trials {
                // Every arm occupies every temporal position twice in the six
                // measured trials; no retained arm is always in the middle.
                let permutations: [[Mode]] = [
                    [.concatenate, .retainedWithViews, .retainedWithoutViews],
                    [.concatenate, .retainedWithoutViews, .retainedWithViews],
                    [.retainedWithViews, .concatenate, .retainedWithoutViews],
                    [.retainedWithViews, .retainedWithoutViews, .concatenate],
                    [.retainedWithoutViews, .concatenate, .retainedWithViews],
                    [.retainedWithoutViews, .retainedWithViews, .concatenate],
                ]
                let order = permutations[(trial + permutations.count - Geometry.warmups) % permutations.count]
                for mode in order {
                    // Admission is outside the steady-state timer, reported separately.
                    var bank = Bank(initial)
                    Stream.gpu.synchronize()
                    let admissionStart = DispatchTime.now().uptimeNanoseconds
                    bank.repack(Array(0..<batch))
                    eval(bank.states)
                    Stream.gpu.synchronize()
                    let admissionEnd = DispatchTime.now().uptimeNanoseconds
                    if mode == .retainedWithoutViews {
                        // A true no-view diagnostic must not pin the admission
                        // bank through obsolete request views for every tick.
                        bank.rows = []
                    }
                    Memory.peakMemory = 0
                    let activeBefore = Memory.activeMemory
                    var lastOutput: [MLXArray] = []
                    var constructionNS: UInt64 = 0
                    let start = DispatchTime.now().uptimeNanoseconds
                    for _ in 0..<Geometry.ticks {
                        let buildStart = DispatchTime.now().uptimeNanoseconds
                        lastOutput = bank.step(data, mode: mode)
                        constructionNS += DispatchTime.now().uptimeNanoseconds - buildStart
                        // Same per-token submission/wait cadence for all arms.
                        // Row slices are views, not independent copy kernels.
                        eval(lastOutput + bank.states)
                    }
                    Stream.gpu.synchronize()
                    let end = DispatchTime.now().uptimeNanoseconds
                    let activeAfter = Memory.activeMemory
                    let peak = Memory.peakMemory
                    assertExact(lastOutput, oracleOutput, "B=\(batch) mode=\(mode) output")
                    assertExact(bank.states, oracle.states, "B=\(batch) mode=\(mode) state")
                    if mode != .retainedWithoutViews {
                        assertExact(bank.rows.flatMap { $0 }, oracle.rows.flatMap { $0 }, "request views")
                    }
                    if trial >= Geometry.warmups {
                        admissions.append(["batch": batch, "trial": trial, "mode": mode.rawValue,
                            "milliseconds": Double(admissionEnd - admissionStart) / 1e6])
                        samples.append(["batch": batch, "trial": trial, "mode": mode.rawValue,
                            "milliseconds_per_tick": Double(end - start) / 1e6 / Double(Geometry.ticks),
                            "construction_ms_per_tick": Double(constructionNS) / 1e6 / Double(Geometry.ticks),
                            "active_before_bytes": activeBefore, "active_after_bytes": activeAfter,
                            "peak_active_bytes": peak,
                            "bank_payload_bytes": initial.reduce(0) { $0 + $1.nbytes }])
                    }
                }
            }
            for mode in Mode.allCases {
                let values = samples.filter { ($0["batch"] as? Int) == batch &&
                    ($0["mode"] as? String) == mode.rawValue }
                    .compactMap { $0["milliseconds_per_tick"] as? Double }.sorted()
                let median = (values[values.count / 2 - 1] + values[values.count / 2]) / 2
                print("GDN bank B=\(batch) mode=\(mode.rawValue) median=\(median) ms/tick")
            }
        }
        let report: [String: Any] = ["samples": samples, "admissions": admissions,
            "seed": Geometry.seed, "ticks": Geometry.ticks, "layers": Geometry.layers,
            "state_dtype": "float32", "activation_dtype": "bfloat16",
            "scope": "Recurrence-only synthetic dependent chain; no weights, convolution, PLE, attention, HTTP or real model quality. Exact recurrence outputs/states verified outside timing. No serving changes."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: url, options: .withoutOverwriting)
        #endif
    }

    func testUnevaluatedViewsAndOutputsSurviveAdvancementAndRowRemoval() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(Geometry.seed)
        let initial = states(4, layers: 2)
        let data = inputs(4)
        eval(initial + data)
        var actual = Bank(initial)
        var expected = Bank(initial)
        var pending: [[MLXArray]] = []
        for _ in 0..<3 {
            let outputs = actual.step(data, mode: .retainedWithViews)
            pending.append(outputs + actual.rows.flatMap { $0 })
        }
        // Repack/continue before submitting any of the retained graphs.
        actual.repack([3, 1])
        let smallerInputs = data.map { $0[MLXArray([Int32(3), 1])] }
        let final = actual.step(smallerInputs, mode: .retainedWithViews)
        eval(final + actual.states + pending.flatMap { $0 })
        for tick in 0..<3 {
            let outputs = expected.step(data, mode: .concatenate)
            eval(outputs + expected.states)
            assertExact(pending[tick], outputs + expected.rows.flatMap { $0 }, "pending tick=\(tick)")
        }
        expected.repack([3, 1])
        let expectedFinal = expected.step(smallerInputs, mode: .concatenate)
        eval(expectedFinal + expected.states)
        assertExact(final + actual.states, expectedFinal + expected.states, "after removal")
    }

    func testRowIdentityAgainstIndependentSingletonOracle() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(Geometry.seed)
        let initial = states(15, layers: 2)
        let fullInputs = inputs(15)
        eval(initial + fullInputs)
        var bank = Bank(initial)
        var active = Array(0..<15)
        // Oracle has stable request IDs and no Bank/repack implementation.
        var byID = Dictionary(uniqueKeysWithValues: active.map { id in
            (id, initial.map { $0[id..<(id + 1)] })
        })
        for (tick, count) in [15, 8, 4, 3, 2, 1].enumerated() {
            if tick > 0 {
                let positions = Array(active.indices.reversed().prefix(count))
                bank.repack(positions)
                active = positions.map { active[$0] }
            }
            if tick == 2 {
                let id = active[0]
                byID[id] = initial.map { $0[id..<(id + 1)] }
                for layer in initial.indices { bank.rows[layer][0] = byID[id]![layer] }
                bank.repack(Array(active.indices))
            }
            let input = fullInputs.map { $0[MLXArray(active.map(Int32.init))] }
            for _ in 0..<3 {
                let outputs = bank.step(input, mode: .retainedWithViews)
                eval(outputs + bank.states)
                for (position, id) in active.enumerated() {
                    let rowInput = fullInputs.map { $0[id..<(id + 1)] }
                    var previous = rowInput[2]
                    for layer in initial.indices {
                        let q = rowInput[0] + previous[0..., 0..., 0..<Geometry.keyHeads] * 0.001
                        let result = gatedDeltaKernel(q: q, k: rowInput[1], v: rowInput[2],
                            g: rowInput[3], beta: rowInput[4], state: byID[id]![layer])
                        previous = result.0
                        byID[id]![layer] = result.1
                        assertExact([outputs[layer][position..<(position + 1)], bank.rows[layer][position]],
                            [result.0, result.1], "identity=\(id) tick=\(tick) layer=\(layer)")
                    }
                }
            }
        }
    }
}
