import Foundation
import MLX
import XCTest

final class CompiledExecutionStreamTests: XCTestCase {
    private static let workerCount = 6
    private static let callsPerWorker = 3
    private static let workerTimeout: TimeInterval = 30

    /// Mutable test observations only; protected independently of MLX's lock.
    private final class Observations: @unchecked Sendable {
        private let lock = NSLock()
        private var traces = 0
        private var outputs: [[Float]] = []

        func traced() { lock.withLock { traces += 1 } }
        func record(_ values: [Float]) { lock.withLock { outputs.append(values) } }
        var traceCount: Int { lock.withLock { traces } }
        var values: [[Float]] { lock.withLock { outputs } }
    }

    private func onFreshThread(_ body: @escaping @Sendable () -> Void) throws {
        let done = DispatchSemaphore(value: 0)
        let thread = Thread {
            body()
            done.signal()
        }
        thread.start()
        guard done.wait(timeout: .now() + Self.workerTimeout) == .success else {
            XCTFail("Compiled execution did not complete on its worker thread")
            throw NSError(domain: "CompiledExecutionStreamTests", code: 1)
        }
    }

    func testIdenticalGraphTracesOnceAcrossExecutorThreads() throws {
        let observations = Observations()
        let function = compile(shapeless: false) { input in
            observations.traced()
            return input * 2 + 1
        }
        for worker in 0..<Self.workerCount {
            try onFreshThread {
                for call in 0..<Self.callsPerWorker {
                    let input = MLXArray(Array(repeating: Float(worker * 10 + call), count: 4))
                    let output = function(input)
                    eval(output)
                    observations.record(output.asArray(Float.self))
                }
            }
        }
        let expected = (0..<Self.workerCount).flatMap { worker in
            (0..<Self.callsPerWorker).map { call in
                Array(repeating: Float((worker * 10 + call) * 2 + 1), count: 4)
            }
        }
        XCTAssertEqual(observations.values, expected)
        print("[CompiledExecutionStream] same-stream traces=\(observations.traceCount) "
              + "workers=\(Self.workerCount) calls=\(expected.count)")
        XCTAssertEqual(observations.traceCount, 1,
                       "Executor-thread migration retraced the same execution-stream graph")
    }

    func testShapesAndDtypesRemainSeparateSpecializations() throws {
        let observations = Observations()
        let function = compile(shapeless: false) { input in
            observations.traced()
            return input * 2 + 1
        }
        let cases: [(Int, DType)] = [(4, .float32), (8, .float32), (4, .float16),
                                     (4, .float32), (8, .float32), (4, .float16)]
        for (count, dtype) in cases {
            try onFreshThread {
                let input = MLXArray(Array(repeating: Float(3), count: count)).asType(dtype)
                let output = function(input)
                eval(output)
                XCTAssertEqual(output.dtype, dtype)
                XCTAssertEqual(output.asType(.float32).asArray(Float.self),
                               Array(repeating: Float(7), count: count))
            }
        }
        XCTAssertEqual(observations.traceCount, 3)
    }

    func testTaskLocalExecutionStreamsRemainSeparateSpecializations() {
        let observations = Observations()
        let function = compile { input in
            observations.traced()
            return input * 2 + 1
        }
        func check() {
            let output = function(MLXArray([Float(1), 2, 3, 4]))
            eval(output)
            XCTAssertEqual(output.asArray(Float.self), [3, 5, 7, 9])
        }
        check()
        XCTAssertEqual(observations.traceCount, 1)
        Stream.withNewDefaultStream(device: .gpu) {
            check()
            check()
            XCTAssertEqual(observations.traceCount, 2)
        }
        check()
        XCTAssertEqual(observations.traceCount, 2, "Returning to a stream must reuse its graph")
        Stream.withNewDefaultStream(device: .gpu) { check() }
        XCTAssertEqual(observations.traceCount, 3)
        check()
        XCTAssertEqual(observations.traceCount, 3)
    }

    func testConcurrentCallersShareOnlyOneSameStreamGraph() throws {
        let observations = Observations()
        let function = compile { input in
            observations.traced()
            return input * 2 + 1
        }
        try onFreshThread {
            DispatchQueue.concurrentPerform(iterations: 18) { index in
                let output = function(MLXArray(Array(repeating: Float(index), count: 4)))
                eval(output)
                observations.record(output.asArray(Float.self))
            }
        }
        XCTAssertEqual(observations.traceCount, 1)
        XCTAssertEqual(observations.values.sorted { $0[0] < $1[0] },
                       (0..<18).map { Array(repeating: Float($0 * 2 + 1), count: 4) })
    }

    func testNestedCompiledCallsCompleteUnderConcurrentEntry() throws {
        let observations = Observations()
        let inner = compile { (input: MLXArray) in input * 2 }
        let outer = compile { (input: MLXArray) in inner(input) + 1 }
        try onFreshThread {
            DispatchQueue.concurrentPerform(iterations: 18) { index in
                let input = MLXArray(Array(repeating: Float(index), count: 4))
                let output = index.isMultiple(of: 2) ? outer(input) : inner(input)
                eval(output)
                observations.record(output.asArray(Float.self))
            }
        }
        let expected = (0..<18).map { index in
            Array(repeating: Float(index * 2 + (index.isMultiple(of: 2) ? 1 : 0)), count: 4)
        }
        XCTAssertEqual(observations.values.sorted { $0[0] < $1[0] },
                       expected.sorted { $0[0] < $1[0] })
    }

    func testCompiledFunctionsDoNotShareOwnersOrRetainReleasedOwner() {
        final class Owner {
            let offset: MLXArray
            init(_ value: Float) { offset = MLXArray(value) }
        }
        weak var releasedOwner: Owner?
        let survivor = compile { (input: MLXArray) in input + 100 }
        autoreleasepool {
            let owner = Owner(7)
            releasedOwner = owner
            let transient = compile { (input: MLXArray) in input + owner.offset }
            let input = MLXArray([Float(1), 2, 3, 4])
            let first = transient(input)
            let second = survivor(input)
            eval(first, second)
            XCTAssertEqual(first.asArray(Float.self), [8, 9, 10, 11])
            XCTAssertEqual(second.asArray(Float.self), [101, 102, 103, 104])
        }
        XCTAssertNil(releasedOwner)
        let output = survivor(MLXArray([Float(1), 2, 3, 4]))
        eval(output)
        XCTAssertEqual(output.asArray(Float.self), [101, 102, 103, 104])
    }
}
