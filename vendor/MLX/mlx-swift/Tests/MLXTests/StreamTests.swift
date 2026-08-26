// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

private final class StreamTestState: @unchecked Sendable {
    let lock = NSLock()
    var value: MLXArray?
    var result: Int?
    var results: [Int] = []
}

class StreamTests: XCTestCase {

    func testEquatableDevice() {
        let s1 = Device.gpu
        let s2 = Device(.gpu, index: 3)
        let s3 = Device.cpu

        // equality ignores index
        XCTAssertEqual(s1, s2)

        XCTAssertNotEqual(s1, s3)
        XCTAssertNotEqual(s2, s3)
    }

    func testDeviceType() {
        let s1 = Device.gpu
        let s2 = Device(.gpu, index: 3)
        let s3 = Device.cpu

        XCTAssertEqual(s1.deviceType, .gpu)
        XCTAssertEqual(s2.deviceType, .gpu)
        XCTAssertEqual(s3.deviceType, .cpu)
    }

    func testUsingDevice() {
        let defaultDevice = Device.defaultDevice()

        Device.withDefaultDevice(.cpu) {
            // these _should_ be the same
            XCTAssertTrue(Device.defaultDevice().description.contains("cpu"))
            XCTAssertTrue(StreamOrDevice.default.description.contains("cpu"))
        }
        XCTAssertEqual(defaultDevice, Device.defaultDevice())

        Device.withDefaultDevice(.gpu) {
            XCTAssertTrue(Device.defaultDevice().description.contains("gpu"))
            XCTAssertTrue(StreamOrDevice.default.description.contains("gpu"))
        }
        XCTAssertTrue(StreamOrDevice.default.description.contains("gpu"))
    }

    func testDefaultGPUStreamCarriesLazyGraphAcrossThreads() {
        let state = StreamTestState()
        let created = DispatchSemaphore(value: 0)
        let releaseCreator = DispatchSemaphore(value: 0)
        let evaluated = DispatchSemaphore(value: 0)

        // Keep the creator alive while a second OS thread evaluates the lazy
        // graph. This makes the thread hop deterministic rather than relying
        // on DispatchQueue.sync scheduling behavior.
        Thread.detachNewThread {
            state.lock.withLock {
                state.value = MLXArray(1) + 1
            }
            created.signal()
            releaseCreator.wait()
        }

        XCTAssertEqual(created.wait(timeout: .now() + 5), .success)

        Thread.detachNewThread {
            let value = state.lock.withLock { state.value! }
            eval(value)
            state.lock.withLock {
                state.result = value.item(Int.self)
            }
            evaluated.signal()
        }

        XCTAssertEqual(evaluated.wait(timeout: .now() + 30), .success)
        releaseCreator.signal()
        XCTAssertEqual(state.lock.withLock { state.result }, 2)
    }

    func testDefaultGPUStreamSerializesConcurrentEvaluation() {
        let state = StreamTestState()
        let ready = DispatchSemaphore(value: 0)
        let start = DispatchSemaphore(value: 0)
        let finished = DispatchSemaphore(value: 0)

        for input in [2, 3] {
            Thread.detachNewThread {
                // Both OS threads remain alive at this barrier, which makes
                // the concurrent attempt deterministic.
                ready.signal()
                start.wait()
                let value = MLXArray(input) * input
                eval(value)
                state.lock.withLock {
                    state.results.append(value.item(Int.self))
                }
                finished.signal()
            }
        }

        XCTAssertEqual(ready.wait(timeout: .now() + 5), .success)
        XCTAssertEqual(ready.wait(timeout: .now() + 5), .success)
        start.signal()
        start.signal()
        XCTAssertEqual(finished.wait(timeout: .now() + 30), .success)
        XCTAssertEqual(finished.wait(timeout: .now() + 30), .success)
        XCTAssertEqual(state.lock.withLock { state.results.sorted() }, [4, 9])
    }

    func testSetUnsetDefaultDevice() {
        // Issue #237 -- setting an unsetting the default device in a loop
        // exhausts many resources
        for _ in 1 ..< 10000 {
            let defaultDevice = MLX.Device.defaultDevice()
            MLX.Device.setDefault(device: .cpu)
            defer {
                MLX.Device.setDefault(device: defaultDevice)
            }

            let x = MLXArray(1)
            let _ = x * x
        }
        print("here")
    }

    func testWithDefaultDevice() {
        // Issue #237 -- scoped variant
        for _ in 1 ..< 10000 {
            Device.withDefaultDevice(.cpu) {
                Device.withDefaultDevice(.gpu) {
                    let x = MLXArray(1)
                    let _ = x * x
                }
            }
        }
        print("here")
    }

    func disabledTestCreateStream() {
        // see https://github.com/ml-explore/mlx/issues/2118
        for _ in 1 ..< 10000 {
            let _ = Stream(.cpu)
        }
        print("here")
    }

    func disabledTestCreateStreamScoped() {
        // see https://github.com/ml-explore/mlx/issues/2118
        for _ in 1 ..< 10000 {
            Stream.withNewDefaultStream(device: .cpu) {
                let x = MLXArray(1)
                let _ = x * x
            }
        }
    }

}
