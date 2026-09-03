import AFMKitCore
import AFMKitServices
import XCTest

@testable import AFMKitMLX

final class MLXGenerationAdmissionTests: XCTestCase {
    private final class CapacityProbe: @unchecked Sendable {
        private let lock = NSLock()
        private var occupied = 0
        private var peak = 0
        private var releases = 0

        func reserve() {
            lock.withLock {
                occupied += 1
                peak = max(peak, occupied)
            }
        }

        func release() {
            lock.withLock {
                occupied = max(0, occupied - 1)
                releases += 1
            }
        }

        var snapshot: (occupied: Int, peak: Int, releases: Int) {
            lock.withLock { (occupied, peak, releases) }
        }
    }

    private final class CompletionProbe: @unchecked Sendable {
        private let lock = NSLock()
        private var completions = 0

        func complete() {
            lock.withLock { completions += 1 }
        }

        var count: Int { lock.withLock { completions } }
    }

    func testSchedulerIdentityMustMatchAndModelSwitchMustBeIdle() {
        XCTAssertTrue(MLXModelService.canUseScheduler(
            schedulerModelID: "model-a",
            currentModelID: "model-a",
            modelSwitchInProgress: false))
        XCTAssertFalse(MLXModelService.canUseScheduler(
            schedulerModelID: "model-a",
            currentModelID: "model-b",
            modelSwitchInProgress: false))
        XCTAssertFalse(MLXModelService.canUseScheduler(
            schedulerModelID: "model-a",
            currentModelID: "model-a",
            modelSwitchInProgress: true))
    }

    func testSuccessfulAndCancelledBatchSubmissionsReleaseCapacityExactlyOnce() {
        let capacity = CapacityProbe()

        for _ in 0..<2 {
            capacity.reserve()
            let lease = AFMGenerationLease(telemetryToken: AFMInferenceRequestToken()) {
                capacity.release()
            }
            AFMGenerationContext.$admissionLease.withValue(lease) {
                _ = MLXModelService.submittingAdmittedBatchRequest { "submitted" }
            }

            // Completion and cancellation share BatchScheduler's finishSlot
            // release path. A later caller release must be a no-op.
            capacity.release()
            lease.release()
        }

        let snapshot = capacity.snapshot
        XCTAssertEqual(snapshot.occupied, 0)
        XCTAssertEqual(snapshot.peak, 1)
        XCTAssertEqual(snapshot.releases, 2)
    }

    func testBatchSetupFailureLeavesCapacityReleaseWithCaller() {
        enum SetupFailure: Error { case expected }
        let capacity = CapacityProbe()
        capacity.reserve()
        let lease = AFMGenerationLease(telemetryToken: AFMInferenceRequestToken()) {
            capacity.release()
        }

        XCTAssertThrowsError(try AFMGenerationContext.$admissionLease.withValue(lease) {
            try MLXModelService.submittingAdmittedBatchRequest { () -> String in
                throw SetupFailure.expected
            }
        })
        lease.release()

        let snapshot = capacity.snapshot
        XCTAssertEqual(snapshot.occupied, 0)
        XCTAssertEqual(snapshot.releases, 1)
    }

    func testOperationOwningStreamCompletesExactlyOnceOnSuccess() async throws {
        let probe = CompletionProbe()
        let (source, continuation) = AsyncThrowingStream<Int, Error>.makeStream()
        let stream = MLXModelService.operationOwningStream(
            source, onFinish: { probe.complete() })

        continuation.yield(7)
        continuation.finish()
        var values: [Int] = []
        for try await value in stream { values.append(value) }

        XCTAssertEqual(values, [7])
        XCTAssertEqual(probe.count, 1)
    }

    func testOperationOwningStreamCompletesExactlyOnceOnError() async {
        enum ExpectedFailure: Error { case expected }
        let probe = CompletionProbe()
        let (source, continuation) = AsyncThrowingStream<Int, Error>.makeStream()
        let stream = MLXModelService.operationOwningStream(
            source, onFinish: { probe.complete() })

        continuation.finish(throwing: ExpectedFailure.expected)
        do {
            for try await _ in stream {}
            XCTFail("expected stream failure")
        } catch is ExpectedFailure {
            // Expected.
        } catch {
            XCTFail("unexpected error: \(error)")
        }
        XCTAssertEqual(probe.count, 1)
    }

    func testOperationOwningStreamCompletesExactlyOnceOnCancellation() async throws {
        let probe = CompletionProbe()
        let (source, continuation) = AsyncThrowingStream<Int, Error>.makeStream()
        let stream = MLXModelService.operationOwningStream(
            source, onFinish: { probe.complete() })
        let consumer = Task {
            for try await _ in stream {}
        }

        consumer.cancel()
        _ = try? await consumer.value
        continuation.finish()
        try await waitForCompletion(probe)
        XCTAssertEqual(probe.count, 1)
    }

    func testSerialAdmissionQueuesUntilCapacityAndRecordsQueueLatency() async throws {
        let collector = InferenceTelemetryCollector()
        let service = MLXModelService(
            resolver: MLXCacheResolver(),
            telemetryObserver: collector
        )
        let first = try await service.admitGeneration(timeout: .seconds(1))
        let waiter = Task {
            try await service.admitGeneration(timeout: .seconds(1))
        }

        try await waitForState(collector) {
            $0.runningRequests == 1 && $0.waitingRequests == 1
        }
        try await Task.sleep(for: .milliseconds(30))
        first.release()
        let second = try await waiter.value
        second.release()

        let snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.acceptedRequestsTotal, 2)
        XCTAssertEqual(snapshot.terminalRequestsTotal, 2)
        XCTAssertEqual(snapshot.runningRequests, 0)
        XCTAssertEqual(snapshot.waitingRequests, 0)
        XCTAssertEqual(snapshot.queueLatency.count, 2)
        XCTAssertGreaterThan(snapshot.queueLatency.sum, 0.02)
    }

    func testSerialAdmissionTimeoutIsProviderFailure() async throws {
        let collector = InferenceTelemetryCollector()
        let service = MLXModelService(
            resolver: MLXCacheResolver(),
            telemetryObserver: collector
        )
        let occupied = try await service.admitGeneration(timeout: .seconds(1))

        do {
            _ = try await service.admitGeneration(timeout: .milliseconds(30))
            XCTFail("serial admission should honor its capacity timeout")
        } catch let error as AFMGenerationAdmissionError {
            XCTAssertEqual(error, .timedOut)
        }

        let snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.acceptedRequestsTotal, 2)
        XCTAssertEqual(snapshot.terminalRequestsTotal, 1)
        XCTAssertEqual(snapshot.runningRequests, 1)
        XCTAssertEqual(snapshot.waitingRequests, 0)
        XCTAssertEqual(snapshot.failureCounts.first { $0.name == "inference" }?.count, 1)
        occupied.release()
    }

    func testSerialAdmissionCancellationMapsToAbort() async throws {
        let collector = InferenceTelemetryCollector()
        let service = MLXModelService(
            resolver: MLXCacheResolver(),
            telemetryObserver: collector
        )
        let occupied = try await service.admitGeneration(timeout: .seconds(1))
        let waiter = Task {
            try await service.admitGeneration(timeout: .seconds(1))
        }

        try await waitForState(collector) { $0.waitingRequests == 1 }
        waiter.cancel()
        do {
            _ = try await waiter.value
            XCTFail("cancelled serial admission must throw")
        } catch let error as AFMGenerationAdmissionError {
            XCTAssertEqual(error, .cancelled)
        }

        let snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.terminalCounts.first { $0.name == "abort" }?.count, 1)
        XCTAssertEqual(snapshot.failureCounts.first { $0.name == "cancelled" }?.count, 1)
        occupied.release()
    }

    func testSerialAdmissionDrainSignalResumesAfterRelease() async throws {
        let service = MLXModelService(
            resolver: MLXCacheResolver(),
            telemetryObserver: InferenceTelemetryCollector()
        )
        XCTAssertTrue(service.tryReserveSerialSlot())
        let waiter = Task {
            try await service.waitForSerialAdmissionsToDrain()
        }

        try await Task.sleep(for: .milliseconds(20))
        service.releaseSerialAdmissionState()
        try await waiter.value
    }

    func testCancelledSerialAdmissionDrainSignalDoesNotRequireRelease() async throws {
        let service = MLXModelService(
            resolver: MLXCacheResolver(),
            telemetryObserver: InferenceTelemetryCollector()
        )
        XCTAssertTrue(service.tryReserveSerialSlot())
        let waiter = Task {
            try await service.waitForSerialAdmissionsToDrain()
        }

        try await Task.sleep(for: .milliseconds(20))
        waiter.cancel()
        do {
            try await waiter.value
            XCTFail("cancelled drain waiter must throw")
        } catch is CancellationError {
            // Expected: cancellation removes and resumes the stored continuation.
        }
        service.releaseSerialAdmissionState()
    }

    private func waitForState(
        _ collector: InferenceTelemetryCollector,
        predicate: (AFMInferenceMetricsSnapshot) -> Bool
    ) async throws {
        let deadline = ContinuousClock.now + .seconds(1)
        while ContinuousClock.now < deadline {
            if predicate(collector.metricsSnapshot()) { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("telemetry state did not reach expected value")
    }

    private func waitForCompletion(_ probe: CompletionProbe) async throws {
        let deadline = ContinuousClock.now + .seconds(1)
        while ContinuousClock.now < deadline {
            if probe.count == 1 { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("stream completion callback was not invoked")
    }
}
