import AFMKitCore
import AFMKitServices
import Foundation
import XCTest

@testable import AFMKitMLX

final class MLXAdmissionTelemetryTests: XCTestCase {
    private final class Counters: @unchecked Sendable {
        let lock = NSLock()
        var running = 0
        var waiting = 0
        var reads = 0

        func set(_ running: Int, _ waiting: Int = 0) {
            lock.withLock { self.running = running; self.waiting = waiting }
        }
        func read() -> MLXAdmissionTelemetry.Counts {
            lock.withLock { reads += 1; return (running, waiting) }
        }
    }

    private final class BlockingStateObserver: AFMInferenceTelemetryObserving, @unchecked Sendable {
        let collector = InferenceTelemetryCollector()
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let publicationCount = Counters()

        func updateProviderState(_ state: AFMInferenceProviderState) {
            publicationCount.lock.withLock { publicationCount.running += 1 }
            if state.runningRequests == 4 {
                entered.signal()
                _ = release.wait(timeout: .now() + 5)
            }
            collector.updateProviderState(state)
        }
        func requestAccepted(at timestamp: Double) -> AFMInferenceRequestToken { .init() }
        func requestStarted(_ token: AFMInferenceRequestToken, at timestamp: Double) {}
        func outputToken(_ token: AFMInferenceRequestToken, at timestamp: Double) {}
        func prefixCacheObserved(queriedTokens: Int, hitTokens: Int) {}
        func speculativeRound(draftTokens: Int, acceptedTokens: Int) {}
        func preemptionObserved() {}
        func requestFinished(_ token: AFMInferenceRequestToken,
                             observation: AFMInferenceRequestFinishObservation) -> Bool { true }
        func requestFailed(_ token: AFMInferenceRequestToken, reason: AFMInferenceFailureReason,
                           at timestamp: Double) -> Bool { true }
    }

    func testObserverDeliveryRemainsSerializedAndDuplicateStateIsSuppressed() {
        let observer = BlockingStateObserver()
        let counts = Counters()
        let bridge = MLXAdmissionTelemetry(observer: observer,
            serial: { counts.read() }, admissionWaiters: { 0 })
        counts.set(4)
        let first = expectation(description: "blocked delivery")
        DispatchQueue.global().async { bridge.publish(); first.fulfill() }
        XCTAssertEqual(observer.entered.wait(timeout: .now() + 5), .success)
        counts.set(1)
        let secondStarted = DispatchSemaphore(value: 0)
        let second = expectation(description: "next delivery")
        DispatchQueue.global().async {
            secondStarted.signal()
            bridge.publish()
            second.fulfill()
        }
        XCTAssertEqual(secondStarted.wait(timeout: .now() + 5), .success)
        // Leave a bounded scheduling window for a broken unlock-before-delivery
        // implementation to overtake the first observer call.
        Thread.sleep(forTimeInterval: 0.05)
        XCTAssertEqual(observer.publicationCount.read().running, 1)
        observer.release.signal()
        wait(for: [first, second], timeout: 5)
        XCTAssertEqual(observer.collector.metricsSnapshot().runningRequests, 1)
        XCTAssertEqual(observer.collector.metricsSnapshot().peakRunningRequests, 4)
        for _ in 0..<100 { bridge.publish() }
        XCTAssertEqual(observer.publicationCount.read().running, 2)
    }

    func testBatchTransitionsPreserveResourcesAndDoNotDuplicateRequestCounters() {
        let collector = InferenceTelemetryCollector()
        let serial = Counters()
        let waiters = Counters()
        let batch = Counters()
        let bridge = MLXAdmissionTelemetry(observer: collector,
            serial: { serial.read() }, admissionWaiters: { waiters.read().waiting })
        bridge.updateResources(memoryCacheUsage: 0.25, prefixCacheFill: 0.5)
        bridge.installBatchReader { batch.read() }

        let token = collector.requestAccepted(at: 1)
        collector.requestStarted(token, at: 2)
        collector.outputToken(token, at: 3)
        collector.prefixCacheObserved(queriedTokens: 10, hitTokens: 6)

        // Reservation, submitted queue, drained cohort, 3 cancellations,
        // survivor completion, queued cancellation, abandoned reservation.
        for counts in [(4, 0), (0, 4), (4, 0), (1, 0), (0, 0),
                       (0, 2), (0, 1), (0, 0), (1, 0), (0, 0)] {
            batch.set(counts.0, counts.1)
            bridge.publish()
            bridge.publish() // No duplicate lifecycle accounting.
            let snapshot = collector.metricsSnapshot()
            XCTAssertEqual(snapshot.runningRequests, counts.0)
            XCTAssertEqual(snapshot.waitingRequests, counts.1)
            XCTAssertEqual(snapshot.memoryCacheUsage, 0.25)
            XCTAssertEqual(snapshot.prefixCacheFill, 0.5)
            XCTAssertEqual(snapshot.acceptedRequestsTotal, 1)
            XCTAssertEqual(snapshot.terminalRequestsTotal, 0)
            XCTAssertEqual(snapshot.generatedTokensTotal, 1)
            XCTAssertEqual(snapshot.prefixCacheQueriesTotal, 10)
            XCTAssertEqual(snapshot.prefixCacheHitsTotal, 6)
        }
        XCTAssertEqual(collector.metricsSnapshot().peakRunningRequests, 4)
        batch.set(2, 3)
        waiters.set(0, 5)
        bridge.publish()
        XCTAssertEqual(collector.metricsSnapshot().runningRequests, 2)
        XCTAssertEqual(collector.metricsSnapshot().waitingRequests, 8)
    }

    func testRetiredNotificationsUseReplacementAndDetachRestoresSerial() {
        let collector = InferenceTelemetryCollector()
        let serial = Counters()
        let old = Counters()
        let replacement = Counters()
        let bridge = MLXAdmissionTelemetry(observer: collector,
            serial: { serial.read() }, admissionWaiters: { 0 })
        serial.set(1)
        bridge.publish()
        XCTAssertEqual(collector.metricsSnapshot().runningRequests, 1)
        old.set(4)
        bridge.installBatchReader { old.read() }
        replacement.set(2, 1)
        bridge.installBatchReader { replacement.read() }
        let readsBefore = old.reads
        old.set(0)
        bridge.publish() // Late shutdown callback from retired scheduler.
        XCTAssertEqual(old.reads, readsBefore)
        XCTAssertEqual(collector.metricsSnapshot().runningRequests, 2)
        XCTAssertEqual(collector.metricsSnapshot().waitingRequests, 1)
        bridge.installBatchReader(nil)
        replacement.set(0)
        bridge.publish()
        XCTAssertEqual(collector.metricsSnapshot().runningRequests, 1)
        XCTAssertEqual(collector.metricsSnapshot().waitingRequests, 0)
        serial.set(0)
        bridge.publish()
        XCTAssertEqual(collector.metricsSnapshot().runningRequests, 0)
    }

    func testDelayedSnapshotCannotOverwriteNewerTransition() {
        let collector = InferenceTelemetryCollector()
        let counts = Counters()
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let blockOnce = Counters()
        let bridge = MLXAdmissionTelemetry(observer: collector,
            serial: {
                let snapshot = counts.read()
                if blockOnce.read().running == 1 {
                    blockOnce.set(0)
                    entered.signal()
                    _ = release.wait(timeout: .now() + 5)
                }
                return snapshot
            }, admissionWaiters: { 0 })
        counts.set(4)
        blockOnce.set(1)
        let firstDone = expectation(description: "older sample delivered")
        DispatchQueue.global().async { bridge.publish(); firstDone.fulfill() }
        XCTAssertEqual(entered.wait(timeout: .now() + 5), .success)
        counts.set(1)
        let secondDone = expectation(description: "newer sample wins")
        let secondStarted = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            secondStarted.signal()
            bridge.publish()
            secondDone.fulfill()
        }
        XCTAssertEqual(secondStarted.wait(timeout: .now() + 5), .success)
        release.signal()
        wait(for: [firstDone, secondDone], timeout: 5)
        XCTAssertEqual(collector.metricsSnapshot().runningRequests, 1)
        XCTAssertEqual(collector.metricsSnapshot().peakRunningRequests, 4)
    }

    func testServiceHeldCallbackNeedsOnlyIndependentCounterLocks() {
        let collector = InferenceTelemetryCollector()
        let serviceLock = NSLock()
        let counters = Counters()
        let bridge = MLXAdmissionTelemetry(observer: collector,
            serial: { (0, 0) }, admissionWaiters: { 0 })
        bridge.installBatchReader { counters.read() }
        let done = expectation(description: "service-held reservation publication")
        DispatchQueue.global().async {
            serviceLock.withLock {
                counters.set(4) // Counter lock released before notification.
                bridge.publish()
            }
            done.fulfill()
        }
        wait(for: [done], timeout: 5)
        XCTAssertEqual(collector.metricsSnapshot().runningRequests, 4)
    }

    func testReconnectCanReplayUnchangedStateWithoutResourceSampling() {
        let first = InferenceTelemetryCollector()
        let second = InferenceTelemetryCollector()
        let relay = AFMInferenceTelemetryRelay(target: first)
        let bridge = MLXAdmissionTelemetry(observer: relay,
            serial: { (1, 0) }, admissionWaiters: { 2 })
        bridge.updateResources(memoryCacheUsage: 0.2, prefixCacheFill: 0.4)
        relay.connect(to: second)
        bridge.publish(force: true)
        XCTAssertEqual(second.metricsSnapshot().runningRequests, 1)
        XCTAssertEqual(second.metricsSnapshot().waitingRequests, 2)
        XCTAssertEqual(second.metricsSnapshot().memoryCacheUsage, 0.2)
        XCTAssertEqual(second.metricsSnapshot().prefixCacheFill, 0.4)
    }
}
