import AFMKitCore
import Foundation
import XCTest

final class InferenceTelemetryRelayTests: XCTestCase {
    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0

        func increment() { lock.withLock { value += 1 } }
        func read() -> Int { lock.withLock { value } }
    }

    private final class RecordingObserver: AFMInferenceTelemetryObserving, @unchecked Sendable {
        private let lock = NSLock()
        private var storage: [String] = []

        var events: [String] { lock.withLock { storage } }

        func requestAccepted(at timestamp: Double) -> AFMInferenceRequestToken {
            let token = AFMInferenceRequestToken()
            lock.withLock { storage.append("accepted:\(token.rawValue)") }
            return token
        }

        func requestStarted(_ token: AFMInferenceRequestToken, at timestamp: Double) {
            lock.withLock { storage.append("started:\(token.rawValue)") }
        }

        func outputToken(_ token: AFMInferenceRequestToken, at timestamp: Double) {
            lock.withLock { storage.append("output:\(token.rawValue)") }
        }

        func prefixCacheObserved(queriedTokens: Int, hitTokens: Int) {}
        func speculativeRound(draftTokens: Int, acceptedTokens: Int) {}
        func preemptionObserved() {}
        func updateProviderState(_ state: AFMInferenceProviderState) {}

        func requestFinished(
            _ token: AFMInferenceRequestToken,
            observation: AFMInferenceRequestFinishObservation
        ) -> Bool {
            lock.withLock { storage.append("finished:\(token.rawValue)") }
            return true
        }

        func requestFailed(
            _ token: AFMInferenceRequestToken,
            reason: AFMInferenceFailureReason,
            at timestamp: Double
        ) -> Bool {
            lock.withLock { storage.append("failed:\(token.rawValue)") }
            return true
        }
    }

    func testReconnectPinsAcceptedRequestToOriginalObserverUntilTerminal() {
        let first = RecordingObserver()
        let second = RecordingObserver()
        let relay = AFMInferenceTelemetryRelay(target: first)
        let token = relay.requestAccepted(at: 1)

        relay.connect(to: second)
        relay.requestStarted(token, at: 2)
        relay.outputToken(token, at: 3)
        XCTAssertTrue(relay.requestFinished(
            token,
            observation: AFMInferenceRequestFinishObservation(
                reason: .stop,
                completedAt: 4,
                fullPromptTokens: 1,
                computedPromptTokens: 1,
                generatedTokens: 1
            )
        ))

        let tokenText = token.rawValue.uuidString
        XCTAssertEqual(first.events.count, 4)
        XCTAssertTrue(first.events.allSatisfy { $0.contains(tokenText) })
        XCTAssertTrue(second.events.isEmpty)

        _ = relay.requestAccepted(at: 5)
        XCTAssertEqual(second.events.count, 1)
    }

    func testTransferredLeaseDoesNotReleaseProviderOwnedCapacityTwice() {
        let releaseCount = Counter()
        let lease = AFMGenerationLease(telemetryToken: AFMInferenceRequestToken()) {
            releaseCount.increment()
        }

        lease.transferReleaseToProvider()
        lease.release()
        lease.release()

        XCTAssertEqual(releaseCount.read(), 0)
    }
}
