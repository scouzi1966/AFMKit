import AFMKitCore
import Foundation
import os

final class AFMDwarfStarGenerationAdmission: AFMGenerationAdmitting, @unchecked Sendable {
    private struct State {
        var reservations: Set<UUID> = []
        var waitingRequests = 0
    }

    private let maximumConcurrentRequests: Int
    private let telemetryObserver: any AFMInferenceTelemetryObserving
    private let providerStateCoordinator: AFMDwarfStarProviderStateCoordinator
    private let providerStateRegistration: UUID
    private let state = OSAllocatedUnfairLock(initialState: State())

    init(
        maximumConcurrentRequests: Int,
        telemetryObserver: any AFMInferenceTelemetryObserving,
        providerStateCoordinator: AFMDwarfStarProviderStateCoordinator = .shared
    ) {
        self.maximumConcurrentRequests = max(1, maximumConcurrentRequests)
        self.telemetryObserver = telemetryObserver
        self.providerStateCoordinator = providerStateCoordinator
        self.providerStateRegistration = providerStateCoordinator.register(telemetryObserver)
    }

    deinit {
        providerStateCoordinator.unregister(providerStateRegistration)
    }

    func admitGeneration(timeout: Duration?) async throws -> AFMGenerationLease {
        let acceptedAt = ProcessInfo.processInfo.systemUptime
        let telemetryToken = telemetryObserver.requestAccepted(at: acceptedAt)
        let reservationID = UUID()
        state.withLock { $0.waitingRequests += 1 }
        providerStateCoordinator.admissionStarted(reservationID)

        let timeoutSeconds = timeout.map(Self.timeInterval) ?? 30
        let deadline = ContinuousClock.now + .seconds(max(0, timeoutSeconds))
        var delay: UInt64 = 10_000_000

        while !reserve(reservationID) {
            if Task.isCancelled {
                failWaitingRequest(reservationID, telemetryToken, reason: .cancelled)
                throw AFMGenerationAdmissionError.cancelled
            }
            guard timeoutSeconds > 0 else {
                failWaitingRequest(reservationID, telemetryToken, reason: .inference)
                throw AFMGenerationAdmissionError.capacity
            }
            guard ContinuousClock.now < deadline else {
                failWaitingRequest(reservationID, telemetryToken, reason: .inference)
                throw AFMGenerationAdmissionError.timedOut
            }
            try? await Task.sleep(nanoseconds: delay)
            delay = min(delay * 2, 500_000_000)
        }

        telemetryObserver.requestStarted(
            telemetryToken,
            at: ProcessInfo.processInfo.systemUptime
        )
        // The lease keeps admission alive until capacity is returned. Otherwise
        // a model could deinitialize first and orphan this runtime-global
        // reservation in the provider-state coordinator.
        return AFMGenerationLease(telemetryToken: telemetryToken) { [self] in
            release(reservationID)
        } onAbandon: { [telemetryObserver] in
            _ = telemetryObserver.requestFailed(
                telemetryToken,
                reason: .internal,
                at: ProcessInfo.processInfo.systemUptime
            )
        }
    }

    private func reserve(_ reservationID: UUID) -> Bool {
        let reserved = state.withLock { state in
            guard state.reservations.count < maximumConcurrentRequests else {
                return false
            }
            state.reservations.insert(reservationID)
            state.waitingRequests = max(0, state.waitingRequests - 1)
            return true
        }
        if reserved {
            providerStateCoordinator.admissionReserved(reservationID)
        }
        return reserved
    }

    private func release(_ reservationID: UUID) {
        let removed = state.withLock { $0.reservations.remove(reservationID) != nil }
        if removed {
            providerStateCoordinator.admissionReleased(reservationID)
        }
    }

    private func failWaitingRequest(
        _ reservationID: UUID,
        _ telemetryToken: AFMInferenceRequestToken,
        reason: AFMInferenceFailureReason
    ) {
        state.withLock { $0.waitingRequests = max(0, $0.waitingRequests - 1) }
        providerStateCoordinator.admissionFailed(reservationID)
        _ = telemetryObserver.requestFailed(
            telemetryToken,
            reason: reason,
            at: ProcessInfo.processInfo.systemUptime
        )
    }

    private static func timeInterval(_ duration: Duration) -> TimeInterval {
        let components = duration.components
        return TimeInterval(components.seconds)
            + TimeInterval(components.attoseconds) / 1_000_000_000_000_000_000
    }
}
