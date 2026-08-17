#if canImport(FoundationModels)
import Foundation
import FoundationModels

@available(macOS 27.0, *)
public enum AFMFoundationNativeAvailability: Equatable, Sendable {
    case available
    case unavailable(reason: String, detail: String)
}

@available(macOS 27.0, *)
public struct AFMFoundationAppleOnDeviceSnapshot: Equatable, Sendable {
    public let availability: AFMFoundationNativeAvailability
    public let localeIdentifier: String
    public let localeSupported: Bool
    public let contextSize: Int

    public init(
        availability: AFMFoundationNativeAvailability,
        localeIdentifier: String,
        localeSupported: Bool,
        contextSize: Int
    ) {
        self.availability = availability
        self.localeIdentifier = localeIdentifier
        self.localeSupported = localeSupported
        self.contextSize = contextSize
    }
}

@available(macOS 27.0, *)
public struct AFMFoundationPrivateCloudComputeSnapshot: Equatable, Sendable {
    public let hasEntitlement: Bool
    public let entitlement: String
    public let availability: AFMFoundationNativeAvailability
    public let localeIdentifier: String
    public let localeSupported: Bool
    public let quotaStatus: String
    public let quotaIsLimitReached: Bool
    public let quotaLimitDetail: String?

    public init(
        hasEntitlement: Bool,
        entitlement: String,
        availability: AFMFoundationNativeAvailability,
        localeIdentifier: String,
        localeSupported: Bool,
        quotaStatus: String,
        quotaIsLimitReached: Bool,
        quotaLimitDetail: String?
    ) {
        self.hasEntitlement = hasEntitlement
        self.entitlement = entitlement
        self.availability = availability
        self.localeIdentifier = localeIdentifier
        self.localeSupported = localeSupported
        self.quotaStatus = quotaStatus
        self.quotaIsLimitReached = quotaIsLimitReached
        self.quotaLimitDetail = quotaLimitDetail
    }
}

/// Reads live Apple Intelligence and Private Cloud Compute availability without
/// coupling the result to an application's provider-status or UI types.
@available(macOS 27.0, *)
public final class AFMFoundationNativeProviderProbe {
    private let privateCloudComputeModel: PrivateCloudComputeLanguageModel

    public init(
        privateCloudComputeModel: PrivateCloudComputeLanguageModel = PrivateCloudComputeLanguageModel()
    ) {
        self.privateCloudComputeModel = privateCloudComputeModel
    }

    public func systemContextWindow() -> Int {
        SystemLanguageModel.default.contextSize
    }

    public func appleOnDeviceSnapshot(
        locale: Locale = .current
    ) -> AFMFoundationAppleOnDeviceSnapshot {
        let model = SystemLanguageModel.default
        let availability: AFMFoundationNativeAvailability
        switch model.availability {
        case .available:
            availability = .available
        case .unavailable(let reason):
            availability = .unavailable(
                reason: "\(reason)",
                detail: AFMFoundationNativeAvailabilityReasonDescriptions.systemLanguageModel(reason)
            )
        }
        return AFMFoundationAppleOnDeviceSnapshot(
            availability: availability,
            localeIdentifier: locale.identifier,
            localeSupported: model.supportsLocale(locale),
            contextSize: model.contextSize
        )
    }

    public func privateCloudComputeQuotaStatus() -> String {
        "\(privateCloudComputeModel.quotaUsage.status)"
    }

    public func privateCloudComputeSnapshot(
        hasEntitlement: Bool,
        entitlement: String = AFMFoundationNativeProviderCapabilities.privateCloudComputeEntitlement,
        locale: Locale = .current
    ) -> AFMFoundationPrivateCloudComputeSnapshot {
        guard hasEntitlement else {
            return AFMFoundationPrivateCloudComputeSnapshot(
                hasEntitlement: false,
                entitlement: entitlement,
                availability: .unavailable(
                    reason: "missingEntitlement",
                    detail: "PCC entitlement missing from signed app: \(entitlement)"
                ),
                localeIdentifier: locale.identifier,
                localeSupported: false,
                quotaStatus: "unknown",
                quotaIsLimitReached: false,
                quotaLimitDetail: nil
            )
        }

        let model = privateCloudComputeModel
        let quota = model.quotaUsage
        let quotaLimitDetail = quota.isLimitReached
            ? AFMFoundationNativeAvailabilityReasonDescriptions.privateCloudComputeQuotaLimit(quota)
            : nil
        let availability: AFMFoundationNativeAvailability
        switch model.availability {
        case .available:
            availability = .available
        case .unavailable(let reason):
            availability = .unavailable(
                reason: "\(reason)",
                detail: AFMFoundationNativeAvailabilityReasonDescriptions.privateCloudCompute(reason)
            )
        }
        return AFMFoundationPrivateCloudComputeSnapshot(
            hasEntitlement: true,
            entitlement: entitlement,
            availability: availability,
            localeIdentifier: locale.identifier,
            localeSupported: model.supportsLocale(locale),
            quotaStatus: "\(quota.status)",
            quotaIsLimitReached: quota.isLimitReached,
            quotaLimitDetail: quotaLimitDetail
        )
    }
}
#endif
