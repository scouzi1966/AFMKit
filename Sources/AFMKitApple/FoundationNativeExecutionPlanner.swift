#if canImport(FoundationModels)
import Foundation

public enum AFMFoundationNativeExecutionError: LocalizedError, Equatable, Sendable {
    case missingEntitlement(String)
    case quotaLimit(String)

    public var errorDescription: String? {
        switch self {
        case .missingEntitlement(let entitlement):
            return "PCC entitlement missing from signed app: \(entitlement)"
        case .quotaLimit(let detail):
            return detail
        }
    }
}

public struct AFMFoundationNativeExecutionPlan: Equatable, Sendable {
    public let provider: AFMFoundationNativeProviderKind
    public let sessionPlan: AFMFoundationSessionPlan
    public let reasoningLevel: AFMFoundationReasoningLevel?

    public init(
        provider: AFMFoundationNativeProviderKind,
        sessionPlan: AFMFoundationSessionPlan,
        reasoningLevel: AFMFoundationReasoningLevel?
    ) {
        self.provider = provider
        self.sessionPlan = sessionPlan
        self.reasoningLevel = reasoningLevel
    }
}

public enum AFMFoundationNativeExecutionPlanner {
    public static func appleOnDevice(
        systemPrompt: String,
        toolsEnabled: Bool
    ) -> AFMFoundationNativeExecutionPlan {
        AFMFoundationNativeExecutionPlan(
            provider: .appleOnDevice,
            sessionPlan: AFMFoundationNativeSessionPolicy.appleOnDevice(
                systemPrompt: systemPrompt,
                toolsEnabled: toolsEnabled
            ),
            reasoningLevel: nil
        )
    }

    public static func privateCloudCompute(
        systemPrompt: String,
        toolsEnabled: Bool,
        reasoningLevel: AFMFoundationReasoningLevel,
        entitlement: String = AFMFoundationNativeProviderCapabilities.privateCloudComputeEntitlement,
        hasEntitlement: Bool,
        quotaLimitDetail: String?
    ) throws -> AFMFoundationNativeExecutionPlan {
        guard hasEntitlement else {
            throw AFMFoundationNativeExecutionError.missingEntitlement(entitlement)
        }
        if let quotaLimitDetail {
            throw AFMFoundationNativeExecutionError.quotaLimit(quotaLimitDetail)
        }
        return AFMFoundationNativeExecutionPlan(
            provider: .privateCloudCompute,
            sessionPlan: AFMFoundationNativeSessionPolicy.privateCloudCompute(
                systemPrompt: systemPrompt,
                toolsEnabled: toolsEnabled
            ),
            reasoningLevel: reasoningLevel
        )
    }
}
#endif
