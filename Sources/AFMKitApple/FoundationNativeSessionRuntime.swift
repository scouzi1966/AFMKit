#if canImport(FoundationModels)
import Foundation
import FoundationModels

@available(macOS 27.0, *)
public struct AFMFoundationNativeSession {
    public let provider: AFMFoundationNativeProviderKind
    public let session: LanguageModelSession
    public let onDeviceModel: SystemLanguageModel?
    public let reasoningLevel: AFMFoundationReasoningLevel?

    public init(
        provider: AFMFoundationNativeProviderKind,
        session: LanguageModelSession,
        onDeviceModel: SystemLanguageModel?,
        reasoningLevel: AFMFoundationReasoningLevel?
    ) {
        self.provider = provider
        self.session = session
        self.onDeviceModel = onDeviceModel
        self.reasoningLevel = reasoningLevel
    }
}

/// Creates reusable Foundation Models sessions for Apple's on-device and PCC
/// providers. Entitlement detection remains the host application's
/// responsibility because it depends on that application's signed code.
@MainActor
@available(macOS 27.0, *)
public final class AFMFoundationNativeSessionRuntime {
    private let coordinator: AFMFoundationModelSessionCoordinator<AFMFoundationNativeProviderKind>
    private let privateCloudComputeModel: PrivateCloudComputeLanguageModel

    public init(
        coordinator: AFMFoundationModelSessionCoordinator<AFMFoundationNativeProviderKind> = .init(),
        privateCloudComputeModel: PrivateCloudComputeLanguageModel = PrivateCloudComputeLanguageModel()
    ) {
        self.coordinator = coordinator
        self.privateCloudComputeModel = privateCloudComputeModel
    }

    public func appleOnDevice(
        systemPrompt: String,
        toolsEnabled: Bool,
        tools: [any FoundationModels.Tool] = []
    ) -> AFMFoundationNativeSession {
        let plan = AFMFoundationNativeExecutionPlanner.appleOnDevice(
            systemPrompt: systemPrompt,
            toolsEnabled: toolsEnabled
        )
        let model = SystemLanguageModel.default
        let session = coordinator.dynamicProfileSession(
            for: plan.provider,
            signature: plan.sessionPlan.signature,
            model: model,
            tools: tools,
            instructions: systemPrompt
        )
        return AFMFoundationNativeSession(
            provider: plan.provider,
            session: session,
            onDeviceModel: model,
            reasoningLevel: plan.reasoningLevel
        )
    }

    public func privateCloudCompute(
        systemPrompt: String,
        toolsEnabled: Bool,
        reasoningLevel: AFMFoundationReasoningLevel,
        entitlement: String = AFMFoundationNativeProviderCapabilities.privateCloudComputeEntitlement,
        hasEntitlement: Bool,
        tools: [any FoundationModels.Tool] = []
    ) throws -> AFMFoundationNativeSession {
        guard hasEntitlement else {
            _ = try AFMFoundationNativeExecutionPlanner.privateCloudCompute(
                systemPrompt: systemPrompt,
                toolsEnabled: toolsEnabled,
                reasoningLevel: reasoningLevel,
                entitlement: entitlement,
                hasEntitlement: false,
                quotaLimitDetail: nil
            )
            preconditionFailure("Missing entitlement validation must throw")
        }

        let quota = privateCloudComputeModel.quotaUsage
        let plan = try AFMFoundationNativeExecutionPlanner.privateCloudCompute(
            systemPrompt: systemPrompt,
            toolsEnabled: toolsEnabled,
            reasoningLevel: reasoningLevel,
            entitlement: entitlement,
            hasEntitlement: true,
            quotaLimitDetail: quota.isLimitReached
                ? AFMFoundationNativeAvailabilityReasonDescriptions.privateCloudComputeQuotaLimit(quota)
                : nil
        )
        let session = coordinator.dynamicProfileSession(
            for: plan.provider,
            signature: plan.sessionPlan.signature,
            model: privateCloudComputeModel,
            tools: tools,
            instructions: systemPrompt
        )
        return AFMFoundationNativeSession(
            provider: plan.provider,
            session: session,
            onDeviceModel: nil,
            reasoningLevel: plan.reasoningLevel
        )
    }

    public nonisolated static func allowsToolCalling(toolCount: Int) -> Bool {
        toolCount > 0
    }
}
#endif
