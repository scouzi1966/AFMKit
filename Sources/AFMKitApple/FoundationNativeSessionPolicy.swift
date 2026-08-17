#if canImport(FoundationModels)
import Foundation

public enum AFMFoundationSessionStyle: Equatable, Sendable {
    case dynamicProfile
    case simple
}

public struct AFMFoundationSessionPlan: Equatable, Sendable {
    public let provider: AFMFoundationNativeProviderKind
    public let signature: String
    public let style: AFMFoundationSessionStyle

    public init(
        provider: AFMFoundationNativeProviderKind,
        signature: String,
        style: AFMFoundationSessionStyle
    ) {
        self.provider = provider
        self.signature = signature
        self.style = style
    }
}

public enum AFMFoundationNativeSessionPolicy {
    public static func appleOnDevice(
        systemPrompt: String,
        toolsEnabled: Bool
    ) -> AFMFoundationSessionPlan {
        AFMFoundationSessionPlan(
            provider: .appleOnDevice,
            signature: "apple.system.default|tools:\(toolsEnabled)|\(systemPrompt)",
            style: .dynamicProfile
        )
    }

    public static func privateCloudCompute(
        systemPrompt: String,
        toolsEnabled: Bool
    ) -> AFMFoundationSessionPlan {
        AFMFoundationSessionPlan(
            provider: .privateCloudCompute,
            signature: "apple.private-cloud-compute|tools:\(toolsEnabled)|\(systemPrompt)",
            style: .dynamicProfile
        )
    }
}
#endif
