import AFMKitCore
import AFMKitInference
import AFMOpenAICompat

/// Compile-time proof that an iOS host can use the provider-neutral contracts.
public enum AFMKitIOSCoreConsumer {
    public static func makeRequest() -> (Message, GenerationConfig, AFMProviderID) {
        let message = Message(role: "user", content: "Hello from iOS")
        let configuration = GenerationConfig(
            temperature: 0.2,
            maxTokens: 128
        )
        let providerID: AFMProviderID = "ios-provider"

        _ = AFMProviderRegistry().providerDescriptors()
        return (message, configuration, providerID)
    }
}
