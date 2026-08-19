#if canImport(FoundationModels)
import AFMKitCore
@testable import AFMKitApple
import XCTest

@available(macOS 27.0, *)
final class FoundationProviderTests: XCTestCase {
    func testFactoryPublishesStableProviderContract() {
        let factory = AFMFoundationProviderFactory()

        XCTAssertEqual(factory.descriptor.id, "apple.foundation-models")
        XCTAssertEqual(factory.descriptor.privacyBoundary, .configurable)
        XCTAssertEqual(
            factory.descriptor.configurationKeys,
            [
                AFMFoundationProviderConfigurationKeys.systemPrompt,
                AFMFoundationProviderConfigurationKeys.reasoningLevel,
            ]
        )
        XCTAssertEqual(
            AFMFoundationManagedCapabilities.privateCloudComputeEntitlement,
            "com.apple.developer.private-cloud-compute"
        )
    }

    func testManagedCapabilityValuesRequireBooleanTrue() {
        XCTAssertTrue(AFMFoundationManagedCapabilities.entitlementValueIsEnabled(true))
        XCTAssertFalse(AFMFoundationManagedCapabilities.entitlementValueIsEnabled(false))
        XCTAssertFalse(AFMFoundationManagedCapabilities.entitlementValueIsEnabled("true"))
        XCTAssertFalse(AFMFoundationManagedCapabilities.entitlementValueIsEnabled(nil))
    }

    func testFactoryListsOnDeviceAndPCCWithoutToolCapabilityWhenNoToolsExist() async throws {
        let descriptors = try await AFMFoundationProviderFactory().modelDescriptors()
        let byID = Dictionary(uniqueKeysWithValues: descriptors.map { ($0.modelID, $0) })

        let onDevice = try XCTUnwrap(byID[AFMFoundationProviderFactory.onDeviceModelID])
        XCTAssertEqual(onDevice.privacyBoundary, .device)
        XCTAssertFalse(onDevice.capabilities.contains(.toolCalling))
        XCTAssertTrue(onDevice.capabilities.contains(.streaming))

        let pcc = try XCTUnwrap(byID[AFMFoundationProviderFactory.privateCloudComputeModelID])
        XCTAssertEqual(pcc.privacyBoundary, .privateCloud)
        XCTAssertTrue(pcc.capabilities.contains(.reasoning))
        XCTAssertFalse(pcc.capabilities.contains(.toolCalling))
    }

    func testFactoryCreatesKnownModelsAndRejectsUnknownModel() throws {
        let factory = AFMFoundationProviderFactory()
        let onDevice = try factory.makeModel(
            id: AFMFoundationProviderFactory.onDeviceModelID,
            configuration: .init()
        )
        XCTAssertEqual(onDevice.descriptor.modelID, AFMFoundationProviderFactory.onDeviceModelID)

        XCTAssertThrowsError(
            try factory.makeModel(id: "apple.unknown", configuration: .init())
        ) { error in
            XCTAssertEqual(
                error as? AFMError,
                .modelNotFound(provider: AFMFoundationProviderFactory.providerID, model: "apple.unknown")
            )
        }
    }

    func testPCCAvailabilityPrioritizesMissingEntitlementOverDerivedLocaleState() async throws {
        let model = try AFMFoundationModel(
            modelID: AFMFoundationProviderFactory.privateCloudComputeModelID,
            hasPrivateCloudComputeEntitlement: { false }
        )

        let availability = await model.availability()

        guard case .unavailable(let reason) = availability else {
            return XCTFail("Expected PCC to be unavailable without its managed entitlement.")
        }
        XCTAssertEqual(
            reason,
            "PCC entitlement missing from signed app: com.apple.developer.private-cloud-compute"
        )
    }

    func testAdapterBuildsInstructionsAndCompleteConversation() throws {
        let request = AFMRequest(
            messages: [
                AFMMessage(role: .system, text: "Answer precisely."),
                AFMMessage(role: .user, text: "First question"),
                AFMMessage(role: .assistant, text: "First answer"),
                AFMMessage(role: .user, text: "Follow-up"),
            ],
            options: AFMGenerationOptions(
                temperature: 0,
                maximumResponseTokens: 128,
                topP: 0.4
            )
        )

        let plan = try AFMFoundationProviderRequestAdapter.plan(
            request: request,
            provider: .appleOnDevice,
            configuredSystemPrompt: "Be concise.",
            configuredReasoningLevel: .automatic,
            availableToolNames: []
        )

        XCTAssertEqual(plan.instructions, "Be concise.\n\nAnswer precisely.")
        XCTAssertEqual(
            plan.conversation,
            "User: First question\n\nAssistant: First answer\n\nUser: Follow-up\n\nAssistant:"
        )
        XCTAssertEqual(plan.generationOptions.sampling, .greedy)
        XCTAssertEqual(plan.generationOptions.maximumResponseTokens, 128)
        XCTAssertNil(plan.reasoningLevel)
    }

    func testAdapterMapsPCCReasoningOverride() throws {
        let plan = try AFMFoundationProviderRequestAdapter.plan(
            request: AFMRequest(
                messages: [AFMMessage(role: .user, text: "Analyze this")],
                metadata: [AFMFoundationProviderConfigurationKeys.reasoningLevel: .string("high")]
            ),
            provider: .privateCloudCompute,
            configuredSystemPrompt: "",
            configuredReasoningLevel: .light,
            availableToolNames: []
        )

        XCTAssertEqual(plan.reasoningLevel, .deep)
    }

    func testAdapterRejectsOnDeviceReasoning() {
        XCTAssertThrowsError(
            try AFMFoundationProviderRequestAdapter.plan(
                request: AFMRequest(
                    messages: [AFMMessage(role: .user, text: "Analyze this")],
                    metadata: [
                        AFMFoundationProviderConfigurationKeys.reasoningLevel: .string("moderate")
                    ]
                ),
                provider: .appleOnDevice,
                configuredSystemPrompt: "",
                configuredReasoningLevel: .automatic,
                availableToolNames: []
            )
        ) { error in
            guard case .unsupportedCapability(let detail) = error as? AFMError else {
                return XCTFail("Expected unsupportedCapability, got \(error)")
            }
            XCTAssertTrue(detail.contains("Private Cloud Compute"))
        }
    }

    func testAdapterRejectsUnsupportedOptionsAndUnboundTools() {
        XCTAssertThrowsError(
            try AFMFoundationProviderRequestAdapter.plan(
                request: AFMRequest(
                    messages: [AFMMessage(role: .user, text: "Hello")],
                    options: AFMGenerationOptions(topK: 20)
                ),
                provider: .appleOnDevice,
                configuredSystemPrompt: "",
                configuredReasoningLevel: .automatic,
                availableToolNames: []
            )
        ) { error in
            guard case .unsupportedCapability(let detail) = error as? AFMError else {
                return XCTFail("Expected unsupportedCapability, got \(error)")
            }
            XCTAssertTrue(detail.contains("topK"))
        }

        XCTAssertThrowsError(
            try AFMFoundationProviderRequestAdapter.plan(
                request: AFMRequest(
                    messages: [AFMMessage(role: .user, text: "Check weather")],
                    tools: [
                        AFMToolDefinition(
                            name: "weather",
                            inputSchema: .object(["type": .string("object")])
                        )
                    ]
                ),
                provider: .appleOnDevice,
                configuredSystemPrompt: "",
                configuredReasoningLevel: .automatic,
                availableToolNames: []
            )
        ) { error in
            guard case .invalidRequest(let detail) = error as? AFMError else {
                return XCTFail("Expected invalidRequest, got \(error)")
            }
            XCTAssertTrue(detail.contains("weather"))
        }
    }

    func testAdapterRejectsReasoningEnabledOption() {
        XCTAssertThrowsError(
            try AFMFoundationProviderRequestAdapter.plan(
                request: AFMRequest(
                    messages: [AFMMessage(role: .user, text: "Analyze this")],
                    options: AFMGenerationOptions(reasoningEnabled: false)
                ),
                provider: .appleOnDevice,
                configuredSystemPrompt: "",
                configuredReasoningLevel: .automatic,
                availableToolNames: []
            )
        ) { error in
            guard case .unsupportedCapability(let detail) = error as? AFMError else {
                return XCTFail("Expected unsupportedCapability, got \(error)")
            }
            XCTAssertTrue(detail.contains("reasoningEnabled"))
        }
    }
}
#endif
