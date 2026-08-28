import AFMKitCore
@testable import AFMKitMLX
import AFMOpenAICompat
import XCTest

final class MLXMediaPreflightTests: XCTestCase {
    func testMediaCapabilityIsDiscoverableThroughServingExistential() async throws {
        let service = MLXModelService(resolver: MLXCacheResolver())
        let serving: any AFMMLXOpenAIChatServing = service
        let media = try XCTUnwrap(serving as? any AFMMLXMediaRequestServing)
        let messages = [Message(role: "user", content: "hello")]

        let resolved = try await media.preflightMediaRequest(
            model: "org/text",
            messages: messages
        )

        XCTAssertEqual(resolved.messages.count, 1)
        XCTAssertTrue(resolved.mediaKinds.isEmpty)
    }

    func testAFMMLXModelPublishesMediaCapability() {
        let model = AFMMLXModel(modelID: "org/text")
        let serving: any AFMMLXOpenAIChatServing = model

        XCTAssertNotNil(serving as? any AFMMLXMediaRequestServing)
    }

    func testMediaErrorsDoNotExposeAbsoluteModelPaths() {
        let missingAssets = MLXServiceError.visionAssetsUnavailable(
            model: "/Users/example/private/qwen-snapshot",
            missing: ["processorConfiguration", "visionWeights"]
        )
        let unsupported = MLXServiceError.unsupportedMediaInput(
            model: "/Users/example/private/qwen-snapshot",
            kind: "image"
        )

        XCTAssertEqual(
            missingAssets.localizedDescription,
            "qwen-snapshot: vision assets are unavailable (missing: processorConfiguration, visionWeights)"
        )
        XCTAssertEqual(
            unsupported.localizedDescription,
            "qwen-snapshot: image input is not supported by the loaded MLX model"
        )
        XCTAssertFalse(missingAssets.localizedDescription.contains("/Users/example"))
        XCTAssertFalse(unsupported.localizedDescription.contains("/Users/example"))
    }

    func testCompleteQwenVLMAllowsImagesAndAdvertisesRuntimeVision() {
        let architecture = qwenArchitecture()
        let qualification = qwenQualification()
        let descriptor = AFMMLXRuntimeVisionPolicy.runtimeDescriptor(
            declared: descriptor(capabilities: [.text, .vision]),
            architecture: architecture,
            qualification: qualification,
            factory: .vlm,
            mtpEnabled: false,
            mtpBindingModelID: nil
        )

        XCTAssertEqual(
            AFMMLXRuntimeVisionPolicy.admission(
                for: .image,
                architecture: architecture,
                qualification: qualification,
                factory: .vlm
            ),
            .allowed
        )
        XCTAssertTrue(descriptor.capabilities.contains(.vision))
    }

    func testIncompleteQwenFallsBackToTypedAssetFailureAndNoRuntimeVision() {
        let architecture = qwenArchitecture()
        let qualification = qwenQualification(
            missing: [.processorConfiguration, .visionWeights]
        )
        let descriptor = AFMMLXRuntimeVisionPolicy.runtimeDescriptor(
            declared: descriptor(capabilities: [.text, .vision]),
            architecture: architecture,
            qualification: qualification,
            factory: .llm,
            mtpEnabled: false,
            mtpBindingModelID: nil
        )

        XCTAssertEqual(
            AFMMLXRuntimeVisionPolicy.admission(
                for: .image,
                architecture: architecture,
                qualification: qualification,
                factory: .llm
            ),
            .visionAssetsUnavailable(
                missing: ["processorConfiguration", "visionWeights"]
            )
        )
        XCTAssertFalse(descriptor.capabilities.contains(.vision))
    }

    func testQwenMissingArchitectureMetadataCannotAdvertiseOrAdmitVision() {
        let architecture = qwenArchitecture()
        let qualification = AFMMLXVisionAssetQualification(
            snapshotIdentity: "fixture",
            modelType: architecture.modelType,
            canonicalModelType: architecture.canonicalModelType,
            isConditionalGeneration: false,
            declaresVision: true,
            processorClass: "Qwen3VLProcessor",
            visionTensorCount: 1,
            missingAssets: [.conditionalGenerationArchitecture]
        )
        let runtime = AFMMLXRuntimeVisionPolicy.runtimeDescriptor(
            declared: descriptor(capabilities: [.text, .vision]),
            architecture: architecture,
            qualification: qualification,
            factory: .vlm,
            mtpEnabled: false,
            mtpBindingModelID: nil
        )

        XCTAssertFalse(runtime.capabilities.contains(.vision))
        XCTAssertEqual(
            AFMMLXRuntimeVisionPolicy.admission(
                for: .image,
                architecture: architecture,
                qualification: qualification,
                factory: .vlm
            ),
            .visionAssetsUnavailable(missing: ["conditionalGenerationArchitecture"])
        )
    }

    func testCompleteQualificationDoesNotAuthorizeAnLLMContainer() {
        XCTAssertEqual(
            AFMMLXRuntimeVisionPolicy.admission(
                for: .image,
                architecture: qwenArchitecture(),
                qualification: qwenQualification(),
                factory: .llm
            ),
            .unsupported
        )
    }

    func testQualifiedQwenVLMDescriptorRequiresActualLoadedMTPBinding() {
        let declared = descriptor(capabilities: [.text, .vision, .speculativeDecoding])
        let inactive = AFMMLXRuntimeVisionPolicy.runtimeDescriptor(
            declared: declared,
            architecture: qwenArchitecture(),
            qualification: qwenQualification(),
            factory: .vlm,
            mtpEnabled: false,
            mtpBindingModelID: declared.modelID.rawValue
        )
        let wrongBinding = AFMMLXRuntimeVisionPolicy.runtimeDescriptor(
            declared: declared,
            architecture: qwenArchitecture(),
            qualification: qwenQualification(),
            factory: .vlm,
            mtpEnabled: true,
            mtpBindingModelID: "mlx-community/other-model"
        )
        let active = AFMMLXRuntimeVisionPolicy.runtimeDescriptor(
            declared: declared,
            architecture: qwenArchitecture(),
            qualification: qwenQualification(),
            factory: .vlm,
            mtpEnabled: true,
            mtpBindingModelID: declared.modelID.rawValue
        )

        XCTAssertTrue(active.capabilities.contains(.vision))
        XCTAssertTrue(active.capabilities.contains(.speculativeDecoding))
        XCTAssertFalse(inactive.capabilities.contains(.speculativeDecoding))
        XCTAssertFalse(wrongBinding.capabilities.contains(.speculativeDecoding))
    }

    func testNonVisionArchitectureRemainsUnsupported() {
        let architecture = AFMMLXModelArchitecturePreflight(
            modelID: "org/text",
            modelType: "qwen3_5",
            canonicalModelType: "qwen3_5",
            isVisionConfiguration: false,
            requiresVisionModelFactory: false
        )
        let qualification = AFMMLXVisionAssetQualification(
            snapshotIdentity: "text",
            modelType: "qwen3_5",
            canonicalModelType: "qwen3_5",
            isConditionalGeneration: false,
            declaresVision: false,
            processorClass: nil,
            visionTensorCount: 0,
            missingAssets: Set(AFMMLXVisionAssetIssue.allCases)
        )

        XCTAssertEqual(
            AFMMLXRuntimeVisionPolicy.admission(
                for: .image,
                architecture: architecture,
                qualification: qualification,
                factory: .llm
            ),
            .unsupported
        )
    }

    func testExistingNonQwenVLMUsesItsArchitectureCapability() {
        let architecture = AFMMLXModelArchitecturePreflight(
            modelID: "org/gemma-vision",
            modelType: "gemma3",
            canonicalModelType: "gemma3",
            isVisionConfiguration: true,
            requiresVisionModelFactory: true
        )
        let qualification = AFMMLXVisionAssetQualification(
            snapshotIdentity: "gemma",
            modelType: "gemma3",
            canonicalModelType: "gemma3",
            isConditionalGeneration: false,
            declaresVision: true,
            processorClass: "Gemma3Processor",
            visionTensorCount: 1,
            missingAssets: [.conditionalGenerationArchitecture]
        )

        XCTAssertEqual(
            AFMMLXRuntimeVisionPolicy.admission(
                for: .image,
                architecture: architecture,
                qualification: qualification,
                factory: .vlm
            ),
            .allowed
        )
    }

    func testGLMRuntimeVisionRequiresQualifiedAssets() {
        let architecture = glmArchitecture()
        let complete = glmQualification()
        let incomplete = glmQualification(missing: [.visionWeights])

        XCTAssertTrue(
            AFMMLXRuntimeVisionPolicy.supportsVision(
                architecture: architecture,
                qualification: complete,
                factory: .vlm))
        XCTAssertEqual(
            AFMMLXRuntimeVisionPolicy.admission(
                for: .video,
                architecture: architecture,
                qualification: complete,
                factory: .vlm),
            .allowed)
        XCTAssertFalse(
            AFMMLXRuntimeVisionPolicy.supportsVision(
                architecture: architecture,
                qualification: incomplete,
                factory: .vlm))
        XCTAssertEqual(
            AFMMLXRuntimeVisionPolicy.admission(
                for: .image,
                architecture: architecture,
                qualification: incomplete,
                factory: .llm),
            .visionAssetsUnavailable(missing: ["visionWeights"]))
    }

    private func qwenArchitecture() -> AFMMLXModelArchitecturePreflight {
        AFMMLXModelArchitecturePreflight(
            modelID: "mlx-community/Qwen3.8-27B-4bit",
            modelType: "qwen3_5",
            canonicalModelType: "qwen3_5",
            isVisionConfiguration: true,
            requiresVisionModelFactory: false
        )
    }

    private func glmArchitecture() -> AFMMLXModelArchitecturePreflight {
        AFMMLXModelArchitecturePreflight(
            modelID: "Vontra/GLM-5.3-Flash-MLX-4bit-MTP",
            modelType: "glm5_next",
            canonicalModelType: "glm5_next",
            isVisionConfiguration: true,
            requiresVisionModelFactory: false)
    }

    private func glmQualification(
        missing: Set<AFMMLXVisionAssetIssue> = []
    ) -> AFMMLXVisionAssetQualification {
        AFMMLXVisionAssetQualification(
            snapshotIdentity: "glm",
            modelType: "glm5_next",
            canonicalModelType: "glm5_next",
            isConditionalGeneration: true,
            declaresVision: true,
            processorClass: missing.contains(.processorConfiguration)
                ? nil : "Glm5NextProcessor",
            visionTensorCount: missing.contains(.visionWeights) ? 0 : 347,
            missingAssets: missing)
    }

    private func qwenQualification(
        missing: Set<AFMMLXVisionAssetIssue> = []
    ) -> AFMMLXVisionAssetQualification {
        AFMMLXVisionAssetQualification(
            snapshotIdentity: "qwen",
            modelType: "qwen3_5",
            canonicalModelType: "qwen3_5",
            isConditionalGeneration: true,
            declaresVision: true,
            processorClass: missing.contains(.processorConfiguration)
                ? nil
                : "Qwen3VLProcessor",
            visionTensorCount: missing.contains(.visionWeights) ? 0 : 2,
            missingAssets: missing
        )
    }

    private func descriptor(
        capabilities: AFMModelCapabilities
    ) -> AFMModelDescriptor {
        AFMModelDescriptor(
            providerID: "mlx",
            modelID: "mlx-community/Qwen3.8-27B-4bit",
            displayName: "Qwen3.8-27B-4bit",
            capabilities: capabilities,
            privacyBoundary: .device
        )
    }
}
