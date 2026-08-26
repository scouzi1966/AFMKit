import XCTest
@testable import AFMKitMLX

final class AFMMLXStartupFactoryPolicyTests: XCTestCase {
    func testAutomaticQualifiedQwenVisionSelectionSupportsMTPRuntime() throws {
        let architecture = try qwenArchitecture()
        let qualification = qualification(
            architecture: architecture,
            missingAssets: []
        )
        let factory = AFMMLXModelFactoryPolicy.initialFactory(
            forceVLM: false,
            architecture: architecture,
            visionQualification: qualification
        )

        XCTAssertEqual(factory, .vlm)
        XCTAssertEqual(
            AFMMLXMTPRuntimePolicy.compatibleModelKind(
                mtpEnabled: true,
                factory: factory,
                canonicalModelType: architecture.canonicalModelType
            ),
            .qwenVision
        )
    }

    func testMTPRequestPolicyUsesCurrentTextOnlyBinding() {
        XCTAssertTrue(
            AFMMLXMTPRuntimePolicy.shouldUseSpeculativeBinding(
                requestedModelID: "qwen",
                mtpEnabled: true,
                bindingModelID: "qwen",
                isMultimodal: false,
                isCancelled: false
            )
        )
        XCTAssertFalse(
            AFMMLXMTPRuntimePolicy.shouldUseSpeculativeBinding(
                requestedModelID: "qwen",
                mtpEnabled: true,
                bindingModelID: "previous-model",
                isMultimodal: false,
                isCancelled: false
            )
        )
        XCTAssertFalse(
            AFMMLXMTPRuntimePolicy.shouldUseSpeculativeBinding(
                requestedModelID: "qwen",
                mtpEnabled: true,
                bindingModelID: "qwen",
                isMultimodal: true,
                isCancelled: false
            )
        )
        XCTAssertFalse(
            AFMMLXMTPRuntimePolicy.shouldUseSpeculativeBinding(
                requestedModelID: "qwen",
                mtpEnabled: true,
                bindingModelID: "qwen",
                isMultimodal: false,
                isCancelled: true
            )
        )
    }

    func testCompleteQwenConditionalGenerationSelectsVLM() throws {
        let architecture = try qwenArchitecture()
        let qualification = qualification(
            architecture: architecture,
            missingAssets: []
        )

        XCTAssertEqual(
            AFMMLXModelFactoryPolicy.initialFactory(
                forceVLM: false,
                architecture: architecture,
                visionQualification: qualification
            ),
            .vlm
        )
    }

    func testIncompleteQwenConditionalGenerationFallsBackToLLM() throws {
        let architecture = try qwenArchitecture()
        let qualification = qualification(
            architecture: architecture,
            missingAssets: [.processorConfiguration]
        )

        XCTAssertEqual(
            AFMMLXModelFactoryPolicy.initialFactory(
                forceVLM: false,
                architecture: architecture,
                visionQualification: qualification
            ),
            .llm
        )
    }

    func testQwenRuleDoesNotChangeGemmaStartup() {
        let architecture = AFMMLXModelArchitecturePreflight(
            modelID: "gemma4-vision",
            modelType: "gemma4",
            canonicalModelType: "gemma4",
            isVisionConfiguration: true,
            requiresVisionModelFactory: false
        )
        let qualification = AFMMLXVisionAssetQualification(
            snapshotIdentity: "gemma",
            modelType: "gemma4",
            canonicalModelType: "gemma4",
            isConditionalGeneration: true,
            declaresVision: true,
            processorClass: "Gemma4Processor",
            visionTensorCount: 1,
            missingAssets: []
        )

        XCTAssertEqual(
            AFMMLXModelFactoryPolicy.initialFactory(
                forceVLM: false,
                architecture: architecture,
                visionQualification: qualification
            ),
            .llm
        )
    }

    func testIncompleteQwenAssetsNeverAuthorizeVLMFactory() throws {
        let architecture = try qwenArchitecture()
        let qualification = qualification(
            architecture: architecture,
            missingAssets: [.visionWeights]
        )

        XCTAssertEqual(
            AFMMLXModelFactoryPolicy.initialFactory(
                forceVLM: true,
                architecture: architecture,
                visionQualification: qualification
            ),
            .llm
        )
    }

    func testForcedQwenVLMRejectsIncompleteAssetsBeforeLoading() throws {
        let architecture = try qwenArchitecture()
        let qualification = qualification(
            architecture: architecture,
            missingAssets: [.processorConfiguration, .visionWeights]
        )

        XCTAssertThrowsError(
            try MLXModelService.validateForcedVisionSelection(
                forceVLM: true,
                modelID: "fixture",
                qualification: qualification
            )
        ) { error in
            guard case MLXServiceError.visionAssetsUnavailable(let model, let missing) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(model, "fixture")
            XCTAssertEqual(missing, ["processorConfiguration", "visionWeights"])
        }
    }

    func testForcedQwenVLMRejectsMissingConditionalGenerationArchitecture() throws {
        let architecture = try qwenArchitecture()
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

        XCTAssertEqual(
            AFMMLXModelFactoryPolicy.initialFactory(
                forceVLM: true,
                architecture: architecture,
                visionQualification: qualification
            ),
            .llm
        )
        XCTAssertThrowsError(
            try MLXModelService.validateForcedVisionSelection(
                forceVLM: true,
                modelID: "fixture",
                qualification: qualification
            )
        ) { error in
            guard case MLXServiceError.visionAssetsUnavailable = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    private func qwenArchitecture() throws -> AFMMLXModelArchitecturePreflight {
        try AFMMLXModelArchitecture.preflightConfiguration(
            Qwen38PublishedConfigFixture.mxfp8,
            modelID: "fixture"
        )
    }

    private func qualification(
        architecture: AFMMLXModelArchitecturePreflight,
        missingAssets: Set<AFMMLXVisionAssetIssue>
    ) -> AFMMLXVisionAssetQualification {
        AFMMLXVisionAssetQualification(
            snapshotIdentity: "fixture",
            modelType: architecture.modelType,
            canonicalModelType: architecture.canonicalModelType,
            isConditionalGeneration: true,
            declaresVision: true,
            processorClass: "Qwen3VLProcessor",
            visionTensorCount: missingAssets.contains(.visionWeights) ? 0 : 1,
            missingAssets: missingAssets
        )
    }
}
