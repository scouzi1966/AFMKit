import Foundation
@testable import AFMKitMLX
import MLXLLM
import MLXLMCommon
import XCTest

final class AFMMLXMTPRuntimePolicyTests: XCTestCase {
    func testQwenNextVerificationDefaultsToStrictAndRequiresExplicitBatchedMode() {
        XCTAssertEqual(
            AFMMLXMTPRuntimePolicy.qwenNextVerificationPolicy(environment: [:]),
            .strictSingletonEquivalent)
        XCTAssertEqual(
            AFMMLXMTPRuntimePolicy.qwenNextVerificationPolicy(environment: [
                "AFM_QWEN_MTP_VERIFICATION_POLICY": "batched"
            ]),
            .batched)
        XCTAssertEqual(
            AFMMLXMTPRuntimePolicy.qwenNextVerificationPolicy(environment: [
                "AFM_QWEN_MTP_VERIFICATION_POLICY": "fast"
            ]),
            .batched)
        XCTAssertEqual(
            AFMMLXMTPRuntimePolicy.qwenNextVerificationPolicy(environment: [
                "AFM_QWEN_MTP_VERIFICATION_POLICY": "strict-singleton-equivalent"
            ]),
            .strictSingletonEquivalent)
        XCTAssertEqual(
            AFMMLXMTPRuntimePolicy.qwenNextVerificationPolicy(environment: [
                "AFM_QWEN_MTP_VERIFICATION_POLICY": "unknown"
            ]),
            .strictSingletonEquivalent)
    }

    func testQwenNextMTPHeadUsesEightBitPrecisionFloor() {
        XCTAssertEqual(Qwen4ExpModel.defaultMTPHeadBits, 8)
        XCTAssertEqual(AFMMLXMTPRuntimePolicy.qwenNextMTPHeadBits(configuredBits: 4), 8)
        XCTAssertEqual(AFMMLXMTPRuntimePolicy.qwenNextMTPHeadBits(configuredBits: 8), 8)
        XCTAssertEqual(AFMMLXMTPRuntimePolicy.qwenNextMTPHeadBits(configuredBits: 16), 16)
    }

    func testQwenNextMTPPreservesTheSelectedTextOrVisionRuntime() {
        XCTAssertEqual(
            AFMMLXMTPRuntimePolicy.compatibleModelKind(
                mtpEnabled: true,
                factory: .llm,
                canonicalModelType: "qwen4_exp"
            ),
            .qwenNextText
        )
        XCTAssertEqual(
            AFMMLXMTPRuntimePolicy.compatibleModelKind(
                mtpEnabled: true,
                factory: .vlm,
                canonicalModelType: "qwen4_exp"
            ),
            .qwenNextVision
        )
    }

    func testQwenNextAndGLMRecognizeQualifiedEmbeddedHeads() {
        XCTAssertTrue(AFMMLXMTPRuntimePolicy.usesEmbeddedHead(
            canonicalModelType: "qwen4_exp", embeddedAssetsPresent: true))
        XCTAssertTrue(AFMMLXMTPRuntimePolicy.usesEmbeddedHead(
            canonicalModelType: "glm5_next", embeddedAssetsPresent: true))
        XCTAssertFalse(AFMMLXMTPRuntimePolicy.usesEmbeddedHead(
            canonicalModelType: "qwen4_exp", embeddedAssetsPresent: false))
        XCTAssertFalse(AFMMLXMTPRuntimePolicy.usesEmbeddedHead(
            canonicalModelType: "qwen3_5", embeddedAssetsPresent: true))
    }

    func testFailedMTPSetupCannotReuseBaseOnlyLoadedStateOnRetry() {
        XCTAssertFalse(
            AFMMLXMTPRuntimePolicy.canReuseLoadedModel(
                loadedModelID: "qwen",
                requestedModelID: "qwen",
                mtpEnabled: true,
                bindingModelID: nil
            )
        )
    }

    func testBaseOnlyLoadedStateRemainsReusableWhenMTPIsDisabled() {
        XCTAssertTrue(
            AFMMLXMTPRuntimePolicy.canReuseLoadedModel(
                loadedModelID: "qwen",
                requestedModelID: "qwen",
                mtpEnabled: false,
                bindingModelID: nil
            )
        )
    }

    func testBindingFromAnotherModelIsNeverUsable() {
        XCTAssertFalse(
            AFMMLXMTPRuntimePolicy.bindingIsUsable(
                for: "model-b",
                mtpEnabled: true,
                bindingModelID: "model-a"
            )
        )
        XCTAssertFalse(
            AFMMLXMTPRuntimePolicy.canReuseLoadedModel(
                loadedModelID: "model-b",
                requestedModelID: "model-b",
                mtpEnabled: true,
                bindingModelID: "model-a"
            )
        )
    }

    func testDisabledMTPUsesBackgroundRatherThanSynchronousDownload() {
        XCTAssertFalse(
            AFMMLXMTPRuntimePolicy.allowSynchronousSidecarDownload(mtpEnabled: false)
        )
        XCTAssertTrue(
            AFMMLXMTPRuntimePolicy.shouldPrefetchInBackground(
                mtpEnabled: false,
                resolvedSidecar: nil,
                automaticRepositoryID: "mlx-community/Qwen3.8-27B-MTP-4bit"
            )
        )
    }

    func testExplicitMTPAllowsSynchronousDownloadAndNoBackgroundDuplicate() {
        XCTAssertTrue(
            AFMMLXMTPRuntimePolicy.allowSynchronousSidecarDownload(mtpEnabled: true)
        )
        XCTAssertFalse(
            AFMMLXMTPRuntimePolicy.shouldPrefetchInBackground(
                mtpEnabled: true,
                resolvedSidecar: nil,
                automaticRepositoryID: "mlx-community/Qwen3.8-27B-MTP-4bit"
            )
        )
    }

    func testDirectSidecarRequiresQuantizationMetadata() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let sidecar = directory.appendingPathComponent("head.safetensors")
        try Data().write(to: sidecar)

        XCTAssertFalse(AFMMLXMTPRuntimePolicy.directSidecarHasRequiredMetadata(sidecar))

        let config: [String: Any] = [
            "quantization_config": ["group_size": 32, "bits": 4, "mode": "mxfp4"]
        ]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: directory.appendingPathComponent("config.json"))
        XCTAssertTrue(AFMMLXMTPRuntimePolicy.directSidecarHasRequiredMetadata(sidecar))
    }
}
