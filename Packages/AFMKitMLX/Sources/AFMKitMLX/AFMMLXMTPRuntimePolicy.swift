import Foundation
import MLXLLM
import MLXLMCommon

enum AFMMLXMTPRuntimeModelKind: Equatable, Sendable {
    case qwenText
    case qwenVision
    case qwenNextText
    case qwenNextVision
    case glmEmbedded
}

/// Pure lifecycle decisions shared by model loading and request dispatch.
/// Keeping these rules independent of MLX objects makes failure/retry and
/// model-switch behavior deterministic and directly testable.
enum AFMMLXMTPRuntimePolicy {
    /// Resolve the Qwen Next verifier once at runtime construction, then keep
    /// it immutable for the lifetime of the generator. Singleton-equivalent
    /// verification is the conformant default. The faster batched reduction
    /// schedule is explicitly opt-in because it can change greedy decisions.
    static func qwenNextVerificationPolicy(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> MTPVerificationPolicy {
        switch environment["AFM_QWEN_MTP_VERIFICATION_POLICY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        {
        case "batched", "fast", "approximate":
            return .batched
        default:
            return .strictSingletonEquivalent
        }
    }

    /// Qwen Next's raw MTP sidecar is precision-sensitive: quantizing its
    /// proposal layer to the trunk's q4 default lowers draft acceptance and
    /// makes speculative decoding slower. Keep the head at a q8 floor while
    /// the target model retains the checkpoint's own quantization.
    static func qwenNextMTPHeadBits(configuredBits: Int) -> Int {
        max(Qwen4ExpModel.defaultMTPHeadBits, configuredBits)
    }

    static func loadingFactory(
        selected: AFMMLXModelFactoryKind,
        mtpEnabled: Bool,
        usesEmbeddedGLMHead: Bool
    ) -> AFMMLXModelFactoryKind {
        // MTP must never change the selected model container. A qualified
        // multimodal GLM remains a VLM and exposes MTP through its text trunk.
        _ = mtpEnabled
        _ = usesEmbeddedGLMHead
        return selected
    }

    static func compatibleModelKind(
        mtpEnabled: Bool,
        factory: AFMMLXModelFactoryKind,
        canonicalModelType: String
    ) -> AFMMLXMTPRuntimeModelKind? {
        guard mtpEnabled else { return nil }
        if canonicalModelType == "glm5_next" {
            return .glmEmbedded
        }
        if canonicalModelType == "qwen4_exp" {
            return factory == .vlm ? .qwenNextVision : .qwenNextText
        }
        guard canonicalModelType == "qwen3_5" || canonicalModelType == "qwen3_5_moe"
        else { return nil }
        return factory == .vlm ? .qwenVision : .qwenText
    }

    static func usesEmbeddedHead(
        canonicalModelType: String,
        embeddedAssetsPresent: Bool
    ) -> Bool {
        (canonicalModelType == "glm5_next" || canonicalModelType == "qwen4_exp")
            && embeddedAssetsPresent
    }

    static func canReuseLoadedModel(
        loadedModelID: String?,
        requestedModelID: String,
        mtpEnabled: Bool,
        bindingModelID: String?
    ) -> Bool {
        guard loadedModelID == requestedModelID else { return false }
        return !mtpEnabled || bindingModelID == requestedModelID
    }

    static func bindingIsUsable(
        for requestedModelID: String,
        mtpEnabled: Bool,
        bindingModelID: String?
    ) -> Bool {
        mtpEnabled && bindingModelID == requestedModelID
    }

    static func shouldUseSpeculativeBinding(
        requestedModelID: String,
        mtpEnabled: Bool,
        bindingModelID: String?,
        isMultimodal: Bool,
        isCancelled: Bool
    ) -> Bool {
        guard !isMultimodal, !isCancelled else { return false }
        return bindingIsUsable(
            for: requestedModelID,
            mtpEnabled: mtpEnabled,
            bindingModelID: bindingModelID
        )
    }

    static func shouldUseSpeculativeBinding(
        requestedModelID: String,
        mtpEnabled: Bool,
        bindingModelID: String?,
        mediaKinds: [AFMMLXRequestMediaKind],
        isCancelled: Bool
    ) -> Bool {
        shouldUseSpeculativeBinding(
            requestedModelID: requestedModelID,
            mtpEnabled: mtpEnabled,
            bindingModelID: bindingModelID,
            isMultimodal: !mediaKinds.isEmpty,
            isCancelled: isCancelled
        )
    }

    static func allowSynchronousSidecarDownload(mtpEnabled: Bool) -> Bool {
        mtpEnabled
    }

    static func shouldPrefetchInBackground(
        mtpEnabled: Bool,
        resolvedSidecar: String?,
        automaticRepositoryID: String?
    ) -> Bool {
        !mtpEnabled && resolvedSidecar == nil && automaticRepositoryID != nil
    }

    static func directSidecarHasRequiredMetadata(_ sidecarURL: URL) -> Bool {
        AFMMLXSpeculativeRuntimeResourceResolver.mtpQuantization(
            resourceDirectory: sidecarURL.deletingLastPathComponent()
        ) != nil
    }
}
