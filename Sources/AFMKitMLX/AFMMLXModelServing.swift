import Foundation
import AFMOpenAICompat

// MARK: - Stable serving facade

package extension AFMMLXModel {
    private var concreteService: MLXModelService {
        guard let service else {
            preconditionFailure("AFMMLXModelServing requires a concrete MLX service.")
        }
        return service
    }

    var maxConcurrent: Int { concreteService.maxConcurrent }

    var servingConfiguration: AFMMLXServingConfiguration {
        concreteService.servingConfiguration
    }

    var defaultGuidedJsonSchema: ResponseFormat? {
        concreteService.defaultGuidedJsonSchema
    }

    func normalizeModel(_ raw: String) -> String {
        concreteService.normalizeModel(raw)
    }

    func resolvedToolCallParser(logBypass: Bool) -> String? {
        concreteService.resolvedToolCallParser(logBypass: logBypass)
    }

    func tryReserveSlot() -> Bool {
        concreteService.tryReserveSlot()
    }

    func waitForSlot(timeout: TimeInterval) async -> Bool {
        await concreteService.waitForSlot(timeout: timeout)
    }

    func releaseSlot() {
        concreteService.releaseSlot()
    }

    func ensureBatchMode(concurrency: Int) async throws {
        try await concreteService.ensureBatchMode(concurrency: concurrency)
    }

    func releaseBatchReference() {
        concreteService.releaseBatchReference()
    }

    func cancelBatchSlots(ids: Set<UUID>) async {
        await concreteService.cancelBatchSlots(ids: ids)
    }

    func startAPIProfile() {
        concreteService.startAPIProfile()
    }

    func stopAPIProfile(
        promptTokens: Int,
        completionTokens: Int,
        promptTime: Double,
        generateTime: Double
    ) -> AFMProfile {
        concreteService.stopAPIProfile(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            promptTime: promptTime,
            generateTime: generateTime
        )
    }

    func stopAPIProfileExtended(
        promptTokens: Int,
        completionTokens: Int,
        promptTime: Double,
        generateTime: Double
    ) -> AFMProfileExtended {
        concreteService.stopAPIProfileExtended(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            promptTime: promptTime,
            generateTime: generateTime
        )
    }

    func resetRequestPeakMemory() {
        concreteService.resetRequestPeakMemory()
    }

    func currentRequestPeakMemoryGib() -> Double? {
        concreteService.currentRequestPeakMemoryGib()
    }

    func effectiveResponseFormat(requestFormat: ResponseFormat?) -> ResponseFormat? {
        concreteService.effectiveResponseFormat(requestFormat: requestFormat)
    }

    func generate(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int?,
        minP: Double?,
        presencePenalty: Double?,
        seed: Int?,
        logprobs: Bool?,
        topLogprobs: Int?,
        tools: [RequestTool]?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?
    ) async throws -> AFMMLXChatGenerationResult {
        _ = try await load(progress: nil)
        return try await concreteService.generate(
            model: model,
            messages: messages,
            temperature: temperature,
            maxTokens: maxTokens,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            topK: topK,
            minP: minP,
            presencePenalty: presencePenalty,
            seed: seed,
            logprobs: logprobs,
            topLogprobs: topLogprobs,
            tools: tools,
            parallelToolCalls: parallelToolCalls,
            stop: stop,
            responseFormat: responseFormat,
            chatTemplateKwargs: chatTemplateKwargs
        )
    }

    func generateStreaming(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int?,
        minP: Double?,
        presencePenalty: Double?,
        seed: Int?,
        logprobs: Bool?,
        topLogprobs: Int?,
        tools: [RequestTool]?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?,
        preserveStructuralTags: Bool,
        requestId: String?
    ) async throws -> AFMMLXChatStreamingResult {
        _ = try await load(progress: nil)
        return try await concreteService.generateStreaming(
            model: model,
            messages: messages,
            temperature: temperature,
            maxTokens: maxTokens,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            topK: topK,
            minP: minP,
            presencePenalty: presencePenalty,
            seed: seed,
            logprobs: logprobs,
            topLogprobs: topLogprobs,
            tools: tools,
            parallelToolCalls: parallelToolCalls,
            stop: stop,
            responseFormat: responseFormat,
            chatTemplateKwargs: chatTemplateKwargs,
            preserveStructuralTags: preserveStructuralTags,
            requestId: requestId
        )
    }
}
