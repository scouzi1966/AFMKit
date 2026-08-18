import Foundation
import AFMOpenAICompat

// MARK: - Stable serving facade

public extension AFMMLXModel {
    var maxConcurrent: Int { service.maxConcurrent }

    var servingConfiguration: AFMMLXServingConfiguration {
        service.servingConfiguration
    }

    var defaultGuidedJsonSchema: ResponseFormat? {
        service.defaultGuidedJsonSchema
    }

    func normalizeModel(_ raw: String) -> String {
        service.normalizeModel(raw)
    }

    func resolvedToolCallParser(logBypass: Bool) -> String? {
        service.resolvedToolCallParser(logBypass: logBypass)
    }

    func tryReserveSlot() -> Bool {
        service.tryReserveSlot()
    }

    func waitForSlot(timeout: TimeInterval) async -> Bool {
        await service.waitForSlot(timeout: timeout)
    }

    func releaseSlot() {
        service.releaseSlot()
    }

    func ensureBatchMode(concurrency: Int) async throws {
        try await service.ensureBatchMode(concurrency: concurrency)
    }

    func releaseBatchReference() {
        service.releaseBatchReference()
    }

    func cancelBatchSlots(ids: Set<UUID>) async {
        await service.cancelBatchSlots(ids: ids)
    }

    func startAPIProfile() {
        service.startAPIProfile()
    }

    func stopAPIProfile(
        promptTokens: Int,
        completionTokens: Int,
        promptTime: Double,
        generateTime: Double
    ) -> AFMProfile {
        service.stopAPIProfile(
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
        service.stopAPIProfileExtended(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            promptTime: promptTime,
            generateTime: generateTime
        )
    }

    func resetRequestPeakMemory() {
        service.resetRequestPeakMemory()
    }

    func currentRequestPeakMemoryGib() -> Double? {
        service.currentRequestPeakMemoryGib()
    }

    func effectiveResponseFormat(requestFormat: ResponseFormat?) -> ResponseFormat? {
        service.effectiveResponseFormat(requestFormat: requestFormat)
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
        return try await service.generate(
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
        return try await service.generateStreaming(
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
