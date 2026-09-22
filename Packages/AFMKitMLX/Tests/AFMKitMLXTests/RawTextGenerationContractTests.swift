import AFMKitCore
import MLXLMCommon
import XCTest

@testable import AFMKitMLX

final class RawTextGenerationContractTests: XCTestCase {
    func testRawGenerationPreservesToolMarkupWithoutMutatingChatConfiguration() {
        let formats: [ToolCallFormat?] = [nil, .xmlFunction, .json, .glm4, .apertus]
        for format in formats {
            let chat = ModelConfiguration(id: "test/model", toolCallFormat: format)
            let raw = MLXModelService.generationConfiguration(chat, rawPrompt: "")
            XCTAssertEqual(raw.toolCallFormat, ToolCallFormat.none)
            XCTAssertEqual(chat.toolCallFormat, format)
            XCTAssertEqual(
                MLXModelService.generationConfiguration(chat, rawPrompt: nil), chat)

            let processor = ToolCallProcessor(format: raw.toolCallFormat ?? .json)
            let chunks = ["</think>\n", "<tool_", "call>\n<function=read_file>",
                "<parameter=path>src/user.js</parameter></function></tool_call>"]
            var emitted = chunks.compactMap { processor.processChunk($0) }.joined()
            emitted += processor.finishPendingText() ?? ""
            XCTAssertEqual(emitted, chunks.joined())
            XCTAssertTrue(processor.drainToolCalls(stopAfterFirst: false).isEmpty)
        }
    }

    private struct PlainModel: AFMModel {
        let descriptor = AFMModelDescriptor(
            providerID: "test",
            modelID: "plain",
            displayName: "Plain",
            capabilities: [.text]
        )

        func availability() async -> AFMModelAvailability { .available }
        func load(progress: (@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor {
            descriptor
        }
        func respond(to request: AFMRequest) async throws -> AFMModelResponse { .init() }
        func streamResponse(
            to request: AFMRequest
        ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
            AsyncThrowingStream { $0.finish() }
        }
    }

    private struct RawModel: AFMModel, AFMRawTextGenerating, AFMGenerationAdmitting {
        let descriptor = AFMModelDescriptor(
            providerID: "test",
            modelID: "raw",
            displayName: "Raw",
            capabilities: [.text, .streaming]
        )

        func availability() async -> AFMModelAvailability { .available }
        func load(progress: (@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor {
            descriptor
        }
        func respond(to request: AFMRequest) async throws -> AFMModelResponse { .init() }
        func streamResponse(
            to request: AFMRequest
        ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
            AsyncThrowingStream { $0.finish() }
        }

        func rawTextGenerationEvents(
            for request: AFMRawTextGenerationRequest
        ) -> AsyncStream<AFMRawTextGenerationEvent> {
            AsyncStream { continuation in
                continuation.yield(.textDelta(text: request.prompt, tokenID: 1, timestamp: 2))
                continuation.yield(
                    .completed(
                        AFMRawTextGenerationResult(
                            finishReason: request.ignoreEndOfSequence ? .length : .stop,
                            promptTokens: 1,
                            completionTokens: 1,
                            totalTokens: 2
                        )
                    )
                )
                continuation.finish()
            }
        }

        func admitGeneration(timeout: Duration?) async throws -> AFMGenerationLease {
            let token = AFMNoopInferenceTelemetryObserver().requestAccepted(at: 0)
            return AFMGenerationLease(telemetryToken: token) {}
        }
    }

    func testAnyModelRetainsOnlyConformingRawCapability() async {
        XCTAssertNil(AnyAFMModel(PlainModel()).rawTextGenerator)
        XCTAssertNil(AnyAFMModel(PlainModel()).generationAdmitter)
        let model = AnyAFMModel(RawModel())
        let generator = model.rawTextGenerator
        XCTAssertNotNil(generator)
        XCTAssertNotNil(model.generationAdmitter)

        let request = AFMRawTextGenerationRequest(
            prompt: "raw prompt",
            modelID: "raw",
            maximumOutputTokens: 1,
            ignoreEndOfSequence: true
        )
        var events: [AFMRawTextGenerationEvent] = []
        if let generator {
            for await event in generator.rawTextGenerationEvents(for: request) {
                events.append(event)
            }
        }
        XCTAssertEqual(events.first, .textDelta(text: "raw prompt", tokenID: 1, timestamp: 2))
        XCTAssertEqual(
            events.last,
            .completed(
                AFMRawTextGenerationResult(
                    finishReason: .length,
                    promptTokens: 1,
                    completionTokens: 1,
                    totalTokens: 2
                )
            )
        )
    }

    func testGenerationOptionsRetainOldInitializerAndAddIgnoreEOSOverload() {
        let legacy = AFMGenerationOptions(temperature: 0.5, stopSequences: ["stop"])
        XCTAssertFalse(legacy.ignoreEndOfSequence)

        let extended = AFMGenerationOptions(
            temperature: 0.5,
            reasoningEnabled: false,
            stopSequences: ["stop"],
            ignoreEndOfSequence: true
        )
        XCTAssertTrue(extended.ignoreEndOfSequence)
        XCTAssertEqual(extended.reasoningEnabled, false)
    }

}
