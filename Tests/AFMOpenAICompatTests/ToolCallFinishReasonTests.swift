import Foundation
import XCTest
import AFMOpenAICompat

final class ToolCallFinishReasonTests: XCTestCase {
    private let call = ResponseToolCall(
        index: 0, id: "call_read", type: "function",
        function: .init(name: "read_file", arguments: #"{"path":"README.md"}"#)
    )

    func testLegacyToolConstructorsKeepToolCallsDefault() {
        let withoutReasoning = ChatCompletionResponse(model: "test", toolCalls: [call])
        let withReasoning = ChatCompletionResponse(
            model: "test", toolCalls: [call], reasoningContent: "Inspect the file."
        )
        XCTAssertEqual(withoutReasoning.choices.first?.finishReason, "tool_calls")
        XCTAssertEqual(withReasoning.choices.first?.finishReason, "tool_calls")
    }

    func testToolConstructorPreservesLengthWithoutDroppingArguments() throws {
        let response = ChatCompletionResponse(
            model: "test", toolCalls: [call], finishReason: "length",
            promptTokens: 10, completionTokens: 20
        )
        let encoded = try JSONEncoder().encode(response)
        let roundTrip = try JSONDecoder().decode(ChatCompletionResponse.self, from: encoded)
        XCTAssertEqual(roundTrip.choices.first?.finishReason, "length")
        XCTAssertEqual(roundTrip.choices.first?.message.toolCalls?.first?.function.arguments, call.function.arguments)
        XCTAssertEqual(roundTrip.usage.completionTokens, 20)
    }

    func testReasoningToolConstructorPreservesExplicitFinishReason() {
        let response = ChatCompletionResponse(
            model: "test", toolCalls: [call], reasoningContent: "Inspect the file.",
            finishReason: "length"
        )
        XCTAssertEqual(response.choices.first?.finishReason, "length")
        XCTAssertEqual(response.choices.first?.message.reasoningContent, "Inspect the file.")
        XCTAssertEqual(response.choices.first?.message.toolCalls?.first?.id, call.id)
    }
}
