import Foundation
import MLXLMCommon
import AFMOpenAICompat
@testable import AFMKitMLX
import XCTest

final class GLMNativeToolDelimiterTests: XCTestCase {
    private let value = "\n  Delimiters used by this format: </tool_call> and </think> and <|im_end|>. <tool_call>\n"
    private var raw: String {
        "<tool_call>write_file<arg_key>content</arg_key><arg_value>" + value + "</arg_value></tool_call>"
    }
    private var specs: [[String: any Sendable]] {
        let properties: [String: any Sendable] = ["content": ["type": "string"]]
        let parameters: [String: any Sendable] = ["type": "object", "properties": properties]
        let function: [String: any Sendable] = ["name": "write_file", "parameters": parameters]
        return [["type": "function", "function": function]]
    }
    private func tools() throws -> [RequestTool] {
        try JSONDecoder().decode([RequestTool].self, from: Data(#"[{"type":"function","function":{"name":"write_file","parameters":{"type":"object","properties":{"content":{"type":"string"}},"required":["content"]}}}]"#.utf8))
    }

    func testDirectParserPreservesLiteralDelimitersAndStringWhitespace() throws {
        let parsed = try XCTUnwrap(GLM4ToolCallParser().parse(content: raw, tools: specs))
        XCTAssertEqual(parsed.function.arguments["content"]?.anyValue as? String, value)
    }

    func testSerialAndBatchToolFramingPreserveLiteralDataAtEverySplit() throws {
        for offset in 0...raw.count {
            let boundary = raw.index(raw.startIndex, offsetBy: offset)
            let pieces = [String(raw[..<boundary]), String(raw[boundary...])]
            let processor = ToolCallProcessor(format: .glm4, tools: specs)
            let visible = pieces.map { processor.processChunk($0) ?? "" }.joined()
                + (processor.finishPendingText() ?? "")
            XCTAssertEqual(visible, "", "serial split \(offset)")
            XCTAssertEqual(processor.toolCalls.count, 1)
            XCTAssertEqual(processor.toolCalls.first?.function.arguments["content"]?.anyValue as? String, value)
            for parser in [nil, "glm4"] as [String?] {
                let runtime = ToolCallStreamingRuntime(toolCallStartTag: "<tool_call>",
                    toolCallEndTag: "</tool_call>", toolCallParser: parser, tools: try tools(),
                    repairToolArguments: false, applyFixToolArgs: { $0 }, remapSingleKey: { key, _ in key })
                let events = pieces.flatMap { runtime.process(piece: $0).events }
                    + runtime.finishIncompleteToolCall()
                let calls = BatchScheduler.completedToolCallsToEmit(from: events)
                let last = try XCTUnwrap(calls.last, "batch split \(offset)")
                let args = try JSONSerialization.jsonObject(with: Data(last.function.arguments.utf8)) as? [String: Any]
                XCTAssertEqual(args?["content"] as? String, value, "batch split \(offset)")
            }
        }
    }

    func testFallbackUsesNativeArgumentAwareEnvelope() throws {
        let (calls, visible) = MLXModelService.extractToolCallsFallback(from: raw, tools: try tools())
        XCTAssertEqual(visible, "")
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.function.arguments["content"]?.anyValue as? String, value)
    }

    func testIncompleteArgumentDoesNotTerminateAtLiteralEndTag() {
        let partial = "<tool_call>write_file<arg_key>content</arg_key><arg_value>keep </tool_call> data"
        let processor = ToolCallProcessor(format: .glm4, tools: specs)
        XCTAssertNil(processor.processChunk(partial))
        XCTAssertTrue(processor.toolCalls.isEmpty)
        XCTAssertEqual(processor.finishPendingText(), partial)
    }

    func testMixedToolsAndStructuredStopsOnlyFilterPassthroughText() throws {
        let runtime = ToolCallStreamingRuntime(toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>", toolCallParser: "glm4", tools: try tools(),
            repairToolArguments: false, applyFixToolArgs: { $0 }, remapSingleKey: { key, _ in key })
        var filter = MLXJSONStopFilter(startTag: "<think>", endTag: "</think>",
            insideReasoning: false, stopSequences: ["[STOP]"])
        let output = runtime.process(piece: raw.replacingOccurrences(of: "Delimiters", with: "[STOP]Delimiters"))
        if let text = output.passthroughText { _ = filter.consume(text) }
        XCTAssertTrue(output.handled)
        XCTAssertFalse(filter.stopped)
        let call = try XCTUnwrap(BatchScheduler.completedToolCallsToEmit(from: output.events).last)
        XCTAssertTrue(call.function.arguments.contains("[STOP]"))
    }

    func testProviderFallbackPreservesIncompleteNativeArgumentInsteadOfInventingCall() throws {
        let partial = "<tool_call>write_file<arg_key>content</arg_key><arg_value>keep </tool_call> data"
        for offset in 0...partial.count {
            var fallback = AFMMLXRawToolStreamFallback(toolCallStartTag: "<tool_call>",
                toolCallEndTag: "</tool_call>", toolCallParser: nil, tools: try tools(),
                applyFixToolArgs: { $0 }, remapSingleKey: { key, _ in key })
            let boundary = partial.index(partial.startIndex, offsetBy: offset)
            let chunks = fallback.consume(.init(text: String(partial[..<boundary])))
                + fallback.consume(.init(text: String(partial[boundary...]))) + fallback.finish()
            XCTAssertEqual(chunks.map(\.text).joined(), partial)
            XCTAssertTrue(chunks.flatMap { $0.toolCalls ?? [] }.isEmpty)
        }
    }
}
