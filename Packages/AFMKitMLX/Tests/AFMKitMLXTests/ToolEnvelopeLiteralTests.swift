import Foundation
import MLXLMCommon
import AFMKitCore
import AFMOpenAICompat
@testable import AFMKitMLX
import XCTest

/// Framing must not reinterpret marker-shaped argument data as control tokens.
/// Exercise the same saved output through the serial and batch paths; no model
/// sampling or GPU is needed to distinguish an engine bug from model behavior.
final class ToolEnvelopeLiteralTests: XCTestCase {
    private let value = #"Delimiters: </tool_call> and </think> and <|im_end|> and <tool_call> and </function>. Example: <function=not_a_call></function>. Quotes: "ready"; café 🚀."#
    private var specs: [[String: any Sendable]] {
        let properties: [String: any Sendable] = ["path": ["type": "string"], "content": ["type": "string"]]
        let parameters: [String: any Sendable] = ["type": "object", "properties": properties]
        let function: [String: any Sendable] = ["name": "write_file", "parameters": parameters]
        return [["type": "function", "function": function]]
    }
    private func tools() throws -> [RequestTool] {
        try JSONDecoder().decode([RequestTool].self, from: Data(#"[{"type":"function","function":{"name":"write_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]"#.utf8))
    }
    private func jsonBody() throws -> String {
        let data = try JSONSerialization.data(withJSONObject: [
            "name": "write_file", "arguments": ["path": "notes.md", "content": value]
        ], options: [.sortedKeys, .withoutEscapingSlashes])
        return try XCTUnwrap(String(data: data, encoding: .utf8))
    }
    private var xmlBody: String {
        "<function=write_file><parameter=path>notes.md</parameter><parameter=content>\(value)</parameter></function>"
    }
    private func wrapped(_ body: String) -> String { "<tool_call>\(body)</tool_call>" }
    private func arguments(_ json: String) throws -> [String: String] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: String])
    }
    private func check(_ call: ToolCall, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(call.function.name, "write_file", file: file, line: line)
        XCTAssertEqual(call.function.arguments.count, 2, file: file, line: line)
        XCTAssertEqual(call.function.arguments["path"]?.anyValue as? String, "notes.md", file: file, line: line)
        XCTAssertEqual(call.function.arguments["content"]?.anyValue as? String, value, file: file, line: line)
    }
    private func splits(_ text: String) -> [[String]] {
        (0...text.count).map { offset in
            let i = text.index(text.startIndex, offsetBy: offset)
            return [String(text[..<i]), String(text[i...])]
        } + [text.map(String.init)]
    }

    func testDirectJSONPreservesMarkersInsideStringsWithAndWithoutEnvelope() throws {
        let parser = JSONToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        for text in [try jsonBody(), wrapped(try jsonBody())] {
            check(try XCTUnwrap(parser.parse(content: text, tools: specs)))
        }
    }

    func testJSONEscapesAndAlternateEndTagsAtEverySplit() throws {
        let contents = [#"odd: \" </tool_call>"#, #"even: \\ </tool_call>"#, #"end \\"#, "café 🚀 [END_TOOL]"]
        for content in contents {
            let data = try JSONSerialization.data(withJSONObject: [
                "name": "write_file", "arguments": ["content": content]
            ], options: [.withoutEscapingSlashes])
            let body = try XCTUnwrap(String(data: data, encoding: .utf8))
            for (start, end) in [("<tool_call>", "</tool_call>"), ("[START_TOOL]", "[END_TOOL]")] {
                let text = start + body + end
                for pieces in splits(text) {
                    var scanner = ToolCallEnvelopeScanner(syntax: .json)
                    var buffer = ""
                    var found: Range<String.Index>?
                    for piece in pieces {
                        buffer += piece
                        found = scanner.closingTagRange(in: buffer, endTag: end) ?? found
                    }
                    let boundary = try XCTUnwrap(found)
                    XCTAssertEqual(String(buffer[boundary.upperBound...]), "")
                    let parsed = try XCTUnwrap(JSONToolCallParser(startTag: start, endTag: end)
                        .parse(content: buffer, tools: specs))
                    XCTAssertEqual(parsed.function.arguments["content"]?.anyValue as? String, content)
                }
            }
        }
    }

    func testDirectXMLPreservesFunctionAndEnvelopeEndsInsideParameters() throws {
        for text in [xmlBody, wrapped(xmlBody)] {
            check(try XCTUnwrap(XMLFunctionParser().parse(content: text, tools: specs)))
        }
    }

    func testSerialProcessorPreservesMarkersAtEveryChunkSplitAndAdjacentCalls() throws {
        let cases: [(ToolCallFormat, String)] = [
            (.json, wrapped(try jsonBody())), (.xmlFunction, wrapped(xmlBody)), (.xmlFunction, xmlBody)
        ]
        for (format, text) in cases {
            for pieces in splits(text + text + "after") {
                let processor = ToolCallProcessor(format: format, tools: specs)
                let visible = pieces.map { processor.processChunk($0) ?? "" }.joined()
                    + (processor.finishPendingText() ?? "")
                XCTAssertEqual(visible, "after")
                XCTAssertEqual(processor.toolCalls.count, 2)
                check(try XCTUnwrap(processor.toolCalls.first))
                check(try XCTUnwrap(processor.toolCalls.last))
            }
        }
    }

    func testRuntimeAndStreamedArgumentsPreserveMarkersAtEverySplit() throws {
        let cases: [(String?, String)] = [
            (nil, wrapped(try jsonBody())), ("afm_adaptive_xml", wrapped(try jsonBody())),
            ("qwen3_xml", wrapped(xmlBody))
        ]
        for (parser, text) in cases {
            for pieces in splits(text + text) {
                let runtime = ToolCallStreamingRuntime(toolCallStartTag: "<tool_call>",
                    toolCallEndTag: "</tool_call>", toolCallParser: parser, tools: try tools(),
                    repairToolArguments: false, applyFixToolArgs: { $0 }, remapSingleKey: { key, _ in key })
                var events: [ToolCallStreamingEvent] = []
                var visible = ""
                for piece in pieces {
                    let output = runtime.process(piece: piece)
                    events += output.events
                    visible += output.handled ? (output.passthroughText ?? "") : piece
                }
                events += runtime.finishIncompleteToolCall()
                visible += runtime.finishPendingText()
                XCTAssertEqual(visible, "")
                let calls = BatchScheduler.completedToolCallsToEmit(from: events)
                XCTAssertEqual(calls.count, 2)
                let expected = ["path": "notes.md", "content": value]
                for call in calls {
                    XCTAssertEqual(try arguments(call.function.arguments), expected)
                }
                var streamed: [Int: String] = [:]
                for delta in BatchScheduler.deltaToolCallsToEmit(from: events) {
                    streamed[delta.index, default: ""] += delta.function?.arguments ?? ""
                }
                XCTAssertEqual(streamed.count, 2)
                for value in streamed.values { XCTAssertEqual(try arguments(value), expected) }
            }
        }
    }

    func testCompletedFallbackPreservesMarkersAndAdjacentCalls() throws {
        let cases: [(String?, String)] = [
            (nil, wrapped(try jsonBody())), ("afm_adaptive_xml", wrapped(try jsonBody())),
            ("qwen3_xml", wrapped(xmlBody)), (nil, wrapped(xmlBody))
        ]
        for (parser, text) in cases {
            let (calls, visible) = ToolCallStreamingRuntime.parseCompletedToolCalls(
                from: "before" + text + text + "after", toolCallParser: parser, tools: try tools())
            XCTAssertEqual(visible, "beforeafter")
            XCTAssertEqual(calls.count, 2)
            check(try XCTUnwrap(calls.first))
            check(try XCTUnwrap(calls.last))
        }
    }

    func testIncompleteMarkerBearingCallsRemainBufferedBySerialProcessor() throws {
        let cases: [(ToolCallFormat, String)] = [
            (.json, #"<tool_call>{"name":"write_file","arguments":{"content":"keep </tool_call> data"#),
            (.xmlFunction, "<tool_call><function=write_file><parameter=content>keep </tool_call> data")
        ]
        for (format, text) in cases {
            for pieces in splits(text) {
                let processor = ToolCallProcessor(format: format, tools: specs)
                XCTAssertEqual(pieces.map { processor.processChunk($0) ?? "" }.joined(), "")
                XCTAssertTrue(processor.toolCalls.isEmpty)
                XCTAssertEqual(processor.finishPendingText(), text)
            }
        }
    }

    func testDeepSeekDSMLKeepsOtherFormatsMarkersAsLiteralData() throws {
        let raw = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"write_file\"><｜DSML｜parameter name=\"path\" string=\"true\">notes.md</｜DSML｜parameter><｜DSML｜parameter name=\"content\" string=\"true\">\(value)</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>"
        let (calls, visible) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: raw, toolCallParser: nil, tools: try tools())
        XCTAssertEqual(visible, "")
        XCTAssertEqual(calls.count, 1)
        check(try XCTUnwrap(calls.first))
    }

    func testDeepSeekRuntimeDoesNotTreatRawQuotesAsJSONFraming() throws {
        let content = "He said \"hello. </tool_call> is literal."
        let raw = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"write_file\"><｜DSML｜parameter name=\"content\" string=\"true\">\(content)</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>"
        for (input, count) in [(raw, 1), (raw + raw, 2)] {
            for pieces in splits(input + "after") {
                let runtime = ToolCallStreamingRuntime(toolCallStartTag: "<｜DSML｜tool_calls>",
                    toolCallEndTag: "</｜DSML｜tool_calls>", toolCallParser: nil, tools: try tools(),
                    repairToolArguments: false, applyFixToolArgs: { $0 }, remapSingleKey: { key, _ in key })
                var events: [ToolCallStreamingEvent] = []
                var visible = ""
                for piece in pieces {
                    let output = runtime.process(piece: piece)
                    events += output.events
                    visible += output.handled ? (output.passthroughText ?? "") : piece
                }
                let calls = BatchScheduler.completedToolCallsToEmit(from: events)
                XCTAssertEqual(calls.count, count)
                XCTAssertEqual(visible, "after")
                for call in calls { XCTAssertEqual(try arguments(call.function.arguments)["content"], content) }
                XCTAssertFalse(runtime.inToolCall)
            }
        }
    }

    func testBareFallbackDoesNotPromoteJSONExamplesIntoCallsOrTruncateXML() throws {
        for text in [try jsonBody(), xmlBody] {
            let (calls, visible) = MLXModelService.extractToolCallsFallback(from: text, tools: try tools())
            XCTAssertEqual(visible, "")
            XCTAssertEqual(calls.count, 1)
            check(try XCTUnwrap(calls.first))
        }
        let nested = #"<tool_call>{"name":"not_a_call","arguments":{}}</tool_call>"#
        let raw = "<function=write_file><parameter=content>\(nested)</parameter></function>"
        let (calls, visible) = MLXModelService.extractToolCallsFallback(from: raw, tools: try tools())
        XCTAssertEqual(visible, "")
        XCTAssertEqual(calls.count, 1)
        let call = try XCTUnwrap(calls.first)
        XCTAssertEqual(call.function.name, "write_file")
        XCTAssertEqual(call.function.arguments["content"]?.anyValue as? String, nested)
    }

    func testAdaptiveMalformedRepairsCoexistWithValidCallsInOrder() throws {
        let valid = wrapped(try jsonBody())
        let malformed = #"<tool_call>{"name="write_file", "arguments":{"path":"second.md","content":"second"}}</tool_call>"#
        for (text, paths) in [(valid + malformed, ["notes.md", "second.md"]), (malformed + valid, ["second.md", "notes.md"])] {
            let (calls, visible) = ToolCallStreamingRuntime.parseCompletedToolCalls(
                from: text, toolCallParser: "afm_adaptive_xml", tools: try tools())
            XCTAssertEqual(visible, "")
            XCTAssertEqual(calls.map { $0.function.arguments["path"]?.anyValue as? String }, paths)
            for pieces in splits(text + "after") {
                let runtime = ToolCallStreamingRuntime(toolCallStartTag: "<tool_call>",
                    toolCallEndTag: "</tool_call>", toolCallParser: "afm_adaptive_xml", tools: try tools(),
                    repairToolArguments: true, applyFixToolArgs: { $0 }, remapSingleKey: { key, _ in key })
                var events: [ToolCallStreamingEvent] = []
                var visible = ""
                for piece in pieces {
                    let output = runtime.process(piece: piece)
                    events += output.events
                    visible += output.handled ? (output.passthroughText ?? "") : piece
                }
                events += runtime.finishIncompleteToolCall()
                visible += runtime.finishPendingText()
                XCTAssertEqual(visible, "after")
                let calls = BatchScheduler.completedToolCallsToEmit(from: events)
                XCTAssertEqual(try calls.map { try arguments($0.function.arguments)["path"] }, paths)
            }
        }
    }

    func testExplicitRepairRetainsLegacyUnclosedXMLParameterSalvage() throws {
        let partial = "<function=write_file><parameter=content>partial</function>"
        let repaired = try XCTUnwrap(MLXModelService.parseXMLFunction(partial, repairArguments: true))
        XCTAssertEqual(repaired.function.arguments["content"]?.anyValue as? String, "partial")
        XCTAssertNil(MLXModelService.parseXMLFunction(partial, repairArguments: false))
        let runtime = ToolCallStreamingRuntime(toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>", toolCallParser: "afm_adaptive_xml", tools: try tools(),
            repairToolArguments: true, applyFixToolArgs: { $0 }, remapSingleKey: { key, _ in key })
        let events = runtime.process(piece: wrapped(partial)).events + runtime.finishIncompleteToolCall()
        let call = try XCTUnwrap(BatchScheduler.completedToolCallsToEmit(from: events).last)
        XCTAssertEqual(try arguments(call.function.arguments)["content"], "partial")
        for raw in [partial, wrapped(partial)] {
            let (calls, visible) = ToolCallStreamingRuntime.parseCompletedToolCalls(
                from: raw, toolCallParser: "afm_adaptive_xml", tools: try tools())
            XCTAssertEqual(visible, "")
            XCTAssertEqual(calls.count, 1)
            XCTAssertEqual(calls.first?.function.arguments["content"]?.anyValue as? String, "partial")
        }
    }

    func testUnrepairableAdaptiveEOFTextIsNotDiscarded() throws {
        let raw = #"<tool_call>{"name="write_file", "arguments":{"content":"unfinished </tool_call>"#
        for pieces in splits(raw) {
            let runtime = ToolCallStreamingRuntime(toolCallStartTag: "<tool_call>",
                toolCallEndTag: "</tool_call>", toolCallParser: "afm_adaptive_xml", tools: try tools(),
                repairToolArguments: true, applyFixToolArgs: { $0 }, remapSingleKey: { key, _ in key })
            let outputs = pieces.map { runtime.process(piece: $0) }
            let events = outputs.flatMap(\.events) + runtime.finishIncompleteToolCall()
            XCTAssertTrue(BatchScheduler.completedToolCallsToEmit(from: events).isEmpty)
            XCTAssertEqual(outputs.compactMap(\.passthroughText).joined() + runtime.finishPendingText(), raw)
        }
    }

    func testSerialCallsCrossProviderFallbackAndTranslatorWithoutChangingData() throws {
        for (format, parser, raw) in [(ToolCallFormat.json, "hermes", wrapped(try jsonBody())),
                                      (.xmlFunction, "qwen3_xml", wrapped(xmlBody))] {
            for pieces in splits(raw) {
                let processor = ToolCallProcessor(format: format, tools: specs)
                var nextIndex = 0
                var chunks: [StreamChunk] = []
                for piece in pieces {
                    if let text = processor.processChunk(piece) { chunks.append(.init(text: text)) }
                    chunks += processor.drainToolCalls().map { MLXModelService.serialToolCallChunk($0, nextIndex: &nextIndex) }
                }
                if let text = processor.finishPendingText() { chunks.append(.init(text: text)) }
                var fallback = AFMMLXRawToolStreamFallback(toolCallStartTag: "<tool_call>",
                    toolCallEndTag: "</tool_call>", toolCallParser: parser, tools: try tools(),
                    applyFixToolArgs: { $0 }, remapSingleKey: { key, _ in key })
                var translator = MLXStreamEventTranslator(thinkStartTag: "<think>", thinkEndTag: "</think>",
                    maximumResponseTokens: 128, tools: try tools())
                var events: [AFMGenerationEvent] = []
                for chunk in chunks {
                    for normalized in fallback.consume(chunk) { events += translator.consume(normalized) }
                }
                for chunk in fallback.finish() { events += translator.consume(chunk) }
                events += translator.finish()
                let calls = events.compactMap { event -> AFMToolCall? in
                    guard case .toolCall(let call, .completed) = event else { return nil }
                    return call
                }
                XCTAssertEqual(calls.count, 1)
                let call = try XCTUnwrap(calls.first)
                XCTAssertEqual(try arguments(call.arguments), ["path": "notes.md", "content": value])
            }
        }
    }

    func testBatchEOFRecoveredTextHonorsStopsWithoutFilteringToolData() throws {
        let raw = #"<tool_call>{"name="write_file", "arguments":{"content":"literal STOP data"}}</tool_call>beforeSTOPafter"#
        for useJSONFilter in [false, true] {
            let runtime = ToolCallStreamingRuntime(toolCallStartTag: "<tool_call>",
                toolCallEndTag: "</tool_call>", toolCallParser: "afm_adaptive_xml", tools: try tools(),
                repairToolArguments: true, applyFixToolArgs: { $0 }, remapSingleKey: { key, _ in key })
            let events = runtime.process(piece: raw).events + runtime.finishIncompleteToolCall()
            let calls = BatchScheduler.completedToolCallsToEmit(from: events)
            XCTAssertEqual(calls.count, 1)
            XCTAssertEqual(try arguments(XCTUnwrap(calls.first).function.arguments)["content"], "literal STOP data")
            var jsonFilter: MLXJSONStopFilter? = useJSONFilter
                ? MLXJSONStopFilter(startTag: "<think>", endTag: "</think>", insideReasoning: false, stopSequences: ["STOP"])
                : nil
            var buffer = ""
            var insideThink = false
            var stopped = false
            let chunks = BatchScheduler.finishTextChunks(pendingText: runtime.finishPendingText(),
                jsonStopFilter: &jsonFilter, stopBuffer: &buffer, activeStops: ["STOP"], maxStopLength: 4,
                insideThink: &insideThink, thinkStartTag: "<think>", thinkEndTag: "</think>", stoppedBySequence: &stopped)
            XCTAssertEqual(chunks.map(\.text).joined(), "before")
            XCTAssertTrue(stopped)
            XCTAssertTrue(buffer.isEmpty)
        }
    }
}
