import AFMKitCore
import AFMOpenAICompat
import Foundation
import MLXLMCommon
@testable import AFMKitMLX
import XCTest

final class ToolCallRejectedEnvelopeTests: XCTestCase {
    // Exact output_text from MissingToolRawCapture20261011/afm-cache/022.sse.
    // This is inert model-output data: the absent function name must not be
    // inferred, repaired, or dispatched as an executable tool call.
    private let capturedMalformedXML = #"""
    <tool_call>
    <function=<parameter name="cmd">cd /Volumes/edata/afm-isolated-coding-20261009/orbital && python3 - <<'PY'
    s=open('tests/scene.test.js').read()
    s=s.replace("""  const fakeCanvas = { style: {}, getContext: () => null };""","""  const fakeCanvas = {
        style: {},
        addEventListener() {},
        removeEventListener() {},
        setPointerCapture() {},
        releasePointerCapture() {},
        getContext: () => null,
      };""")
    open('tests/scene.test.js','w').write(s)
    PY
    node --test 2>&1 | tail -14
    </parameter>
    </function>
    </tool_call>
    """#

    private let validXML = "<tool_call><function=exec_command><parameter=cmd>echo verified</parameter></function></tool_call>"

    func testCapturedMalformedXMLRemainsVisibleThroughProductionChains() throws {
        for prefix in ["", "before\n", " \n\t"] {
            let raw = prefix + capturedMalformedXML + "\nafter\n"
            let (calls, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
                from: raw, toolCallParser: "qwen3_xml", tools: try tools())
            XCTAssertTrue(calls.isEmpty)
            XCTAssertEqual(remaining, raw)
            for pieces in splits(raw) {
                try assertProductionOutput(pieces, visible: raw, toolNames: [])
            }
        }
        try assertProductionOutput([capturedMalformedXML], visible: capturedMalformedXML, toolNames: [])
    }

    func testRejectedXMLAndAdjacentValidCallPreserveTextAndOneInvocation() throws {
        for body in [capturedMalformedXML + validXML, validXML + capturedMalformedXML] {
            let raw = "before\n" + body + "\nafter"
            for pieces in splits(raw) {
                try assertProductionOutput(
                    pieces, visible: "before\n" + capturedMalformedXML + "\nafter",
                    toolNames: ["exec_command"])
            }
        }
    }

    func testForcedNativeXMLStillRejectsJSONAndPreservesItsText() throws {
        let raw = #"before<tool_call>{"name":"exec_command","arguments":{"cmd":"echo verified"}}</tool_call>after"#
        let (calls, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: raw, toolCallParser: "qwen3_xml", tools: try tools())
        XCTAssertTrue(calls.isEmpty)
        XCTAssertEqual(remaining, raw)
        for pieces in splits(raw) {
            try assertProductionOutput(pieces, visible: raw, toolNames: [])
        }
    }

    func testUnacceptedNativeEOFTextIsPreservedExactlyOnce() throws {
        let cases = [
            "<tool_call>",
            " \n<tool_call>\n",
            String(capturedMalformedXML.dropLast("</tool_call>".count)),
            #"<tool_call>{"name":"exec_command","arguments":{"cmd":"unfinished"#,
        ]
        for raw in cases {
            for pieces in splits(raw) {
                try assertProductionOutput(pieces, visible: raw, toolNames: [])
            }
        }
    }

    func testRejectedEnvelopeDoesNotEraseReasoningOrCapturedUsage() throws {
        let reasoning = "The previous command needs another attempt."
        let raw = "<think>" + reasoning + "</think>" + capturedMalformedXML
        try assertProductionOutput([raw], visible: capturedMalformedXML, toolNames: [])
        for pipeline in Pipeline.allCases {
            let events = try productionEvents([raw], pipeline: pipeline, includeCapturedUsage: true)
            let retainedReasoning = events.compactMap { event -> String? in
                guard case .reasoningText(_, let text, _) = event else { return nil }
                return text
            }.joined()
            XCTAssertEqual(retainedReasoning, reasoning)
            XCTAssertTrue(events.contains(.usage(.init(
                inputTokens: 24435, cachedInputTokens: 23696, outputTokens: 243))))
            XCTAssertEqual(events.last, .completed(.stop))
        }
    }

    private enum Pipeline: CaseIterable { case serial, batch }

    private func assertProductionOutput(
        _ pieces: [String], visible expectedText: String, toolNames: [String],
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        for pipeline in Pipeline.allCases {
            let events = try productionEvents(pieces, pipeline: pipeline)
            let visible = events.compactMap { event -> String? in
                guard case .responseText(_, let text, _) = event else { return nil }
                return text
            }.joined()
            let completed = events.compactMap { event -> AFMToolCall? in
                guard case .toolCall(let call, .completed) = event else { return nil }
                return call
            }
            XCTAssertEqual(visible, expectedText, "\(pipeline)", file: file, line: line)
            XCTAssertEqual(completed.map(\.name), toolNames, "\(pipeline)", file: file, line: line)
            XCTAssertEqual(Set(completed.map(\.id)).count, toolNames.count, file: file, line: line)
            for call in completed {
                let arguments = try JSONSerialization.jsonObject(with: Data(call.arguments.utf8)) as? [String: String]
                XCTAssertEqual(arguments, ["cmd": "echo verified"], file: file, line: line)
            }
            if toolNames.isEmpty {
                XCTAssertFalse(events.contains { event in
                    if case .toolCall = event { return true }
                    return false
                }, "Rejected text must not start a tool call", file: file, line: line)
            }
        }
    }

    private func productionEvents(
        _ pieces: [String], pipeline: Pipeline, includeCapturedUsage: Bool = false
    ) throws -> [AFMGenerationEvent] {
        var chunks: [StreamChunk] = []
        switch pipeline {
        case .serial:
            let processor = ToolCallProcessor(format: .xmlFunction)
            var nextIndex = 0
            for piece in pieces {
                if let text = processor.processChunk(piece) { chunks.append(.init(text: text)) }
                chunks += processor.drainToolCalls().map {
                    MLXModelService.serialToolCallChunk($0, nextIndex: &nextIndex)
                }
            }
            if let text = processor.finishPendingText() { chunks.append(.init(text: text)) }
        case .batch:
            let runtime = ToolCallStreamingRuntime(
                toolCallStartTag: "<tool_call>", toolCallEndTag: "</tool_call>",
                toolCallParser: "qwen3_xml", tools: try tools(), repairToolArguments: false,
                applyFixToolArgs: { $0 }, remapSingleKey: { key, _ in key })
            for piece in pieces {
                let output = runtime.process(piece: piece)
                let text = output.handled ? (output.passthroughText ?? "") : piece
                if !text.isEmpty { chunks.append(.init(text: text)) }
                chunks += BatchScheduler.streamChunksToEmit(from: output.events)
            }
            chunks += BatchScheduler.streamChunksToEmit(from: runtime.finishIncompleteToolCall())
            let pending = runtime.finishPendingText()
            if !pending.isEmpty { chunks.append(.init(text: pending)) }
            XCTAssertTrue(runtime.finishIncompleteToolCall().isEmpty)
            XCTAssertEqual(runtime.finishPendingText(), "", "EOF must not emit held text twice")
        }
        if includeCapturedUsage {
            chunks.append(.init(text: "", promptTokens: 24435, completionTokens: 243, cachedTokens: 23696))
        }
        // Both service paths pass through the provider's raw fallback. Batch
        // rejections must remain literal even when that layer sees them again.
        var fallback = AFMMLXRawToolStreamFallback(
            toolCallStartTag: "<tool_call>", toolCallEndTag: "</tool_call>",
            toolCallParser: "qwen3_xml", tools: try tools(),
            applyFixToolArgs: { $0 }, remapSingleKey: { key, _ in key })
        var normalizedChunks: [StreamChunk] = []
        for chunk in chunks { normalizedChunks += fallback.consume(chunk) }
        normalizedChunks += fallback.finish()
        XCTAssertTrue(fallback.finish().isEmpty, "EOF must not emit held text twice")
        var translator = MLXStreamEventTranslator(
            thinkStartTag: "<think>", thinkEndTag: "</think>", maximumResponseTokens: 1024, tools: try tools())
        var events = normalizedChunks.flatMap { translator.consume($0) }
        events += translator.finish()
        return events
    }

    private func tools() throws -> [RequestTool] {
        try JSONDecoder().decode([RequestTool].self, from: Data(#"[{"type":"function","function":{"name":"exec_command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}]"#.utf8))
    }

    private func splits(_ text: String) -> [[String]] {
        (0...text.count).map { offset in
            let index = text.index(text.startIndex, offsetBy: offset)
            return [String(text[..<index]), String(text[index...])]
        } + [text.map(String.init)]
    }
}
