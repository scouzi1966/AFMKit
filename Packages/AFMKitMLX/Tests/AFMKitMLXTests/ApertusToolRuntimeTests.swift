import XCTest
@testable import AFMKitMLX

final class ApertusToolRuntimeTests: XCTestCase {
    func testNativeMultipleCallsAcrossAllChunkSplitsAndConsumedEOS() {
        let body = "<|tools_prefix|>[{\"weather\":{\"city\":\"Zürich\"}},{\"clock\":{}}]"
        for suffix in ["", "<|tools_suffix|>"] {
            let raw = body + suffix
            for split in 0...raw.count {
                let runtime = ToolCallStreamingRuntime(
                    toolCallStartTag: "<|tools_prefix|>", toolCallEndTag: "<|tools_suffix|>",
                    toolCallParser: nil, tools: nil, applyFixToolArgs: { $0 },
                    remapSingleKey: { key, _ in key })
                let boundary = raw.index(raw.startIndex, offsetBy: split)
                let events = runtime.process(piece: String(raw[..<boundary])).events
                    + runtime.process(piece: String(raw[boundary...])).events
                    + runtime.finishIncompleteToolCall()
                let names = events.compactMap { event -> String? in
                    if case .appendCollected(let call) = event { return call.function.name }
                    return nil
                }
                XCTAssertEqual(names, ["weather", "clock"], "suffix \(suffix), split \(split)")
            }
        }
    }

    func testMalformedNativeArrayDoesNotPartiallyExecute() {
        let text = "<|tools_prefix|>[{\"valid\":{}},{\"invalid\":null}]"
        let (calls, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: text, toolCallParser: nil, tools: nil)
        XCTAssertTrue(calls.isEmpty)
        XCTAssertEqual(remaining, text)
    }
}
