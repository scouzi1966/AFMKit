import AFMKitCore
@testable import AFMKitMLX
import AFMOpenAICompat
import XCTest

final class MLXOutputReasoningPolicyTests: XCTestCase {
    func testJSONDoesNotAcquireTemplateOpenedReasoning() {
        for format in ["json_object", "json_schema"] {
            let tags = MLXOutputReasoningPolicy.tags(
                responseFormat: .init(type: format), isRawPrompt: false,
                start: "<think>", end: "</think>")
            // GLM can retain this prompt suffix even when thinking is reduced.
            // A JSON grammar emits JSON directly; the prefix is not generated.
            let suffix = "[gMASK]<think>\n"
            let opensReasoning = tags.start.map {
                MLXModelService.promptSuffixOpensThink(suffix, startTag: $0, endTag: tags.end)
            } ?? false
            XCTAssertFalse(opensReasoning, format)
            XCTAssertNil(tags.start, format)
            XCTAssertNil(tags.end, format)
        }
    }

    func testOrdinaryChatRetainsTemplateOpenedReasoningAndTokenAccounting() {
        let tags = MLXOutputReasoningPolicy.tags(
            responseFormat: nil, isRawPrompt: false,
            start: "<think>", end: "</think>")
        XCTAssertEqual(tags.start, "<think>")
        XCTAssertEqual(tags.end, "</think>")
        XCTAssertTrue(MLXModelService.promptSuffixOpensThink(
            "<think>\n", startTag: tags.start!, endTag: tags.end))
        var translator = MLXStreamEventTranslator(
            thinkStartTag: tags.start, thinkEndTag: tags.end, maximumResponseTokens: 100)
        let events = [
            translator.consume(.init(syntheticText: tags.start!)),
            translator.consume(.init(text: "private")),
            translator.consume(.init(text: "</think>answer")),
            translator.finish()
        ].flatMap { $0 }
        XCTAssertEqual(events.compactMap { event -> String? in
            if case .responseText(_, let text, _) = event { return text }; return nil
        }.joined(), "answer")
        XCTAssertEqual(events.compactMap { event -> String? in
            if case .reasoningText(_, let text, _) = event { return text }; return nil
        }.joined(), "private")
        XCTAssertEqual(events.reduce(0) { count, event in
            switch event {
            case .responseText(_, _, let tokens), .reasoningText(_, _, let tokens):
                return count + tokens
            default:
                return count
            }
        }, 2)
    }

    func testRawCompletionsDoNotAcquireChatFraming() {
        let tags = MLXOutputReasoningPolicy.tags(
            responseFormat: nil, isRawPrompt: true,
            start: "<think>", end: "</think>")
        XCTAssertNil(tags.start)
        XCTAssertNil(tags.end)
    }

    func testBatchStopsPreserveJSONMarkerDataAtEverySplit() {
        let json = #"{"note":"<think>kept</think> <|channel|>final <tool_call>literal</tool_call>"}"#
        let output = json + "[STOP]discarded"
        for format in ["json_object", "json_schema"] {
            let tags = MLXOutputReasoningPolicy.tags(
                responseFormat: .init(type: format), isRawPrompt: false,
                start: "<think>", end: "</think>")
            for split in 0...output.count {
                var stopBuffer = ""
                var insideThink = false
                var collected = ""
                var stopped = false
                let boundary = output.index(output.startIndex, offsetBy: split)
                for text in [String(output[..<boundary]), String(output[boundary...])] {
                    let result = BatchScheduler.stopChunksToEmit(
                        from: text, stopBuffer: &stopBuffer,
                        activeStops: ["[STOP]"], maxStopLength: 6,
                        insideThink: &insideThink,
                        thinkStartTag: tags.start, thinkEndTag: tags.end)
                    collected += result.chunks.map(\.text).joined()
                    if result.stopped { stopped = true; break }
                }
                XCTAssertTrue(stopped, "\(format) split \(split)")
                XCTAssertFalse(insideThink, "\(format) split \(split)")
                XCTAssertEqual(collected, json, "\(format) split \(split)")
            }
        }
    }
}
