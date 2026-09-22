import AFMKitCore
@testable import AFMKitMLX
import XCTest

final class MLXJSONReasoningFramerTests: XCTestCase {
    func testStopAndFramingBuffersRetainTokenMetadataUntilItsTextIsVisible() {
        for withStop in [false, true] {
            var filter = MLXJSONStopFilter(startTag: "<think>", endTag: "</think>",
                insideReasoning: true, stopSequences: ["[STOP]"])
            let pieces = ["private", "</thi", "nk>", "{\"note\":", "\"<think>keep\"}"]
                + (withStop ? ["[ST", "OP]hidden"] : [])
            var chunks: [StreamChunk] = []
            for (index, piece) in pieces.enumerated() {
                chunks += filter.consume(piece, logprobs: [
                    .init(token: piece, tokenId: index, logprob: -1, topTokens: [])])
            }
            chunks += filter.finish()
            XCTAssertEqual(chunks.map(\.text).joined(), "private</think>{\"note\":\"<think>keep\"}")
            XCTAssertEqual(chunks.flatMap { $0.logprobs ?? [] }.map(\.tokenId), [0, 1, 2, 3, 4])
            XCTAssertEqual(filter.stopped, withStop)
        }
    }

    func testPromptedJSONStopsPreserveMarkersAtEverySplitAndOneCharacterAtATime() {
        for json in [
            #"{"note":"<think>kept</think>"}"#,
            #"{"note":"<think>unclosed"}"#,
            #""<think>root string</think>""#,
            "```json\n" + #"{"note":"<think>fenced</think>"}"# + "\n```"
        ] {
            for templateOpened in [false, true] {
                let prefix = templateOpened ? "" : "<think>"
                let raw = prefix + "private [STOP] {draft}</think>\n" + json + "[STOP]discarded"
                let splits = (0...raw.count).map { offset -> [String] in
                    let index = raw.index(raw.startIndex, offsetBy: offset)
                    return [String(raw[..<index]), String(raw[index...])]
                } + [raw.map(String.init)]
                for pieces in splits {
                    var filter = MLXJSONStopFilter(startTag: "<think>", endTag: "</think>",
                        insideReasoning: templateOpened, stopSequences: ["[STOP]"])
                    var translator = MLXStreamEventTranslator(thinkStartTag: "<think>",
                        thinkEndTag: "</think>", maximumResponseTokens: 256, preserveReasoningMarkers: true)
                    var events: [AFMGenerationEvent] = templateOpened
                        ? translator.consume(.init(syntheticText: "<think>")) : []
                    for piece in pieces {
                        for chunk in filter.consume(piece) { events += translator.consume(chunk) }
                    }
                    for chunk in filter.finish() { events += translator.consume(chunk) }
                    events += translator.finish()
                    XCTAssertTrue(filter.stopped)
                    XCTAssertEqual(events.compactMap { event -> String? in
                        if case .responseText(_, let text, _) = event { return text }; return nil
                    }.joined(), "\n" + json)
                    XCTAssertEqual(events.compactMap { event -> String? in
                        if case .reasoningText(_, let text, _) = event { return text }; return nil
                    }.joined(), "private [STOP] {draft}")
                }
            }
        }
    }

    func testFencedAndRootStringJSONAreChunkIndependentWithoutStops() {
        for json in [#""<think>kept</think>""#,
                     "```json\n" + #"{"note":"<think>kept</think>"}"# + "\n```"] {
            for prefix in ["", "<think>private</think>"] {
                let raw = prefix + json
                for offset in 0...raw.count {
                    var translator = MLXStreamEventTranslator(thinkStartTag: "<think>",
                        thinkEndTag: "</think>", maximumResponseTokens: 256, preserveReasoningMarkers: true)
                    let index = raw.index(raw.startIndex, offsetBy: offset)
                    var events = translator.consume(.init(text: String(raw[..<index])))
                    events += translator.consume(.init(text: String(raw[index...])))
                    events += translator.finish()
                    XCTAssertEqual(events.compactMap { event -> String? in
                        if case .responseText(_, let text, _) = event { return text }; return nil
                    }.joined(), json, "split \(offset)")
                }
            }
        }
    }

    func testPartialReasoningDelimiterAtEndIsNotLost() {
        var filter = MLXJSONStopFilter(startTag: "<think>", endTag: "</think>",
            insideReasoning: true, stopSequences: ["STOP"])
        let chunks = filter.consume("private</thi") + filter.finish()
        XCTAssertEqual(chunks.map(\.text).joined(), "private</thi")
        XCTAssertFalse(filter.stopped)
    }
}
