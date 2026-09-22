import AFMKitCore
@testable import AFMKitMLX
import XCTest

final class MLXBatchReasoningStopTests: XCTestCase {
    private func filter(
        _ pieces: [String], templateOpenedThink: Bool = false,
        stops: [String] = ["<AFM_STOP>"]
    ) -> (text: String, reasoning: String, stopped: Bool) {
        var buffer = ""
        var reasoningBuffer = ""
        var insideThink = templateOpenedThink
        var stopped = false
        var translator = MLXStreamEventTranslator(
            thinkStartTag: "<think>", thinkEndTag: "</think>", maximumResponseTokens: 1000)
        var events: [AFMGenerationEvent] = []
        if templateOpenedThink {
            events += translator.consume(.init(syntheticText: "<think>"))
        }
        for piece in pieces {
            let result = BatchScheduler.stopChunksToEmit(
                from: piece, stopBuffer: &buffer, reasoningBuffer: &reasoningBuffer, activeStops: stops,
                maxStopLength: stops.map(\.count).max() ?? 0,
                insideThink: &insideThink, thinkStartTag: "<think>", thinkEndTag: "</think>")
            for chunk in result.chunks { events += translator.consume(chunk) }
            if result.stopped { stopped = true; break }
        }
        var jsonFilter: MLXJSONStopFilter?
        for chunk in BatchScheduler.finishTextChunks(
            pendingText: "", jsonStopFilter: &jsonFilter, stopBuffer: &buffer,
            reasoningBuffer: &reasoningBuffer,
            activeStops: stops, maxStopLength: stops.map(\.count).max() ?? 0,
            insideThink: &insideThink, thinkStartTag: "<think>", thinkEndTag: "</think>",
            stoppedBySequence: &stopped)
        {
            events += translator.consume(chunk)
        }
        events += translator.finish()
        return (
            events.compactMap { if case .responseText(_, let text, _) = $0 { text } else { nil } }.joined(),
            events.compactMap { if case .reasoningText(_, let text, _) = $0 { text } else { nil } }.joined(),
            stopped)
    }

    func testGeneratedReasoningEndDelimiterReachesTheProductionTranslator() {
        let result = filter(["<think>", "private", "</think>", "AFM_PREFIX<AFM_STOP>AFM_SUFFIX"])
        XCTAssertEqual(result.text, "AFM_PREFIX")
        XCTAssertEqual(result.reasoning, "private")
        XCTAssertTrue(result.stopped)
    }

    func testTemplateOpenedReasoningIgnoresStopsUntilTheVisibleAnswerAtEverySplit() {
        let output = "private <AFM_STOP> stays in reasoning</think>AFM_PREFIX<AFM_STOP>AFM_SUFFIX"
        for offset in 0...output.count {
            let split = output.index(output.startIndex, offsetBy: offset)
            let result = filter([String(output[..<split]), String(output[split...])], templateOpenedThink: true)
            XCTAssertEqual(result.text, "AFM_PREFIX", "split \(offset)")
            XCTAssertEqual(result.reasoning, "private <AFM_STOP> stays in reasoning", "split \(offset)")
            XCTAssertTrue(result.stopped, "split \(offset)")
        }
    }

    func testGeneratedReasoningAndVisibleStopSurviveEveryChunkBoundary() {
        let output = "<think>private <AFM_STOP> remains</think>AFM_PREFIX<AFM_STOP>AFM_SUFFIX"
        for offset in 0...output.count {
            let split = output.index(output.startIndex, offsetBy: offset)
            let result = filter([String(output[..<split]), String(output[split...])])
            XCTAssertEqual(result.text, "AFM_PREFIX", "split \(offset)")
            XCTAssertEqual(result.reasoning, "private <AFM_STOP> remains", "split \(offset)")
            XCTAssertTrue(result.stopped, "split \(offset)")
        }
        let characterChunks = filter(output.map(String.init))
        XCTAssertEqual(characterChunks.text, "AFM_PREFIX")
        XCTAssertEqual(characterChunks.reasoning, "private <AFM_STOP> remains")
        XCTAssertTrue(characterChunks.stopped)
    }

    func testEarlierVisibleStopWinsBeforeALaterReasoningOpener() {
        let result = filter(["before FIRST later SECOND<think>private</think>answer"], stops: ["SECOND", "FIRST"])
        XCTAssertEqual(result.text, "before ")
        XCTAssertEqual(result.reasoning, "")
        XCTAssertTrue(result.stopped)
    }

    func testUnmatchedPartialDelimitersFlushAtEndWithoutBeingLost() {
        let result = filter(["<think>private</think>answer <thi"])
        XCTAssertEqual(result.text, "answer <thi")
        XCTAssertEqual(result.reasoning, "private")
        XCTAssertFalse(result.stopped)
    }

    func testNoStopsKeepsTheUnbufferedFastPath() {
        var buffer = ""
        var reasoningBuffer = ""
        var insideThink = false
        let result = BatchScheduler.stopChunksToEmit(
            from: "ordinary text", stopBuffer: &buffer, reasoningBuffer: &reasoningBuffer,
            activeStops: [], maxStopLength: 0,
            insideThink: &insideThink, thinkStartTag: "<think>", thinkEndTag: "</think>")
        XCTAssertEqual(result.chunks.map(\.text), ["ordinary text"])
        XCTAssertEqual(buffer, "")
        XCTAssertFalse(result.stopped)
    }

    func testAStopInsideASplitReasoningOpenerIsNotVisible() {
        let result = filter(["<think", ">private</think>answer"], stops: ["think"])
        XCTAssertEqual(result.text, "answer")
        XCTAssertEqual(result.reasoning, "private")
        XCTAssertFalse(result.stopped)
    }

    func testVisibleStopCanSpanAnInterveningReasoningBlock() {
        let result = filter(["BE", "<think>", "private", "</think>", "FOREtail"], stops: ["BEFORE"])
        XCTAssertEqual(result.text, "")
        XCTAssertEqual(result.reasoning, "private")
        XCTAssertTrue(result.stopped)
    }

    func testIncompleteOpeningDelimiterAtEOFStillHonorsVisibleStop() {
        let result = filter(["before <think"], stops: ["think"])
        XCTAssertEqual(result.text, "before <")
        XCTAssertEqual(result.reasoning, "")
        XCTAssertTrue(result.stopped)
    }

    func testVisiblePrefixRetainsItsChannelWhenReasoningNeverCloses() {
        let result = filter(["BE", "<think>private"], stops: ["STOP"])
        XCTAssertEqual(result.text, "BE")
        XCTAssertEqual(result.reasoning, "private")
        XCTAssertFalse(result.stopped)
    }

    func testVisiblePrefixRetainsItsChannelWithAPartialClosingDelimiterAtEOF() {
        let result = filter(["BE", "<think>private</thi"], stops: ["STOP"])
        XCTAssertEqual(result.text, "BE")
        XCTAssertEqual(result.reasoning, "private</thi")
        XCTAssertFalse(result.stopped)
    }

    func testEOFFramingDoesNotAddGeneratedTokensOrChangeLengthFinishReason() {
        var buffer = "BE"
        var reasoningBuffer = ""
        var insideThink = true
        var stopped = false
        var jsonFilter: MLXJSONStopFilter?
        let tail = BatchScheduler.finishTextChunks(
            pendingText: "", jsonStopFilter: &jsonFilter, stopBuffer: &buffer,
            reasoningBuffer: &reasoningBuffer, activeStops: ["STOP"], maxStopLength: 4,
            insideThink: &insideThink, thinkStartTag: "<think>", thinkEndTag: "</think>",
            stoppedBySequence: &stopped)
        XCTAssertEqual(tail.first?.text, "</think>")
        XCTAssertEqual(tail.first?.generatedTokenCountOverride, 0)
        var translator = MLXStreamEventTranslator(
            thinkStartTag: "<think>", thinkEndTag: "</think>", maximumResponseTokens: 64)
        _ = translator.consume(.init(syntheticText: "<think>"))
        for chunk in tail { _ = translator.consume(chunk) }
        _ = translator.consume(.init(text: "", promptTokens: 100, completionTokens: 64))
        XCTAssertTrue(translator.finish().contains(.completed(.length)))
    }
}
