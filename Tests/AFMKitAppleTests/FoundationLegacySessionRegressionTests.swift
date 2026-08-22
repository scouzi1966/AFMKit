#if canImport(FoundationModels)
import Foundation
import FoundationModels
import XCTest
@testable import AFMKitApple

final class FoundationStopSequenceFilterTests: XCTestCase {
    func testWithholdsSplitStopPrefixesAndDropsStopAndSuffix() {
        var filter = FoundationStopSequenceFilter(stopSequences: ["<END>", "STOP"])
        var output = ""
        output += filter.consume("hello<")
        output += filter.consume("EN")
        output += filter.consume("D>ignored")
        output += filter.finish()

        XCTAssertEqual(output, "hello")
        XCTAssertTrue(filter.stopped)
    }

    func testFlushesUnmatchedPrefixAtEndOfStream() {
        var filter = FoundationStopSequenceFilter(stopSequences: ["<END>"])
        XCTAssertEqual(filter.consume("hello<EN"), "hello")
        XCTAssertEqual(filter.finish(), "<EN")
    }

    func testAcceptedTranscriptExcludesHiddenPostStopOutput() throws {
        let original = Transcript(entries: [
            .instructions(.init(
                segments: [.text(.init(content: "Be concise"))],
                toolDefinitions: []
            )),
            .prompt(.init(segments: [.text(.init(content: "Earlier question"))])),
            .response(.init(
                assetIDs: [],
                segments: [.text(.init(content: "Earlier answer"))]
            ))
        ])

        let accepted = FoundationModelService.acceptedTranscript(
            from: original,
            prompt: "User: continue\n\nAssistant: ",
            response: "visible prefix",
            options: GenerationOptions()
        )

        XCTAssertEqual(accepted.count, original.count + 2)
        guard case .response(let response) = accepted[accepted.index(before: accepted.endIndex)] else {
            return XCTFail("Expected the replacement transcript to end with a response")
        }
        let text = response.segments.compactMap { segment -> String? in
            guard case .text(let value) = segment else { return nil }
            return value.content
        }.joined()
        XCTAssertEqual(text, "visible prefix")
    }
}

private actor FoundationOperationProbe {
    private var active = 0
    private var maximumActive = 0
    private var events: [String] = []

    func start(_ name: String) {
        active += 1
        maximumActive = max(maximumActive, active)
        events.append("start:\(name)")
    }

    func finish(_ name: String) {
        events.append("finish:\(name)")
        active -= 1
    }

    func snapshot() -> (events: [String], maximumActive: Int) {
        (events, maximumActive)
    }
}

final class FoundationSessionOperationGateTests: XCTestCase {
    func testEarlierReservationBlocksLaterResetAndExcludesConcurrentUse() async throws {
        let gate = FoundationSessionOperationGate()
        let streamReservation = gate.reserve()
        let resetReservation = gate.reserve()
        let probe = FoundationOperationProbe()

        let resetTask = Task {
            try await resetReservation.perform {
                await probe.start("reset")
                await probe.finish("reset")
            }
        }
        for _ in 0..<10 { await Task.yield() }
        let beforeStreamStarts = await probe.snapshot()
        XCTAssertTrue(beforeStreamStarts.events.isEmpty)

        let streamTask = Task {
            try await streamReservation.perform {
                await probe.start("stream")
                try await Task.sleep(for: .milliseconds(20))
                await probe.finish("stream")
            }
        }

        try await streamTask.value
        try await resetTask.value
        let snapshot = await probe.snapshot()
        XCTAssertEqual(snapshot.maximumActive, 1)
        XCTAssertEqual(snapshot.events, [
            "start:stream", "finish:stream", "start:reset", "finish:reset"
        ])
    }
}
#endif
