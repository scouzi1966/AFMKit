import XCTest
@testable import MLXLLM
import MLXLMCommon
import Tokenizers

final class ApertusToolTests: XCTestCase {
    func testEOSConsumedSuffixFinalizesOnlyCompleteArrays() {
        let valid = "<|tools_prefix|>[{\"weather\":{}},{\"clock\":{}}]"
        let processor = ToolCallProcessor(format: .apertus)
        XCTAssertNil(processor.processChunk(valid))
        XCTAssertTrue(processor.toolCalls.isEmpty)
        XCTAssertNil(processor.finishPendingText())
        XCTAssertEqual(processor.toolCalls.map(\.function.name), ["weather", "clock"])
        for invalid in [String(valid.dropLast()), "<|tools_prefix|>[{\"weather\":{}},{\"bad\":null}]"] {
            let partial = ToolCallProcessor(format: .apertus)
            XCTAssertNil(partial.processChunk(invalid))
            XCTAssertEqual(partial.finishPendingText(), invalid)
            XCTAssertTrue(partial.toolCalls.isEmpty)
        }
    }

    func testNativeDetectionAndMultipleCallsAcrossEveryChunkBoundary() {
        XCTAssertEqual(ToolCallFormat.infer(from: "apertus"), .apertus)
        let text = "before<|tools_prefix|>[{\"weather\":{\"city\":\"Zürich\"}},{\"clock\":{}}]<|tools_suffix|>after"
        let characters = Array(text)
        for split in 0...characters.count {
            let processor = ToolCallProcessor(format: .apertus)
            let visible = (processor.processChunk(String(characters.prefix(split))) ?? "")
                + (processor.processChunk(String(characters.dropFirst(split))) ?? "")
            XCTAssertEqual(visible, "beforeafter", "split \(split)")
            XCTAssertEqual(processor.toolCalls.map(\.function.name), ["weather", "clock"])
        }
    }

    func testMalformedEnvelopeDoesNotPartiallyExecuteCalls() {
        let processor = ToolCallProcessor(format: .apertus)
        let text = "<|tools_prefix|>[{\"valid\":{}},{\"bad\":null}]<|tools_suffix|>"
        XCTAssertEqual(processor.processChunk(text), text)
        XCTAssertTrue(processor.toolCalls.isEmpty)
    }

    func testToolSchemaAdapterUnwrapsOpenAIEnvelope() {
        let definitions = ApertusChatSupport.tools([
            ["type": "function", "function": ["name": "weather"] as ToolSpec]
        ])
        XCTAssertEqual(definitions?.first?["name"] as? String, "weather")
        XCTAssertEqual(definitions?.first?["description"] as? String, "")
        XCTAssertNil(definitions?.first?["function"])
    }
}
