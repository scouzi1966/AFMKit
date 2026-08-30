import AFMOpenAICompat
import XCTest

final class OpenAIRequestDecodingTests: XCTestCase {
    func testChatRequestDecodesBareStopString() throws {
        let request = try decode(#"{"messages":[{"role":"user","content":"hi"}],"stop":"END"}"#)

        XCTAssertEqual(request.stop, ["END"])
    }

    func testChatRequestDecodesStopArrayAndChoiceCount() throws {
        let request = try decode(#"{"messages":[{"role":"user","content":"hi"}],"n":2,"stop":["END","STOP"]}"#)

        XCTAssertEqual(request.n, 2)
        XCTAssertEqual(request.stop, ["END", "STOP"])
    }

    private func decode(_ json: String) throws -> ChatCompletionRequest {
        try JSONDecoder().decode(ChatCompletionRequest.self, from: XCTUnwrap(json.data(using: .utf8)))
    }
}
