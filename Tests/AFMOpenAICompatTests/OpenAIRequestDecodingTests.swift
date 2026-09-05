import AFMOpenAICompat
import XCTest

final class OpenAIRequestDecodingTests: XCTestCase {
    private func decode(_ json: String) throws -> ChatCompletionRequest {
        try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
    }

    func testBareStopString() throws {
        XCTAssertEqual(try decode(#"{"messages":[],"stop":"END"}"#).stop, ["END"])
    }

    func testStopArrayAndChoiceCount() throws {
        let request = try decode(#"{"messages":[],"stop":["END","STOP"],"n":2}"#)
        XCTAssertEqual(request.stop, ["END", "STOP"])
        XCTAssertEqual(request.n, 2)
    }

    func testNullAndMissingStop() throws {
        XCTAssertNil(try decode(#"{"messages":[],"stop":null}"#).stop)
        XCTAssertNil(try decode(#"{"messages":[]}"#).stop)
    }

    func testMalformedStopIsRejected() {
        for value in ["42", "true", "{}", "[1]", "[\"END\",null]"] {
            XCTAssertThrowsError(try decode("{\"messages\":[],\"stop\":\(value)}"))
        }
    }

    func testEveryCurrentFieldSurvivesDecodeAndEncode() throws {
        let json = #"{"model":"test","messages":[{"role":"user","content":"hi"}],"temperature":0.2,"max_tokens":123,"max_completion_tokens":124,"top_p":0.9,"repetition_penalty":1.1,"repeat_penalty":1.2,"frequency_penalty":0.3,"presence_penalty":0.4,"top_k":5,"min_p":0.1,"seed":42,"logprobs":true,"top_logprobs":3,"n":1,"stop":["END"],"stream":true,"stream_options":{"include_usage":true,"continuous_usage_stats":true},"ignore_eos":true,"user":"client","tools":[],"tool_choice":"auto","parallel_tool_calls":false,"response_format":{"type":"json_object"},"chat_template_kwargs":{"enable_thinking":false},"reasoning_effort":"low"}"#
        let request = try decode(json)
        XCTAssertEqual(request.ignoreEOS, true)
        let original = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8)) as? NSDictionary)
        let encoded = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? NSDictionary)
        XCTAssertEqual(original, encoded)
    }
}
