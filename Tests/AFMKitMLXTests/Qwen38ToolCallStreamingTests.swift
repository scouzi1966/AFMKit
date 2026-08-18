import AFMKitCore
import AFMOpenAICompat
import Foundation
@testable import AFMKitMLX
import Testing

struct Qwen38ToolCallStreamingTests {
    @Test("Qwen 3.8 JSON-in-XML preserves nested Unicode arguments")
    func nativeJSONInXMLPreservesComplexArguments() throws {
        let text = #"""
        <tool_call>
        {"name":"create_event","arguments":{"title":"Café 🚀","attendees":["Zoë","Miyuki"],"metadata":{"priority":2,"remote":true},"note":null}}
        </tool_call>
        """#

        let (calls, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: text,
            toolCallParser: "afm_adaptive_xml",
            tools: [eventTool]
        )

        let call = try #require(calls.first)
        #expect(calls.count == 1)
        #expect(call.function.name == "create_event")
        #expect(call.function.arguments["title"]?.anyValue as? String == "Café 🚀")
        #expect(call.function.arguments["attendees"]?.anyValue as? [Any] != nil)
        #expect(call.function.arguments["metadata"]?.anyValue as? [String: Any] != nil)
        #expect(call.function.arguments["note"] == .null)
        #expect(remaining.isEmpty)
    }

    @Test("Qwen 3.8 streaming assembles a tool call split at arbitrary boundaries")
    func streamingAssemblesFragmentedJSONInXML() throws {
        let runtime = makeRuntime(tools: [weatherTool])
        let pieces = [
            "<tool", "_call>\n{\"na", "me\":\"get_weather\",",
            "\"arguments\":{\"location\":\"Montr", "éal\",\"days\":3}}\n",
            "</tool_", "call>",
        ]

        var events: [ToolCallStreamingEvent] = []
        for piece in pieces {
            events.append(contentsOf: runtime.process(piece: piece).events)
        }

        let call = try #require(collectedCall(from: events))
        let arguments = try decodeArguments(call.function.arguments)
        #expect(call.function.name == "get_weather")
        #expect(arguments["location"] as? String == "Montréal")
        #expect(arguments["days"] as? Int == 3)
    }

    @Test("Qwen 3.8 adaptive mode repairs malformed name-equals output")
    func adaptiveModeRepairsMalformedOutput() throws {
        let text = #"<tool_call>{"name="get_weather", "arguments":{"location":"Toronto","days":2}}</tool_call>"#

        let (strictCalls, strictRemaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: text,
            toolCallParser: nil,
            tools: [weatherTool]
        )
        #expect(strictCalls.isEmpty)
        #expect(strictRemaining.contains("<tool_call>"))

        let (adaptiveCalls, adaptiveRemaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: text,
            toolCallParser: "afm_adaptive_xml",
            tools: [weatherTool]
        )
        let call = try #require(adaptiveCalls.first)
        #expect(call.function.name == "get_weather")
        #expect(call.function.arguments["location"]?.anyValue as? String == "Toronto")
        #expect(call.function.arguments["days"]?.anyValue as? Int == 2)
        #expect(adaptiveRemaining.isEmpty)
    }

    @Test("Qwen 3.8 incomplete stream salvages typed trailing parameters")
    func incompleteStreamIsSalvaged() throws {
        let runtime = makeRuntime(tools: [weatherTool])
        var events = runtime.process(piece: "<tool_call>").events
        events += runtime.process(piece: "<function=get_weather>").events
        events += runtime.process(piece: "<parameter=location>Toronto</parameter>").events
        events += runtime.process(piece: "<parameter=days>3").events
        events += runtime.finishIncompleteToolCall()

        let call = try #require(replacementCall(from: events))
        let arguments = try decodeArguments(call.function.arguments)
        #expect(call.function.name == "get_weather")
        #expect(arguments["location"] as? String == "Toronto")
        #expect(arguments["days"] as? Int == 3)

        let streamed = try decodeArguments(streamedArguments(from: events))
        #expect(streamed["location"] as? String == "Toronto")
        #expect(streamed["days"] as? Int == 3)
        #expect(NSDictionary(dictionary: streamed).isEqual(to: arguments))
    }

    @Test("Qwen 3.8 streaming preserves adjacent parallel calls")
    func streamingPreservesAdjacentParallelCalls() throws {
        let runtime = makeRuntime(tools: [weatherTool])
        let output = runtime.process(piece: #"<tool_call>{"name":"get_weather","arguments":{"location":"Toronto","days":1}}</tool_call><tool_call>{"name":"get_weather","arguments":{"location":"Vancouver","days":2}}</tool_call>"#)
        let calls = collectedCalls(from: output.events)

        #expect(calls.count == 2)
        #expect(try decodeArguments(calls[0].function.arguments)["location"] as? String == "Toronto")
        #expect(try decodeArguments(calls[1].function.arguments)["location"] as? String == "Vancouver")
        #expect(output.passthroughText == nil)
    }

    private var weatherTool: RequestTool {
        makeTool(
            name: "get_weather",
            properties: [
                "location": ["type": "string"],
                "days": ["type": "integer"],
            ],
            required: ["location"]
        )
    }

    private var eventTool: RequestTool {
        makeTool(
            name: "create_event",
            properties: [
                "title": ["type": "string"],
                "attendees": ["type": "array", "items": ["type": "string"]],
                "metadata": ["type": "object"],
                "note": ["type": ["string", "null"]],
            ],
            required: ["title"]
        )
    }

    private func makeRuntime(tools: [RequestTool]) -> ToolCallStreamingRuntime {
        ToolCallStreamingRuntime(
            toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>",
            toolCallParser: "afm_adaptive_xml",
            tools: tools,
            applyFixToolArgs: { $0 },
            remapSingleKey: { key, _ in key }
        )
    }

    private func makeTool(
        name: String,
        properties: [String: Any],
        required: [String]
    ) -> RequestTool {
        let schema: [String: Any] = [
            "type": "object",
            "properties": properties,
            "required": required,
        ]
        let data = try! JSONSerialization.data(withJSONObject: schema)
        return RequestTool(
            type: "function",
            function: .init(
                name: name,
                description: nil,
                parameters: try! JSONDecoder().decode(AnyCodable.self, from: data),
                strict: nil
            )
        )
    }

    private func collectedCall(from events: [ToolCallStreamingEvent]) -> ResponseToolCall? {
        for event in events.reversed() {
            switch event {
            case .replaceCollected(_, let call), .appendCollected(let call):
                return call
            default:
                continue
            }
        }
        return nil
    }

    private func collectedCalls(from events: [ToolCallStreamingEvent]) -> [ResponseToolCall] {
        events.compactMap { event in
            if case .appendCollected(let call) = event, !call.function.arguments.isEmpty {
                return call
            }
            return nil
        }
    }

    private func replacementCall(from events: [ToolCallStreamingEvent]) -> ResponseToolCall? {
        for event in events.reversed() {
            if case .replaceCollected(_, let call) = event {
                return call
            }
        }
        return nil
    }

    private func streamedArguments(from events: [ToolCallStreamingEvent]) -> String {
        events.compactMap { event in
            guard case .delta(let delta) = event else { return nil }
            return delta.function?.arguments
        }.joined()
    }

    private func decodeArguments(_ arguments: String) throws -> [String: Any] {
        try #require(
            JSONSerialization.jsonObject(with: Data(arguments.utf8)) as? [String: Any]
        )
    }
}
