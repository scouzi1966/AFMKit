import AFMKitCore
import AFMOpenAICompat
import Foundation
@testable import AFMKitMLX
import Testing

struct Qwen38ToolCallStreamingTests {
    @Test("Native Qwen template keeps the model template with the tool JSON serialization shim")
    func nativeQwenUsesModelOwnedToolTemplateSerialization() {
        #expect(MLXModelService.usesModelOwnedToolTemplate(parser: "qwen3_xml"))
        #expect(MLXModelService.usesModelOwnedToolTemplate(parser: nil))
        #expect(!MLXModelService.usesModelOwnedToolTemplate(parser: "afm_adaptive_xml"))

        let template = "{% for tool in tools %}{{- tool | tojson }}{% endfor %}"
        let patched = MLXModelService.patchNativeTemplateForPythonToolJSON(template)
        #expect(patched.contains("tool.__python_json__"))
        #expect(!patched.contains("tool | tojson"))
    }

    @Test("Auto-detected Qwen XML uses the native parser without compatibility mode")
    func autoDetectedXMLUsesNativeParser() {
        #expect(MLXModelService.effectiveToolCallParser(
            configuredParser: nil,
            detectedFormat: .xmlFunction
        ) == "qwen3_xml")
        #expect(MLXModelService.effectiveChatTemplateToolCallParser(
            configuredParser: nil,
            detectedFormat: .xmlFunction
        ) == "qwen3_xml")
    }

    @Test("Native Qwen parser does not reinterpret compatibility JSON")
    func nativeParserRejectsCompatibilityJSON() {
        let text = #"<tool_call>{"name":"get_weather","arguments":{"location":"Paris"}}</tool_call>"#
        let (nativeCalls, nativeRemaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: text,
            toolCallParser: "qwen3_xml",
            tools: [weatherTool]
        )
        let (adaptiveCalls, _) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: text,
            toolCallParser: "afm_adaptive_xml",
            tools: [weatherTool]
        )

        #expect(nativeCalls.isEmpty)
        #expect(nativeRemaining == text)
        #expect(adaptiveCalls.count == 1)
    }

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

    @Test("Qwen 3.8 native XML preserves distinct adjacent calls through translation")
    func nativeXMLPreservesDistinctAdjacentCallsThroughTranslation() throws {
        let runtime = makeRuntime(
            tools: [weatherTool, timeTool],
            parser: "qwen3_xml"
        )
        let output = runtime.process(piece: #"<tool_call><function=get_weather><parameter=location>London</parameter><parameter=days>1</parameter></function></tool_call><tool_call><function=get_time><parameter=timezone>Asia/Tokyo</parameter></function></tool_call>"#)
        let chunks = BatchScheduler.streamChunksToEmit(from: output.events)
        var translator = MLXStreamEventTranslator(
            thinkStartTag: nil,
            thinkEndTag: nil,
            maximumResponseTokens: 100,
            tools: [weatherTool, timeTool]
        )
        var translated = chunks.flatMap { translator.consume($0) }
        translated += translator.finish()

        let completed = translated.compactMap { event -> AFMToolCall? in
            guard case .toolCall(let call, .completed) = event else { return nil }
            return call
        }
        #expect(completed.map(\.name) == ["get_weather", "get_time"])
        #expect(try decodeArguments(completed[0].arguments)["location"] as? String == "London")
        #expect(try decodeArguments(completed[1].arguments)["timezone"] as? String == "Asia/Tokyo")
    }

    @Test("Qwen 3.8 native XML preserves distinct calls across token fragments")
    func nativeXMLPreservesDistinctCallsAcrossFragments() throws {
        let runtime = makeRuntime(tools: [weatherTool, timeTool], parser: "qwen3_xml")
        let pieces = [
            "<tool_call><function=get_wea", "ther><parameter=location>Lon",
            "don</parameter><parameter=days>1</parameter></function></tool_call>",
            "<tool_call><function=get_time><parameter=timezone>Asia/",
            "Tokyo</parameter></function></tool_call>",
        ]
        let events = pieces.flatMap { runtime.process(piece: $0).events }
        let completed = events.compactMap { event -> ResponseToolCall? in
            guard case .replaceCollected(_, let call) = event else { return nil }
            return call
        }

        #expect(completed.map(\.function.name) == ["get_weather", "get_time"])
        #expect(completed.map(\.index) == [0, 1])
        #expect(try decodeArguments(completed[0].function.arguments)["location"] as? String == "London")
        #expect(try decodeArguments(completed[1].function.arguments)["timezone"] as? String == "Asia/Tokyo")
    }

    @Test("Native Qwen coercion does not fabricate omitted required arguments")
    func nativeCoercionDoesNotFabricateRequiredArguments() throws {
        let tool = makeTool(
            name: "set_alarm",
            properties: [
                "hour": ["type": "integer"],
                "enabled": ["type": "boolean"],
            ],
            required: ["hour", "enabled"]
        )
        let raw = ResponseToolCall(
            index: 0,
            id: "call_alarm",
            type: "function",
            function: .init(name: "set_alarm", arguments: #"{"hour":"7"}"#)
        )

        let native = MLXModelService.coerceArgumentTypes(
            raw,
            tools: [tool],
            repairArguments: false
        )
        let repaired = MLXModelService.coerceArgumentTypes(
            raw,
            tools: [tool],
            repairArguments: true
        )

        let nativeArguments = try decodeArguments(native.function.arguments)
        #expect(nativeArguments["hour"] as? Int == 7)
        #expect(nativeArguments["enabled"] == nil)
        #expect(try decodeArguments(repaired.function.arguments)["enabled"] as? Bool == false)
    }

    @Test("Native Qwen parser preserves model-emitted argument names")
    func nativeParserDoesNotRemapArgumentNames() throws {
        let tool = makeTool(
            name: "schedule_event",
            properties: ["startDate": ["type": "string"]],
            required: ["startDate"]
        )
        let text = #"<tool_call><function=schedule_event><parameter=start_date>2026-08-25</parameter></function></tool_call>"#

        let native = makeRuntime(tools: [tool], parser: "qwen3_xml")
        let nativeCalls = native.process(piece: text).events.compactMap { event -> ResponseToolCall? in
            guard case .appendCollected(let call) = event else { return nil }
            return call
        }
        #expect(try decodeArguments(nativeCalls[0].function.arguments)["start_date"] as? String == "2026-08-25")
        #expect(try decodeArguments(nativeCalls[0].function.arguments)["startDate"] == nil)

        let adaptive = makeRuntime(tools: [tool], parser: "afm_adaptive_xml")
        let repairedCalls = adaptive.process(piece: text).events.compactMap { event -> ResponseToolCall? in
            guard case .appendCollected(let call) = event else { return nil }
            return call
        }
        #expect(try decodeArguments(repairedCalls[0].function.arguments)["startDate"] as? String == "2026-08-25")
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

    private var timeTool: RequestTool {
        makeTool(
            name: "get_time",
            properties: ["timezone": ["type": "string"]],
            required: ["timezone"]
        )
    }

    private func makeRuntime(
        tools: [RequestTool],
        parser: String = "afm_adaptive_xml"
    ) -> ToolCallStreamingRuntime {
        ToolCallStreamingRuntime(
            toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>",
            toolCallParser: parser,
            tools: tools,
            repairToolArguments: false,
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
