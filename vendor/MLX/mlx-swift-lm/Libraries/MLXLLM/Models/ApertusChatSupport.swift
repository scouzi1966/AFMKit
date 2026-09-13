import Foundation
import MLXLMCommon
import Tokenizers

/// The model-owned Apertus template renders flat function definitions, not
/// OpenAI's {type: function, function: {...}} envelopes. Keep this adaptation
/// independent of the selected output parser (including raw mode).
enum ApertusChatSupport {
    static func tools(_ tools: [ToolSpec]?) -> [ToolSpec]? {
        tools?.map { tool in
            var function = (tool["function"] as? ToolSpec) ?? tool
            if function["description"] == nil { function["description"] = "" }
            return function
        }
    }
}
