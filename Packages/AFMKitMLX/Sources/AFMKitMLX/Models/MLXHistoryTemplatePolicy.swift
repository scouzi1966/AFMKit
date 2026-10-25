import Foundation

enum MLXHistoryTemplatePolicy {
    /// Match the tokenizer loader's string-template precedence. Read once at model
    /// load, not on the decode or request path. Overrides are selected by the caller.
    static func nativeTemplate(directory: URL) -> String? {
        let config = directory.appendingPathComponent("tokenizer_config.json")
        if let data = try? Data(contentsOf: config),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let template = json["chat_template"] as? String {
            return template
        }
        return try? String(contentsOf: directory.appendingPathComponent("chat_template.jinja"),
            encoding: .utf8)
    }

    static func applyingDefaults(_ kwargs: [String: any Sendable], template: String?) -> [String: any Sendable] {
        // This controls previous-turn history, not whether the new answer thinks.
        // Only templates exposing this option receive it; explicit server/request
        // choices retain priority. Current-turn reasoning remains template-owned.
        guard template?.contains("preserve_thinking") == true,
              kwargs["preserve_thinking"] == nil else { return kwargs }
        var result = kwargs
        result["preserve_thinking"] = false
        return result
    }
}
