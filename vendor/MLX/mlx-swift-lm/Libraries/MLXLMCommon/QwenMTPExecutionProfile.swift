import Foundation

/// Process-start Qwen Next tuning. This does not mutate the process environment.
/// The profile is opt-in; explicit individual settings always take precedence.
/// Call validate before model loading so a misspelled profile fails clearly.
public enum QwenMTPExecutionProfile {
    public static let variable = "AFM_QWEN_MTP_PROFILE"
    public static let throughputV1 = "throughput-v1"
    public static let throughputV2 = "throughput-v2"

    public enum ProfileError: LocalizedError {
        case unknown(String)

        public var errorDescription: String? {
            switch self {
            case .unknown(let value):
                return "Unknown Qwen MTP profile '\(value)'; use throughput-v1, throughput-v2 or off."
            }
        }
    }

    private static func name(in environment: [String: String]) -> String {
        (environment[variable] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
    }

    public static func validate(environment: [String: String]) throws {
        let value = name(in: environment)
        guard value.isEmpty || value == "off" || value == throughputV1 || value == throughputV2 else {
            throw ProfileError.unknown(value)
        }
    }

    /// Pure expansion also permits tests and runtime policy construction to
    /// use explicit dictionaries without changing global environment values.
    public static func resolved(environment: [String: String]) -> [String: String] {
        let explicit = environment.filter { $0.key.hasPrefix("AFM_QWEN_") }
        let profile = name(in: environment)
        guard profile == throughputV1 || profile == throughputV2 else { return explicit }
        var defaults = throughputDefaults
        if profile == throughputV2 {
            // The October 4 corrected-HC recipe. Keep v1 and normal defaults
            // unchanged; callers explicitly select this numerical policy.
            defaults["AFM_QWEN_FUSED_QUANTIZED_HC"] = "1"
            defaults["AFM_QWEN_VERIFY_SPARSE_ATTENTION"] = "1"
            defaults["AFM_QWEN_VERIFY_ASYNC_LADDER"] = "2"
        }
        return defaults.merging(explicit) { _, override in override }
    }

    /// Preserve the existing lookup timing of individual tuning switches.
    /// Kernel owners still capture their settings at initialization; no model
    /// instance calls setenv or changes another instance's environment.
    public static var environment: [String: String] {
        resolved(environment: ProcessInfo.processInfo.environment)
    }

    private static let throughputDefaults: [String: String] = [
        "AFM_QWEN_EXPERT_DOWN_GROUP64_REUSE": "1",
        "AFM_QWEN_HC_NATIVE_CHAIN": "1",
        "AFM_QWEN_MTP_CACHE_ONLY_REPAIR": "1",
        "AFM_QWEN_MTP_DRAFT_ASYNC_LADDER": "1",
        "AFM_QWEN_MTP_DRAFT_SHORTLIST": "1",
        "AFM_QWEN_MTP_INDEPENDENT_ATTENTION": "1",
        "AFM_QWEN_MTP_ONE_PASS_CAPTURE": "1",
        "AFM_QWEN_MTP_REPLAY_BACKOFF": "31",
        "AFM_QWEN_MTP_REPLAY_BACKOFF_ON_MISS": "1",
        "AFM_QWEN_MTP_REPLAY_MIB": "4096",
        "AFM_QWEN_MTP_SCHEDULER": "1",
        "AFM_QWEN_MTP_SHARED_HEAD": "1",
        "AFM_QWEN_MTP_SHARED_VERIFY": "1",
        "AFM_QWEN_MTP_SUBMISSION_WINDOW": "8",
        "AFM_QWEN_MTP_VERIFICATION_POLICY": "batched",
        "AFM_QWEN_RESIDENT_CPU_NGRAM": "1",
        "AFM_QWEN_VERIFY_ASYNC_LADDER": "8",
        "AFM_QWEN_VERIFY_ATTENTION_CHUNK": "2",
        "AFM_QWEN_VERIFY_FUSED_HC": "auto",
        "AFM_QWEN_VERIFY_FUSED_ROUTER": "1",
        "AFM_QWEN_VERIFY_GROUP64_EXPERT_ROWS": "1",
        "AFM_QWEN_VERIFY_QMM": "1",
        "AFM_QWEN_VERIFY_SHARED_ASYNC_LADDER": "4",
    ]
}
