import Foundation
import AFMKitCore

public enum AFMMLXSpeculativeDecodingMode: String, Codable, CaseIterable, Identifiable, Hashable, Sendable {
    case off
    case auto
    case mtp
    case eagle3

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .off: return "Off"
        case .auto: return "Auto"
        case .mtp: return "MTP"
        case .eagle3: return "EAGLE3"
        }
    }
}

public struct AFMMLXSpeculativeModeAvailability: Equatable, Sendable {
    public let mode: AFMMLXSpeculativeDecodingMode
    public let isSelectable: Bool
    public let reason: String

    public init(
        mode: AFMMLXSpeculativeDecodingMode,
        isSelectable: Bool,
        reason: String
    ) {
        self.mode = mode
        self.isSelectable = isSelectable
        self.reason = reason
    }

    public static func evaluate(
        modelLoaded: Bool,
        mtpCompatible: Bool,
        denseGemma4Verifier: Bool
    ) -> [AFMMLXSpeculativeDecodingMode: AFMMLXSpeculativeModeAvailability] {
        let hasAccelerationPath = mtpCompatible || denseGemma4Verifier

        return [
            .off: AFMMLXSpeculativeModeAvailability(
                mode: .off,
                isSelectable: true,
                reason: "Use standard MLX generation."
            ),
            .auto: AFMMLXSpeculativeModeAvailability(
                mode: .auto,
                isSelectable: modelLoaded && hasAccelerationPath,
                reason: modelLoaded
                    ? "Auto requires a loaded model with MTP or dense Gemma4 EAGLE3 support."
                    : "Load a supported MLX model before enabling acceleration."
            ),
            .mtp: AFMMLXSpeculativeModeAvailability(
                mode: .mtp,
                isSelectable: modelLoaded && mtpCompatible,
                reason: modelLoaded
                    ? "MTP requires a compatible loaded model with an embedded or sidecar head."
                    : "Load a model with an embedded or sidecar MTP head before selecting MTP."
            ),
            .eagle3: AFMMLXSpeculativeModeAvailability(
                mode: .eagle3,
                isSelectable: modelLoaded && denseGemma4Verifier,
                reason: modelLoaded
                    ? "EAGLE3 requires a loaded dense Gemma4 verifier model."
                    : "Load a dense Gemma4 model before selecting EAGLE3."
            )
        ]
    }

    public static let unloaded = evaluate(
        modelLoaded: false,
        mtpCompatible: false,
        denseGemma4Verifier: false
    )

    public static func pendingSelection(
        mtpCompatible: Bool,
        denseGemma4Verifier: Bool
    ) -> [AFMMLXSpeculativeDecodingMode: AFMMLXSpeculativeModeAvailability] {
        let hasAccelerationPath = mtpCompatible || denseGemma4Verifier

        return [
            .off: AFMMLXSpeculativeModeAvailability(
                mode: .off,
                isSelectable: true,
                reason: "Use standard MLX generation."
            ),
            .auto: AFMMLXSpeculativeModeAvailability(
                mode: .auto,
                isSelectable: hasAccelerationPath,
                reason: hasAccelerationPath
                    ? "Use the selected model's acceleration path after loading."
                    : "Select a model with MTP or dense Gemma4 EAGLE3 support."
            ),
            .mtp: AFMMLXSpeculativeModeAvailability(
                mode: .mtp,
                isSelectable: mtpCompatible,
                reason: mtpCompatible
                    ? "Use MTP after loading the selected model."
                    : "Select a model with an embedded or sidecar MTP head before selecting MTP."
            ),
            .eagle3: AFMMLXSpeculativeModeAvailability(
                mode: .eagle3,
                isSelectable: denseGemma4Verifier,
                reason: denseGemma4Verifier
                    ? "Use EAGLE3 after loading the selected dense Gemma4 model."
                    : "Select a dense Gemma4 model before selecting EAGLE3."
            )
        ]
    }
}

public struct AFMMLXSpeculativeModelCompatibility: Equatable, Sendable {
    public let mtpCompatible: Bool
    public let denseGemma4Verifier: Bool

    public init(mtpCompatible: Bool, denseGemma4Verifier: Bool) {
        self.mtpCompatible = mtpCompatible
        self.denseGemma4Verifier = denseGemma4Verifier
    }

    public static let unavailable = AFMMLXSpeculativeModelCompatibility(
        mtpCompatible: false,
        denseGemma4Verifier: false
    )

    public static func evaluate(
        config: [String: Any],
        hasMTPSidecar: Bool
    ) -> AFMMLXSpeculativeModelCompatibility {
        evaluate(
            config: config,
            hasMTPSidecar: hasMTPSidecar,
            embeddedAssetsPresent: false)
    }

    static func evaluate(
        config: [String: Any],
        hasMTPSidecar: Bool,
        embeddedAssetsPresent: Bool
    ) -> AFMMLXSpeculativeModelCompatibility {
        AFMMLXSpeculativeModelCompatibility(
            mtpCompatible: (hasMTPSidecar && isMTPCompatibleConfiguration(config))
                || (embeddedAssetsPresent && hasEmbeddedMTPConfiguration(config)),
            denseGemma4Verifier: isDenseGemma4VerifierConfiguration(config)
        )
    }

    public static func evaluate(modelDirectory: URL) -> AFMMLXSpeculativeModelCompatibility {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return .unavailable
        }

        let hasMTPSidecar = FileManager.default.fileExists(
            atPath: modelDirectory.appendingPathComponent("mtp.safetensors").path
        )
        let hasEmbeddedAssets = hasCompleteEmbeddedGLMMTP(
            modelDirectory: modelDirectory, config: config)
            || hasCompleteEmbeddedQwenNextMTP(
                modelDirectory: modelDirectory, config: config)
        return evaluate(
            config: config,
            hasMTPSidecar: hasMTPSidecar,
            embeddedAssetsPresent: hasEmbeddedAssets)
    }

    private static func isMTPCompatibleConfiguration(_ config: [String: Any]) -> Bool {
        let topLevelType = AFMMLXModelArchitecture.canonicalModelType(config["model_type"] as? String ?? "")
        let textConfig = config["text_config"] as? [String: Any]
        let textType = AFMMLXModelArchitecture.canonicalModelType(textConfig?["model_type"] as? String ?? "")
        let architecture = ((config["architectures"] as? [String]) ?? []).joined(separator: " ").lowercased()

        return topLevelType.hasPrefix("qwen3_5")
            || topLevelType.hasPrefix("qwen3_6")
            || topLevelType == "qwen4_exp"
            || textType.hasPrefix("qwen3_5")
            || textType.hasPrefix("qwen3_6")
            || textType == "qwen4_exp_text"
            || architecture.contains("qwen3_5")
            || architecture.contains("qwen3_6")
            || architecture.contains("qwen4exp")
            || architecture.contains("qwen3.5")
            || architecture.contains("qwen3.6")
    }

    private static func hasEmbeddedGLMMTPConfiguration(_ config: [String: Any]) -> Bool {
        let topLevelType = AFMMLXModelArchitecture.canonicalModelType(
            config["model_type"] as? String ?? "")
        let text = config["text_config"] as? [String: Any] ?? config
        let count = (text["num_nextn_predict_layers"] as? NSNumber)?.intValue ?? 0
        return topLevelType == "glm5_next" && count == 1
    }

    private static func hasEmbeddedMTPConfiguration(_ config: [String: Any]) -> Bool {
        if hasEmbeddedGLMMTPConfiguration(config) { return true }
        let topLevelType = AFMMLXModelArchitecture.canonicalModelType(
            config["model_type"] as? String ?? "")
        let text = config["text_config"] as? [String: Any] ?? config
        let textType = AFMMLXModelArchitecture.canonicalModelType(
            text["model_type"] as? String ?? "")
        let mtp = text["mtp"] as? [String: Any]
        let layers = (mtp?["num_hidden_layers"] as? NSNumber)?.intValue ?? 0
        return (topLevelType == "qwen4_exp" || textType == "qwen4_exp_text")
            && layers == 1
    }

    private static func hasCompleteEmbeddedQwenNextMTP(
        modelDirectory: URL,
        config: [String: Any]
    ) -> Bool {
        guard hasEmbeddedMTPConfiguration(config) else { return false }
        let prefixes = ["language_model.mtp.", "mtp."]
        guard let expected = embeddedQwenNextMTPExpectedTensors(config: config) else {
            return false
        }
        let requiredSuffixes = expected.keys

        let indexURL = modelDirectory.appendingPathComponent(
            "model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            guard let data = boundedData(
                at: indexURL, maximumBytes: 128 * 1_024 * 1_024),
                let object = try? JSONSerialization.jsonObject(with: data)
                    as? [String: Any],
                let map = object["weight_map"] as? [String: String],
                let prefix = prefixes.first(where: { candidate in
                    requiredSuffixes.allSatisfy { map[candidate + $0] != nil }
                })
            else { return false }

            for suffix in requiredSuffixes {
                let key = prefix + suffix
                guard let shard = map[key],
                      let shardURL = containedShardURL(
                        named: shard, modelDirectory: modelDirectory),
                      let header = safeTensorHeader(at: shardURL),
                      hasSaneOffsets(header),
                      header.tensors.contains(where: {
                          $0.name == key && $0.dtype == expected[suffix]?.0
                              && $0.shape == expected[suffix]?.1
                      })
                else { return false }
            }
            return true
        }

        guard let tensorURL = containedShardURL(
            named: "model.safetensors", modelDirectory: modelDirectory),
            let header = safeTensorHeader(at: tensorURL),
            hasSaneOffsets(header)
        else { return false }
        let tensors = Dictionary(uniqueKeysWithValues: header.tensors.map { ($0.name, $0) })
        return prefixes.contains { prefix in
            requiredSuffixes.allSatisfy { suffix in
                guard let tensor = tensors[prefix + suffix], let requirement = expected[suffix]
                else { return false }
                return tensor.dtype == requirement.0 && tensor.shape == requirement.1
            }
        }
    }

    static let embeddedQwenNextMTPRequiredSuffixes: [String] = {
        let quantizedBases = [
            "fc_embedding", "fc_hidden",
            "hyper_connection_mixer.input_mix_weight_down",
            "hyper_connection_mixer.input_mix_weight_up",
            "layers.0.attn_hyper_connection.input_mix_weight_down",
            "layers.0.attn_hyper_connection.input_mix_weight_up",
            "layers.0.mlp.gate", "layers.0.mlp.shared_expert.down_proj",
            "layers.0.mlp.shared_expert.gate_proj",
            "layers.0.mlp.shared_expert.up_proj",
            "layers.0.mlp.switch_mlp.down_proj",
            "layers.0.mlp.switch_mlp.gate_proj",
            "layers.0.mlp.switch_mlp.up_proj",
            "layers.0.mlp_hyper_connection.input_mix_weight_down",
            "layers.0.mlp_hyper_connection.input_mix_weight_up",
            "layers.0.self_attn.indexer.index_qk_proj",
            "layers.0.self_attn.k_proj", "layers.0.self_attn.o_proj",
            "layers.0.self_attn.q_proj", "layers.0.self_attn.v_proj",
        ]
        let unquantized = [
            "hyper_connection_mixer.hc_norm.weight",
            "layers.0.attn_hyper_connection.block_inject_weight.weight",
            "layers.0.attn_hyper_connection.hc_norm.weight",
            "layers.0.mlp.shared_expert_gate.weight",
            "layers.0.mlp_hyper_connection.block_inject_weight.weight",
            "layers.0.mlp_hyper_connection.hc_norm.weight",
            "layers.0.self_attn.indexer.k_layernorm.weight",
            "layers.0.self_attn.indexer.q_layernorm.weight",
            "layers.0.self_attn.k_norm.weight",
            "layers.0.self_attn.q_norm.weight",
            "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
        ]
        return quantizedBases.flatMap { base in
            ["\(base).weight", "\(base).scales", "\(base).biases"]
        } + unquantized
    }()

    static func embeddedQwenNextMTPExpectedTensors(
        config: [String: Any]
    ) -> [String: (AFMSafetensorHeader.DType, [Int])]? {
        let text = config["text_config"] as? [String: Any] ?? config
        let quantization = config["quantization"] as? [String: Any]
            ?? config["quantization_config"] as? [String: Any]
        func positive(_ key: String, default fallback: Int? = nil) -> Int? {
            let value = (text[key] as? NSNumber)?.intValue ?? fallback
            guard let value, value > 0, value <= 1_000_000 else { return nil }
            return value
        }
        guard let hidden = positive("hidden_size"),
              let heads = positive("num_attention_heads"),
              let kvHeads = positive("num_key_value_heads"),
              let headDim = positive("head_dim", default: hidden / heads),
              let experts = positive("num_experts"),
              let moe = positive("moe_intermediate_size"),
              let shared = positive("shared_expert_intermediate_size"),
              let hcCount = positive("hc_count", default: 4),
              let hcLowRank = positive("hc_lowrank", default: 320),
              let indexHeads = positive("indexer_n_heads", default: 4),
              let indexKVHeads = positive("indexer_kv_heads", default: 1),
              let indexHeadDim = positive("indexer_head_dim", default: 128),
              let groupSize = (quantization?["group_size"] as? NSNumber)?.intValue,
              let bits = (quantization?["bits"] as? NSNumber)?.intValue,
              groupSize > 0, bits > 0, bits <= 8
        else { return nil }

        let hcWidth = hidden * hcCount
        var expected = [String: (AFMSafetensorHeader.DType, [Int])]()
        func affine(_ base: String, output: Int, input: Int, leading: [Int] = []) {
            guard input.isMultiple(of: groupSize), (input * bits).isMultiple(of: 32) else {
                return
            }
            expected[base + ".weight"] = (.uint32, leading + [output, input * bits / 32])
            expected[base + ".scales"] = (.bfloat16, leading + [output, input / groupSize])
            expected[base + ".biases"] = (.bfloat16, leading + [output, input / groupSize])
        }
        affine("fc_embedding", output: hidden, input: hidden)
        affine("fc_hidden", output: hidden, input: hidden)
        for base in [
            "hyper_connection_mixer", "layers.0.attn_hyper_connection",
            "layers.0.mlp_hyper_connection",
        ] {
            affine(base + ".input_mix_weight_down", output: hcLowRank, input: hcWidth)
            affine(base + ".input_mix_weight_up", output: hcWidth, input: hcLowRank)
        }
        affine("layers.0.mlp.gate", output: experts, input: hidden)
        affine("layers.0.mlp.shared_expert.down_proj", output: hidden, input: shared)
        affine("layers.0.mlp.shared_expert.gate_proj", output: shared, input: hidden)
        affine("layers.0.mlp.shared_expert.up_proj", output: shared, input: hidden)
        affine("layers.0.mlp.switch_mlp.down_proj", output: hidden, input: moe,
               leading: [experts])
        affine("layers.0.mlp.switch_mlp.gate_proj", output: moe, input: hidden,
               leading: [experts])
        affine("layers.0.mlp.switch_mlp.up_proj", output: moe, input: hidden,
               leading: [experts])
        affine("layers.0.self_attn.indexer.index_qk_proj",
               output: (indexHeads + indexKVHeads) * indexHeadDim, input: hidden)
        affine("layers.0.self_attn.k_proj", output: kvHeads * headDim, input: hidden)
        affine("layers.0.self_attn.o_proj", output: hidden, input: heads * headDim)
        affine("layers.0.self_attn.q_proj", output: heads * headDim * 2, input: hidden)
        affine("layers.0.self_attn.v_proj", output: kvHeads * headDim, input: hidden)

        let bf16: [String: [Int]] = [
            "hyper_connection_mixer.hc_norm.weight": [hcWidth],
            "layers.0.attn_hyper_connection.block_inject_weight.weight": [hcCount, hcWidth],
            "layers.0.attn_hyper_connection.hc_norm.weight": [hcWidth],
            "layers.0.mlp.shared_expert_gate.weight": [1, hidden],
            "layers.0.mlp_hyper_connection.block_inject_weight.weight": [hcCount, hcWidth],
            "layers.0.mlp_hyper_connection.hc_norm.weight": [hcWidth],
            "layers.0.self_attn.indexer.k_layernorm.weight": [indexHeadDim],
            "layers.0.self_attn.indexer.q_layernorm.weight": [indexHeadDim],
            "layers.0.self_attn.k_norm.weight": [headDim],
            "layers.0.self_attn.q_norm.weight": [headDim],
            "pre_fc_norm_embedding.weight": [hidden],
            "pre_fc_norm_hidden.weight": [hcWidth],
        ]
        for (name, shape) in bf16 { expected[name] = (.bfloat16, shape) }
        guard Set(expected.keys) == Set(embeddedQwenNextMTPRequiredSuffixes) else {
            return nil
        }
        return expected
    }

    private static func hasCompleteEmbeddedGLMMTP(
        modelDirectory: URL,
        config: [String: Any]
    ) -> Bool {
        guard hasEmbeddedGLMMTPConfiguration(config) else { return false }
        let text = config["text_config"] as? [String: Any] ?? config
        guard let dimensions = EmbeddedGLMMTPDimensions(config: config, text: text) else {
            return false
        }
        let prefixes = [
            "model.language_model.layers.\(dimensions.layers).",
            "language_model.model.layers.\(dimensions.layers).",
            "model.layers.\(dimensions.layers).",
        ]
        var prefix: String
        var tensors = [String: AFMSafetensorHeader.Tensor]()

        let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            guard let data = boundedData(at: indexURL, maximumBytes: 128 * 1_024 * 1_024),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let map = object["weight_map"] as? [String: String] else { return false }
            let matches = prefixes.filter { candidate in
                map.keys.contains { $0.hasPrefix(candidate) }
            }
            guard matches.count == 1, let selected = matches.first else { return false }
            prefix = selected
            let mtpEntries = map.filter { $0.key.hasPrefix(prefix) }
            guard !mtpEntries.isEmpty else { return false }
            for (shard, entries) in Dictionary(grouping: mtpEntries, by: \.value) {
                guard let shardURL = containedShardURL(
                    named: shard, modelDirectory: modelDirectory) else { return false }
                guard let header = safeTensorHeader(at: shardURL),
                      hasSaneOffsets(header),
                      entries.allSatisfy({ entry in
                          header.tensors.contains(where: { $0.name == entry.key })
                      }) else { return false }
                let indexedNames = Set(entries.map(\.key))
                let headerNames = Set(header.tensors.lazy
                    .map(\.name)
                    .filter { $0.hasPrefix(prefix) })
                guard headerNames == indexedNames else { return false }
                for tensor in header.tensors where mtpEntries[tensor.name] == shard {
                    guard tensors.updateValue(tensor, forKey: tensor.name) == nil else {
                        return false
                    }
                }
            }
        } else {
            guard let tensorURL = containedShardURL(
                named: "model.safetensors", modelDirectory: modelDirectory) else { return false }
            guard let header = safeTensorHeader(at: tensorURL),
                  hasSaneOffsets(header) else { return false }
            let matches = prefixes.filter { candidate in
                header.tensors.contains { $0.name.hasPrefix(candidate) }
            }
            guard matches.count == 1, let selected = matches.first else { return false }
            prefix = selected
            for tensor in header.tensors where tensor.name.hasPrefix(prefix) {
                tensors[tensor.name] = tensor
            }
        }

        return dimensions.validate(tensors: tensors, prefix: prefix)
    }

    static func containedShardURL(named name: String, modelDirectory: URL) -> URL? {
        guard !name.hasPrefix("/"), !name.split(separator: "/").contains("..") else {
            return nil
        }
        let root = modelDirectory.resolvingSymlinksInPath().standardizedFileURL.path + "/"
        let url = modelDirectory.appendingPathComponent(name).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let resolved = url.resolvingSymlinksInPath()
        return resolved.path.hasPrefix(root) ? resolved : nil
    }

    private static func safeTensorHeader(at url: URL) -> AFMSafetensorHeader? {
        try? AFMSafetensorHeader(url: url)
    }

    private static func boundedData(at url: URL, maximumBytes: Int) -> Data? {
        guard let values = try? url.resourceValues(forKeys: [.fileSizeKey]),
              let size = values.fileSize, size >= 0, size <= maximumBytes else { return nil }
        return try? Data(contentsOf: url, options: [.mappedIfSafe])
    }

    private static func hasSaneOffsets(_ header: AFMSafetensorHeader) -> Bool {
        var priorEnd = 0
        for tensor in header.tensors {
            guard tensor.dataOffsets[0] >= priorEnd else { return false }
            priorEnd = tensor.dataOffsets[1]
        }
        return true
    }

    private static func isDenseGemma4VerifierConfiguration(_ config: [String: Any]) -> Bool {
        let modelType = AFMMLXModelArchitecture.canonicalModelType(config["model_type"] as? String ?? "")
        let architecture = ((config["architectures"] as? [String]) ?? []).joined(separator: " ").lowercased()
        return modelType == "gemma4" && !architecture.contains("moe")
    }
}

private struct EmbeddedGLMMTPDimensions {
    struct Quantization {
        let groupSize: Int
        let bits: Int
    }

    let layers: Int
    let hidden: Int
    let qLora: Int
    let kvLora: Int
    let attentionHeads: Int
    let qkNopeHead: Int
    let valueHead: Int
    let indexHeads: Int
    let indexHead: Int
    let indexPool: Int
    let experts: Int
    let sharedExperts: Int
    let moeIntermediate: Int
    let attentionBias: Bool
    let quantization: Quantization?

    init?(config: [String: Any], text: [String: Any]) {
        func integer(_ key: String) -> Int? {
            (text[key] as? NSNumber)?.intValue
        }
        guard let layers = integer("num_hidden_layers"), (1 ... 4_096).contains(layers),
              let hidden = integer("hidden_size"), (1 ... 1_048_576).contains(hidden),
              let qLora = integer("q_lora_rank"), (1 ... 1_048_576).contains(qLora),
              let kvLora = integer("kv_lora_rank"), (1 ... 1_048_576).contains(kvLora),
              let attentionHeads = integer("num_attention_heads"),
              (1 ... 65_536).contains(attentionHeads),
              let qkNopeHead = integer("qk_nope_head_dim"),
              (1 ... 1_048_576).contains(qkNopeHead),
              let valueHead = integer("v_head_dim"), (1 ... 1_048_576).contains(valueHead),
              let indexHeads = integer("index_n_heads"), (1 ... 65_536).contains(indexHeads),
              let indexHead = integer("index_head_dim"), (1 ... 1_048_576).contains(indexHead),
              let indexPool = integer("index_kpool"), (1 ... 65_536).contains(indexPool),
              let experts = integer("n_routed_experts"), (1 ... 65_536).contains(experts),
              let moeIntermediate = integer("moe_intermediate_size"),
              (1 ... 1_048_576).contains(moeIntermediate)
        else { return nil }
        let sharedExperts = integer("n_shared_experts") ?? 0
        guard (0 ... 65_536).contains(sharedExperts) else { return nil }

        self.layers = layers
        self.hidden = hidden
        self.qLora = qLora
        self.kvLora = kvLora
        self.attentionHeads = attentionHeads
        self.qkNopeHead = qkNopeHead
        self.valueHead = valueHead
        self.indexHeads = indexHeads
        self.indexHead = indexHead
        self.indexPool = indexPool
        self.experts = experts
        self.sharedExperts = sharedExperts
        self.moeIntermediate = moeIntermediate
        attentionBias = (text["attention_bias"] as? Bool) ?? false

        let rawQuantization = (config["quantization"] as? [String: Any])
            ?? (config["quantization_config"] as? [String: Any])
            ?? (text["quantization"] as? [String: Any])
            ?? (text["quantization_config"] as? [String: Any])
        if let rawQuantization,
           let groupSize = (rawQuantization["group_size"] as? NSNumber)?.intValue,
           let bits = (rawQuantization["bits"] as? NSNumber)?.intValue,
           groupSize > 0, bits > 0, bits <= 16 {
            quantization = Quantization(groupSize: groupSize, bits: bits)
        } else if rawQuantization != nil {
            return nil
        } else {
            quantization = nil
        }
    }

    func validate(
        tensors: [String: AFMSafetensorHeader.Tensor],
        prefix: String
    ) -> Bool {
        guard let doubledHidden = multiplied(hidden, 2),
              let queryWidth = multiplied(attentionHeads, qkNopeHead),
              let keyValueHead = added(qkNopeHead, valueHead),
              let keyValueWidth = multiplied(attentionHeads, keyValueHead),
              let outputWidth = multiplied(attentionHeads, valueHead),
              let indexQueryWidth = multiplied(indexHeads, indexHead),
              let sharedWidth = multiplied(moeIntermediate, sharedExperts)
        else { return false }
        var fixedShapes: [String: [Int]] = [
            "enorm.weight": [hidden],
            "hnorm.weight": [hidden],
            "input_layernorm.weight": [hidden],
            "post_attention_layernorm.weight": [hidden],
            "self_attn.q_a_layernorm.weight": [qLora],
            "self_attn.kv_a_layernorm.weight": [kvLora],
            "self_attn.indexer.k_norm.weight": [indexHead],
            "self_attn.indexer.k_norm.bias": [indexHead],
            "self_attn.indexer.index_kpool_compress_ape": [indexPool, indexHead],
            "self_attn.indexer.index_kpool_compress_gate": [indexHead, hidden],
            "mlp.gate.weight": [experts, hidden],
            "mlp.gate.e_score_correction_bias": [experts],
            "shared_head.norm.weight": [hidden],
        ]
        var linearShapes: [String: (output: Int, input: Int)] = [
            "eh_proj": (hidden, doubledHidden),
            "self_attn.q_a_proj": (qLora, hidden),
            "self_attn.q_b_proj": (queryWidth, qLora),
            "self_attn.kv_a_proj_with_mqa": (kvLora, hidden),
            "self_attn.kv_b_proj": (keyValueWidth, kvLora),
            "self_attn.o_proj": (hidden, outputWidth),
            "self_attn.indexer.wq_b": (indexQueryWidth, qLora),
            "self_attn.indexer.wk": (indexHead, hidden),
            "self_attn.indexer.weights_proj": (indexHeads, hidden),
        ]
        for expert in 0 ..< experts {
            linearShapes["mlp.experts.\(expert).gate_proj"] = (moeIntermediate, hidden)
            linearShapes["mlp.experts.\(expert).up_proj"] = (moeIntermediate, hidden)
            linearShapes["mlp.experts.\(expert).down_proj"] = (hidden, moeIntermediate)
        }
        if sharedExperts > 0 {
            linearShapes["mlp.shared_experts.gate_proj"] = (sharedWidth, hidden)
            linearShapes["mlp.shared_experts.up_proj"] = (sharedWidth, hidden)
            linearShapes["mlp.shared_experts.down_proj"] = (hidden, sharedWidth)
        }
        if attentionBias {
            fixedShapes["self_attn.q_a_proj.bias"] = [qLora]
            fixedShapes["self_attn.kv_a_proj_with_mqa.bias"] = [kvLora]
            fixedShapes["self_attn.o_proj.bias"] = [hidden]
        }

        for (name, shape) in fixedShapes {
            guard let tensor = tensors[prefix + name],
                  tensor.shape == shape,
                  tensor.dtype.isFloatingPoint else { return false }
        }
        for (name, dimensions) in linearShapes {
            guard validateLinear(
                name: prefix + name,
                output: dimensions.output,
                input: dimensions.input,
                tensors: tensors) else { return false }
        }
        return true
    }

    private func validateLinear(
        name: String,
        output: Int,
        input: Int,
        tensors: [String: AFMSafetensorHeader.Tensor]
    ) -> Bool {
        guard let weight = tensors[name + ".weight"] else { return false }
        let scales = tensors[name + ".scales"]
        let biases = tensors[name + ".biases"]
        guard (scales == nil) == (biases == nil) else { return false }

        if weight.dtype == .uint32 {
            guard let packedBits = multiplied(input, quantization?.bits ?? 0) else {
                return false
            }
            guard let quantization,
                  input % quantization.groupSize == 0,
                  packedBits % 32 == 0,
                  weight.shape == [output, packedBits / 32],
                  let scales, let biases,
                  scales.shape == [output, input / quantization.groupSize],
                  biases.shape == scales.shape,
                  scales.dtype.isFloatingPoint,
                  biases.dtype.isFloatingPoint else { return false }
            return true
        }

        return weight.shape == [output, input]
            && weight.dtype.isFloatingPoint
            && scales == nil
    }

    private func multiplied(_ lhs: Int, _ rhs: Int) -> Int? {
        let result = lhs.multipliedReportingOverflow(by: rhs)
        return result.overflow ? nil : result.partialValue
    }

    private func added(_ lhs: Int, _ rhs: Int) -> Int? {
        let result = lhs.addingReportingOverflow(rhs)
        return result.overflow ? nil : result.partialValue
    }
}

public enum AFMMLXSpeculativeRuntimeKind: Equatable, Sendable {
    case none
    case mtp
    case eagle3
}

public enum AFMMLXSpeculativeGenerationPath: String, Equatable, Sendable {
    case normal = "Normal MLX"
    case mtp = "MTP"
    case eagle3 = "EAGLE3"
    case fallback = "Fallback"
}

public enum AFMMLXSpeculativeFallbackReason: String, Equatable, Sendable {
    case modeOff = "Acceleration off"
    case runtimeUnavailable = "Runtime unavailable"
    case samplingEnabled = "Sampling enabled"
    case generationModifiers = "Generation modifiers enabled"
    case reasoningOutput = "Reasoning output enabled"
    case visionInput = "Vision input"
    case stopSequences = "Stop sequences enabled"
}

public struct AFMMLXSpeculativeGenerationDecision: Equatable, Sendable {
    public let path: AFMMLXSpeculativeGenerationPath
    public let reason: AFMMLXSpeculativeFallbackReason?

    public init(
        path: AFMMLXSpeculativeGenerationPath,
        reason: AFMMLXSpeculativeFallbackReason?
    ) {
        self.path = path
        self.reason = reason
    }

    public static func evaluate(
        mode: AFMMLXSpeculativeDecodingMode,
        installedRuntime: AFMMLXSpeculativeRuntimeKind,
        temperature: Double,
        hasUnsupportedGenerationModifiers: Bool,
        hasReasoningOutput: Bool,
        hasImages: Bool,
        hasStopSequences: Bool
    ) -> AFMMLXSpeculativeGenerationDecision {
        guard mode != .off else {
            return AFMMLXSpeculativeGenerationDecision(path: .normal, reason: .modeOff)
        }
        guard !hasImages else {
            return AFMMLXSpeculativeGenerationDecision(path: .fallback, reason: .visionInput)
        }
        guard !hasStopSequences else {
            return AFMMLXSpeculativeGenerationDecision(path: .fallback, reason: .stopSequences)
        }
        guard !hasUnsupportedGenerationModifiers else {
            return AFMMLXSpeculativeGenerationDecision(path: .fallback, reason: .generationModifiers)
        }
        guard !hasReasoningOutput else {
            return AFMMLXSpeculativeGenerationDecision(path: .fallback, reason: .reasoningOutput)
        }
        guard abs(temperature) < 0.000_001 else {
            return AFMMLXSpeculativeGenerationDecision(path: .fallback, reason: .samplingEnabled)
        }

        switch (mode, installedRuntime) {
        case (.auto, .mtp), (.mtp, .mtp):
            return AFMMLXSpeculativeGenerationDecision(path: .mtp, reason: nil)
        case (.auto, .eagle3), (.eagle3, .eagle3):
            return AFMMLXSpeculativeGenerationDecision(path: .eagle3, reason: nil)
        case (.mtp, _), (.eagle3, _), (.auto, .none):
            return AFMMLXSpeculativeGenerationDecision(path: .fallback, reason: .runtimeUnavailable)
        case (.off, _):
            return AFMMLXSpeculativeGenerationDecision(path: .normal, reason: .modeOff)
        }
    }

    public static func completedRuntimeDecision(
        initialDecision: AFMMLXSpeculativeGenerationDecision,
        emittedChunkCount: Int
    ) -> AFMMLXSpeculativeGenerationDecision {
        guard (initialDecision.path == .mtp || initialDecision.path == .eagle3),
              emittedChunkCount <= 0 else {
            return initialDecision
        }
        return AFMMLXSpeculativeGenerationDecision(
            path: .fallback,
            reason: .runtimeUnavailable
        )
    }
}

public struct AFMMLXSpeculativeGenerationCompletionSummary: Equatable, Sendable {
    public let shouldCommit: Bool
    public let historyText: String
    public let finishReason: AFMFinishReason
    public let tokensPerSecond: Double

    public init(
        shouldCommit: Bool,
        historyText: String,
        finishReason: AFMFinishReason,
        tokensPerSecond: Double
    ) {
        self.shouldCommit = shouldCommit
        self.historyText = historyText
        self.finishReason = finishReason
        self.tokensPerSecond = tokensPerSecond
    }
}

public enum AFMMLXSpeculativeGenerationCompletionPolicy {
    public static func summary(
        accumulatedText: String
    ) -> AFMMLXSpeculativeGenerationCompletionSummary {
        guard !accumulatedText.isEmpty else {
            return AFMMLXSpeculativeGenerationCompletionSummary(
                shouldCommit: false,
                historyText: "",
                finishReason: .stop,
                tokensPerSecond: 0
            )
        }
        return AFMMLXSpeculativeGenerationCompletionSummary(
            shouldCommit: true,
            historyText: accumulatedText,
            finishReason: .stop,
            tokensPerSecond: 0
        )
    }
}
