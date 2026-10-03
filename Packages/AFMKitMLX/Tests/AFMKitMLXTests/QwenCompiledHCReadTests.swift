import Foundation
import MLX
import MLXNN
import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

final class QwenCompiledHCReadTests: XCTestCase {
    private let columns = 10240

    private func requireCompilation() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        try XCTSkipUnless(HardwareInfo.isModelOwnedCompiledDecodeSupported,
                          "Model-owned compilation is unsupported on this device")
        try XCTSkipUnless(Qwen4ExpHyperConnectionFusion.permitsStrictReadCompilation,
                          "Diagnostic HC configuration excludes strict read compilation")
    }

    private func configuration() throws -> Qwen4ExpTextConfiguration {
        let values: [String: Any] = [
            "hidden_size": 2560, "num_hidden_layers": 2, "vocab_size": 32,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 256,
            "linear_num_value_heads": 2, "linear_num_key_heads": 1,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64, "num_experts_per_tok": 1,
            "num_experts": 2, "hc_count": 4, "hc_lowrank": 320,
            "layer_types": ["linear_attention", "full_attention"],
            "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 128,
            "indexer_budget": 4, "indexer_compress_ratio": 4,
        ]
        return try JSONDecoder().decode(Qwen4ExpTextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: values))
    }

    private func read(seed: UInt64, group: Int? = 32) throws -> Qwen4ExpGatedResidual {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(seed)
        let value = Qwen4ExpGatedResidual(try configuration())
        value.update(parameters: value.mapParameters { (MLXRandom.normal($0.shape) * 0.03).asType(.bfloat16) })
        if let group { quantize(model: value, groupSize: group, bits: 4) }
        eval(value)
        return value
    }

    private func input(width: Int, seed: Int, strided: Bool = false) -> MLXArray {
        let features = strided ? columns * 2 : columns
        let storage = sin(MLXArray(0..<((width + 1) * features)).asType(.float32) * 0.021 + Float(seed))
            .asType(.bfloat16).reshaped(1, width + 1, features)
        let x = strided ? storage[0..., 1..<(width + 1), .stride(by: 2)]
            : storage[0..., 1..<(width + 1), 0...]
        eval(x)
        return x
    }

    private func values(_ tuple: (MLXArray, MLXArray, MLXArray)) -> [MLXArray] {
        [tuple.0, tuple.1, tuple.2]
    }

    private func exact(_ a: [MLXArray], _ b: [MLXArray], _ label: String) {
        XCTAssertEqual(a.count, b.count, label)
        for (x, y) in zip(a, b) {
            XCTAssertEqual(x.shape, y.shape, label)
            XCTAssertEqual(x.dtype, y.dtype, label)
            XCTAssertTrue(arrayEqual(x, y).item(Bool.self), label)
        }
    }

    func testProductionReadMatchesChangedStridedInputsAndIndependentOwners() throws {
        try requireCompilation()
        let owners = try [read(seed: 115), read(seed: 119)]
        var pending: [([MLXArray], [MLXArray])] = []
        for width in [2, 4, 7, 8] {
            for strided in [false, true] {
                for seed in [3, 17] {
                    let x = input(width: width, seed: seed, strided: strided)
                    for owner in owners {
                        XCTAssertTrue(owner.canCompileAttentionRead(x, policy: .strictSingletonEquivalent))
                        pending.append((values(owner.attentionRead(x, policy: .strictSingletonEquivalent, compiled: true)),
                            values(owner.mix(x, verificationPolicy: .strictSingletonEquivalent))))
                    }
                }
            }
        }
        for (actual, oracle) in pending.reversed() { exact(actual, oracle, "pending read") }
        let counts = owners.map(\.compiledReadTraceCount)
        XCTAssertTrue(counts.allSatisfy { (4...8).contains($0) })
        for owner in owners {
            for width in [2, 4, 7, 8] {
                for strided in [false, true] {
                    let x = input(width: width, seed: 43, strided: strided)
                    exact(values(owner.attentionRead(x, policy: .strictSingletonEquivalent, compiled: true)),
                          values(owner.mix(x, verificationPolicy: .strictSingletonEquivalent)), "changed input")
                }
            }
        }
        XCTAssertEqual(owners.map(\.compiledReadTraceCount), counts)
        let x = input(width: 4, seed: 71)
        XCTAssertFalse(arrayEqual(owners[0].mix(x).0, owners[1].mix(x).0).item(Bool.self),
                       "Distinct owners must not reuse another model's captured weights")
    }

    func testUnsupportedGeometryAndNestedTracingUseOriginalPath() throws {
        let owner = try read(seed: 127)
        let x = input(width: 4, seed: 3)
        for policy: MTPVerificationPolicy? in [nil, .batched] {
            XCTAssertFalse(owner.canCompileAttentionRead(x, policy: policy))
            exact(values(owner.attentionRead(x, policy: policy, compiled: true)),
                  values(owner.mix(x, verificationPolicy: policy)), "non-strict")
        }
        for width in [1, 9] {
            let x = input(width: width, seed: 4)
            XCTAssertFalse(owner.canCompileAttentionRead(x, policy: .strictSingletonEquivalent))
            exact(values(owner.attentionRead(x, policy: .strictSingletonEquivalent, compiled: true)),
                  values(owner.mix(x, verificationPolicy: .strictSingletonEquivalent)), "width fallback")
        }
        for dtype in [DType.float16, .float32] {
            XCTAssertFalse(owner.canCompileAttentionRead(x.asType(dtype), policy: .strictSingletonEquivalent))
        }
        XCTAssertFalse(owner.canCompileAttentionRead(concatenated([x, x], axis: 0), policy: .strictSingletonEquivalent))
        CompiledDecodeTrace.withActive {
            XCTAssertFalse(owner.canCompileAttentionRead(x, policy: .strictSingletonEquivalent))
            exact(values(owner.attentionRead(x, policy: .strictSingletonEquivalent, compiled: true)),
                  values(owner.mix(x, verificationPolicy: .strictSingletonEquivalent)), "nested marker")
        }
        exact(values(owner.attentionRead(x, policy: .strictSingletonEquivalent, compiled: false)),
              values(owner.mix(x, verificationPolicy: .strictSingletonEquivalent)), "disabled")
        XCTAssertEqual(owner.compiledReadTraceCount, 0)
        for group: Int? in [64, nil] {
            let unsupported = try read(seed: 139, group: group)
            XCTAssertFalse(unsupported.canCompileAttentionRead(x, policy: .strictSingletonEquivalent))
            exact(values(unsupported.attentionRead(x, policy: .strictSingletonEquivalent, compiled: true)),
                  values(unsupported.mix(x, verificationPolicy: .strictSingletonEquivalent)), "weight fallback")
            XCTAssertEqual(unsupported.compiledReadTraceCount, 0)
        }
    }

    func testExecutionStreamsAndPendingOutputsAfterOwnerRelease() throws {
        try requireCompilation()
        var owner: Qwen4ExpGatedResidual? = try read(seed: 149)
        weak var weakOwner = owner
        let x = input(width: 4, seed: 9)
        let expected = values(owner!.mix(x, verificationPolicy: .strictSingletonEquivalent)).map { $0.asArray(Float.self) }
        var pending = values(owner!.attentionRead(x, policy: .strictSingletonEquivalent, compiled: true))
        let firstCount = owner!.compiledReadTraceCount
        Stream.withNewDefaultStream(device: .gpu) {
            let actual = values(owner!.attentionRead(x, policy: .strictSingletonEquivalent, compiled: true))
            for (value, oracle) in zip(actual, expected) { XCTAssertEqual(value.asArray(Float.self), oracle) }
        }
        XCTAssertEqual(owner!.compiledReadTraceCount, firstCount + 1)
        pending += values(owner!.attentionRead(x, policy: .strictSingletonEquivalent, compiled: true))
        XCTAssertEqual(owner!.compiledReadTraceCount, firstCount + 1)
        owner = nil
        XCTAssertNil(weakOwner)
        for (index, value) in pending.enumerated() { XCTAssertEqual(value.asArray(Float.self), expected[index % 3]) }
    }

    func testRecurrentAndAttentionLayerStateAndContinuationRemainExact() throws {
        try requireCompilation()
        MLXRandom.seed(157)
        for index in [0, 1] {
            let layer = Qwen4ExpDecoderLayer(try configuration(), layerIndex: index)
            layer.update(parameters: layer.mapParameters { (MLXRandom.normal($0.shape) * 0.03).asType(.bfloat16) })
            quantize(model: layer, groupSize: 32, bits: 4)
            eval(layer)
            for width in [2, 4, 7, 8] {
                for keep in 1...width {
                    let pair: [KVCache] = (0..<2).map { _ -> KVCache in
                        if index == 0 { return Qwen4ExpLayerCache() }
                        return Qwen4ExpAttentionCache(indexerCompressRatio: 4)
                    }
                    let prefix = input(width: 7, seed: 21)
                    func forward(_ x: MLXArray, row: Int, policy: MTPVerificationPolicy?) -> MLXArray {
                        layer.compiledHCReadOverrideForTesting = row == 1
                        return layer(x, inputIDs: MLXArray.zeros([1, x.dim(1)], dtype: .int32),
                            attentionMask: .causal, positionIDs: nil, cache: pair[row], verificationPolicy: policy)
                    }
                    exact([forward(prefix, row: 0, policy: nil)], [forward(prefix, row: 1, policy: nil)], "prefill")
                    for cache in pair {
                        (cache as? Qwen4ExpLayerCache)?.beginMTPVerification(width: width)
                        (cache as? Qwen4ExpAttentionCache)?.beginMTPVerification(width: width)
                    }
                    let x = input(width: width, seed: 23)
                    let a = forward(x, row: 0, policy: .strictSingletonEquivalent)
                    let b = forward(x, row: 1, policy: .strictSingletonEquivalent)
                    exact([a], [b], "integrated verification")
                    exact(pair[0].state, pair[1].state, "verified cache")
                    for cache in pair {
                        if let recurrent = cache as? Qwen4ExpLayerCache {
                            layer.rollbackGatedDeltaForTesting(recurrent, keeping: keep)
                            recurrent.clearMTPRollback()
                        } else { _ = cache.trim(width - keep) }
                    }
                    exact(pair[0].state, pair[1].state, "rolled-back cache")
                    let next = input(width: 1, seed: 31)
                    exact([forward(next, row: 0, policy: nil)], [forward(next, row: 1, policy: nil)], "next decode")
                    exact(pair[0].state, pair[1].state, "continued cache")
                    XCTAssertEqual(pair[0].offset, pair[1].offset)
                }
            }
            XCTAssertGreaterThan(layer.compiledHCReadTraceCountForTesting, 0)
        }
    }
}
