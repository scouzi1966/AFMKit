import Foundation
import MLX
import XCTest
@testable import AFMKitMLX
@testable import MLXLLM

/// Independent host ordering: score descending, then original block index
/// ascending for exact ties. Selected blocks are emitted in cache order.
enum QwenQSARawScoreOracle {
    static func select(
        _ scores: [Float], blocks: Int, bounds: [Int], topK: Int
    ) -> [Int32] {
        precondition(scores.count == blocks * bounds.count)
        return bounds.enumerated().flatMap { row, bound in
            precondition((0...blocks).contains(bound))
            let ranked = (0..<bound).sorted { left, right in
                let lhs = scores[row * blocks + left]
                let rhs = scores[row * blocks + right]
                if lhs.isNaN != rhs.isNaN { return lhs.isNaN }
                if (lhs.isNaN && rhs.isNaN) || lhs == rhs { return left < right }
                return lhs > rhs
            }
            let selected = ranked.prefix(topK).sorted().map(Int32.init)
            return selected + Array(repeating: Int32.max, count: topK - selected.count)
        }
    }
}

final class QwenQSARawScoreSelectionTests: XCTestCase {
    private static let hiddenDimension = 16
    private static let indexHeadDimension = 128
    private static let compressionRatio = 4
    private static let topK = 512
    private static let earlyBlock = 600
    private static let baseScore: Float = 625
    private static let strongerScore: Float = 1_000

    func testDecodeAndPrefillKeepHigherRawScoreAcrossSeparatedBlocks() throws {
        try assertIndexerSelection(exactTie: false)
    }

    func testDecodeAndPrefillResolveExactRawScoreTiesByLowerIndex() throws {
        try assertIndexerSelection(exactTie: true)
    }

    func testMaskedDecodePreservesUnequalScoresAndExactTies() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        // A captured warm decode chose between these scores. An index bias
        // reversed them, despite the later block having the higher raw score.
        let blocks = 1_818, early = 994, late = 1_727
        let low = Float(bitPattern: 0x42e54423)
        let high = Float(bitPattern: 0x42e5442a)
        for exactTie in [false, true] {
            var scores = [Float](repeating: 0, count: blocks)
            for block in 0..<(Self.topK - 1) { scores[block] = Self.strongerScore }
            scores[early] = low
            scores[late] = exactTie ? low : high
            let selected = Set(QwenQSARawScoreOracle.select(
                scores, blocks: blocks, bounds: [blocks], topK: Self.topK))
            for tail in [0, 1, 3] {
                let visible = blocks * Self.compressionRatio + tail
                let capacity = visible + Self.compressionRatio
                let mask = try XCTUnwrap(Qwen4ExpQSADecodeMask.call(
                    scores: MLXArray(scores).reshaped(1, blocks),
                    visibleCount: visible, keyLength: capacity,
                    compressionRatio: Self.compressionRatio, blockTopK: Self.topK))
                let actual = mask.asArray(Bool.self)
                let expected = (0..<capacity).map { token in
                    token < visible && (token >= blocks * Self.compressionRatio
                        || selected.contains(Int32(token / Self.compressionRatio)))
                }
                XCTAssertEqual(actual.filter { $0 }.count,
                               Self.topK * Self.compressionRatio + tail)
                XCTAssertEqual(actual, expected, "exactTie=\(exactTie), tail=\(tail)")
            }
        }
    }

    private func assertIndexerSelection(exactTie: Bool) throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let values: [String: Any] = [
            "hidden_size": Self.hiddenDimension, "num_hidden_layers": 1,
            "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 256,
            "moe_intermediate_size": Self.hiddenDimension,
            "shared_expert_intermediate_size": Self.hiddenDimension,
            "num_experts_per_tok": 1, "num_experts": 1,
            "layer_types": ["full_attention"], "vocab_size": 32,
            "indexer_n_heads": 1, "indexer_kv_heads": 1,
            "indexer_head_dim": Self.indexHeadDimension,
            "indexer_budget": Self.topK * Self.compressionRatio,
            "indexer_compress_ratio": Self.compressionRatio,
            // Make the all-ones query normalization exactly the identity.
            // This fixture isolates selection from normalization/rotary math.
            "rms_norm_eps": 0.0,
        ]
        let configuration = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: values))
        let cases: [(blocks: Int, lateBlock: Int, increment: Float)] = [
            (1_536, 1_450, Self.baseScore.nextUp - Self.baseScore),
            (10_000, 9_500, 1.0 / 2_048),
        ]
        for fixture in cases {
            var rawScores = [Float](repeating: 0, count: fixture.blocks)
            for block in 0..<(Self.topK - 1) {
                rawScores[block] = Self.strongerScore
            }
            rawScores[Self.earlyBlock] = Self.baseScore
            rawScores[fixture.lateBlock] = Self.baseScore + (exactTie ? 0 : fixture.increment)

            for width in [1, 16] {
                let indexer = Qwen4ExpQSAIndexer(configuration)
                let cache = Qwen4ExpAttentionCache(indexerCompressRatio: Self.compressionRatio)
                let prefix = fixture.blocks * Self.compressionRatio
                _ = cache.updateIndexKeys(
                    MLXArray.zeros([1, prefix, Self.indexHeadDimension], dtype: .float32),
                    positionIDs: MLXArray.zeros([1, prefix], dtype: .int32))
                let kv = MLXArray.zeros([1, 1, prefix, 1], dtype: .bfloat16)
                _ = cache.update(keys: kv, values: kv)
                // With an all-ones query, each pooled key's first channel is
                // its raw dot-product score. Preserve near-ties in Float32.
                let bank = rawScores.flatMap { score in
                    [score] + [Float](repeating: 0, count: Self.indexHeadDimension - 1)
                }
                _ = cache.appendPooledIndexKeys(
                    MLXArray(bank).reshaped(1, fixture.blocks, Self.indexHeadDimension))
                let projectedQK = concatenated([
                    MLXArray.ones([1, width, Self.indexHeadDimension], dtype: .float32),
                    MLXArray.ones([1, width, Self.indexHeadDimension], dtype: .float32),
                ], axis: -1)
                let selection = indexer(
                    MLXArray.zeros([1, width, Self.hiddenDimension], dtype: .bfloat16),
                    positionIDs: MLXArray.zeros([1, width], dtype: .int32),
                    cache: cache, verificationPolicy: .strictSingletonEquivalent,
                    projectedQK: projectedQK)
                let totalLength = prefix + width
                let blocks = totalLength / Self.compressionRatio
                let rowScores = rawScores + [Float](
                    repeating: Float(Self.indexHeadDimension), count: blocks - fixture.blocks)
                let bounds = (0..<width).map { (prefix + $0 + 1) / Self.compressionRatio }
                let expected = QwenQSARawScoreOracle.select(
                    Array(repeating: rowScores, count: width).flatMap { $0 },
                    blocks: blocks, bounds: bounds, topK: Self.topK)
                let winner = exactTie ? Self.earlyBlock : fixture.lateBlock
                XCTAssertTrue(expected.prefix(Self.topK).contains(Int32(winner)))

                switch selection {
                case .blocks(let selected):
                    XCTAssertEqual(selected.asArray(Int32.self), expected,
                        "blocks=\(fixture.blocks), width=\(width), exactTie=\(exactTie)")
                case .mask(let mask):
                    // Short prefill uses the same selector, then expands its
                    // chosen IDs into a token mask. Check that dispatch too.
                    let expectedMask = bounds.enumerated().flatMap { row, bound in
                        let chosen = Set(expected[(row * Self.topK)..<((row + 1) * Self.topK)])
                        let end = prefix + row + 1
                        return (0..<totalLength).map { token in
                            token < end && (token >= bound * Self.compressionRatio
                                || chosen.contains(Int32(token / Self.compressionRatio)))
                        }
                    }
                    XCTAssertEqual(mask.asArray(Bool.self), expectedMask,
                        "blocks=\(fixture.blocks), width=\(width), exactTie=\(exactTie)")
                default:
                    XCTFail("Expected sparse QSA selection")
                }
            }
        }
    }
}
