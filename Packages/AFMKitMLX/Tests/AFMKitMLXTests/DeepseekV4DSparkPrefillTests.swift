import MLX
import MLXLMCommon
@testable import MLXLLM
import XCTest
@testable import AFMKitMLX

final class DeepseekV4DSparkPrefillTests: XCTestCase {
    override func setUpWithError() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
    }

    private func makeModel() -> DeepseekV4Model {
        var config = DeepseekV4Configuration()
        config.vocabSize = 32
        config.hiddenSize = 64
        config.numHiddenLayers = 3
        config.numAttentionHeads = 2
        config.headDim = 64
        config.qkRopeHeadDim = 16
        config.qLoraRank = 32
        config.oGroups = 2
        config.oLoraRank = 16
        config.nRoutedExperts = 2
        config.numExpertsPerTok = 1
        config.moeIntermediateSize = 32
        config.numHashLayers = 0
        config.slidingWindow = 8
        config.compressRatios = [0, 4, 128, 0, 0]
        config.indexNHeads = 2
        config.indexHeadDim = 64
        config.indexTopk = 4
        config.dsparkBlockSize = 3
        config.dsparkNoiseTokenId = 0
        config.dsparkTargetLayerIds = [0, 2]
        config.dsparkMarkovRank = 8
        config.activationQATEnabled = false
        return DeepseekV4Model(config)
    }

    private func ids(_ values: [Int]) -> MLXArray {
        MLXArray(values.map(Int32.init)).reshaped([1, values.count])
    }

    private func assertClose(_ actual: MLXArray, _ expected: MLXArray,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertTrue(allClose(actual, expected, rtol: 0.005, atol: 0.005)
            .item(Bool.self), file: file, line: line)
    }

    func testFullAndChunkedPrefillPreserveLogitsAndNextProposalAcrossBoundaries() throws {
        let model = makeModel()
        // Local-ring, ratio-4 and ratio-128 boundaries, including a one-token
        // final chunk and chunk sizes that do not divide compression groups.
        for (length, step) in [(1, 1), (7, 3), (8, 4), (9, 4), (17, 5), (129, 17)] {
            let prompt = ids((0..<length).map { $0 % 31 })
            let fullTarget = model.newCache(parameters: nil)
            let fullDraft = model.newDSparkCache()
            let full = try XCTUnwrap(model.forwardDSparkVerifier(prompt, cache: fullTarget))
            MLX.eval([full.logits, full.captured] + fullTarget.flatMap { $0.innerState() })
            let anchor = ids([argMax(full.logits[0, -1, 0...]).item(Int.self)])
            XCTAssertTrue(model.prefillDSpark(anchorTokenIds: anchor,
                capturedHidden: full.captured, cache: fullDraft))
            MLX.eval(fullDraft.flatMap { $0.innerState() })

            let chunkTarget = model.newCache(parameters: nil)
            let chunkDraft = model.newDSparkCache()
            let chunk = try XCTUnwrap(model.prefillDSparkVerifier(prompt,
                verifierCache: chunkTarget, drafterCache: chunkDraft, stepSize: step))
            assertClose(chunk, full.logits[0..., (-1)..., 0...])
            XCTAssertEqual(chunkTarget.map { $0.offset }, fullTarget.map { $0.offset })
            XCTAssertEqual(chunkDraft.map { $0.offset }, fullDraft.map { $0.offset })
            // A subsequent append normalizes rotating-cache over-allocation;
            // compare actual consumers, not prefill-dependent buffer capacity.
            let fullWarm = try XCTUnwrap(model.forwardDSparkVerifier(anchor, cache: fullTarget))
            let chunkWarm = try XCTUnwrap(model.forwardDSparkVerifier(anchor, cache: chunkTarget))
            assertClose(chunkWarm.logits, fullWarm.logits)
            assertClose(chunkWarm.captured, fullWarm.captured)
            let fullProposal = try XCTUnwrap(model.proposeDSpark(anchorTokenIds: anchor,
                capturedHidden: fullWarm.captured, cache: fullDraft))
            let chunkProposal = try XCTUnwrap(model.proposeDSpark(anchorTokenIds: anchor,
                capturedHidden: chunkWarm.captured, cache: chunkDraft))
            assertClose(chunkProposal.logits, fullProposal.logits)
            assertClose(chunkProposal.confidence, fullProposal.confidence)
            for (a, b) in zip(chunkDraft, fullDraft) {
                XCTAssertEqual(a.metaState, b.metaState)
                for (x, y) in zip(a.state, b.state) { assertClose(x, y) }
            }
        }
    }

    func testPrefillRejectsReusingPopulatedCaches() throws {
        let model = makeModel()
        let target = model.newCache(parameters: nil)
        let draft = model.newDSparkCache()
        XCTAssertNotNil(model.prefillDSparkVerifier(ids([1, 2]),
            verifierCache: target, drafterCache: draft, stepSize: 1))
        let offsets = target.map { $0.offset }
        XCTAssertNil(model.prefillDSparkVerifier(ids([3]),
            verifierCache: target, drafterCache: draft))
        XCTAssertEqual(target.map { $0.offset }, offsets)
    }
}
