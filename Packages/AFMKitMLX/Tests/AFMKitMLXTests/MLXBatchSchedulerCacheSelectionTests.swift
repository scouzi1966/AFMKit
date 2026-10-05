import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import XCTest

@testable import AFMKitMLX

private final class TestOnlyArraysCache: ArraysCache {}

private class TestUniformDecodeCache: ArraysCache, UniformBatchKVCache {
    init(offset: Int, value: Float, width: Int = 2, dtype: DType = .float32) {
        super.init(size: 1)
        self.offset = offset
        self[0] = MLXArray.full([1, 1, 1, width], values: MLXArray(value)).asType(dtype)
    }

    func mergedUniformBatch(_ caches: [KVCache]) -> KVCache {
        let merged = TestUniformDecodeCache(offset: offset, value: 0)
        merged.state = [concatenated(caches.map { $0.state[0] }, axis: 0)]
        return merged
    }

    func extendUniformBatch(with cache: KVCache) {
        state = [concatenated([state[0], cache.state[0]], axis: 0)]
    }

    func filterUniformBatch(_ indices: [Int]) {
        precondition(!indices.isEmpty)
        state = [state[0][MLXArray(indices.map(Int32.init))]]
    }
}

private final class OtherUniformDecodeCache: TestUniformDecodeCache {}

final class MLXBatchSchedulerCacheSelectionTests: XCTestCase {
    func testARCoarseAnchorRequiresExplicitEligibleNonMTPReplay() {
        for enabled in [false, true] {
            for mtp in [false, true] {
                for backoff in [0, 1, 2, 31, 256] {
                    XCTAssertEqual(BatchScheduler.qwenARReplayUsesCoarseAnchor(
                        backoffTokens: backoff, ownsMTP: mtp, enabled: enabled),
                        enabled && !mtp && backoff > 1)
                }
            }
        }
    }

    func testCoarseAnchorIsBoundedOrderedAndPreservesNearEndBoundary() {
        XCTAssertEqual(MLXReplayPrefill.boundaries(restoredPrefix: 0, finalBoundary: 2010,
            promptSnapshotBackoffTokens: 31, retainCoarseAnchor: true), [1792, 1980])
        XCTAssertEqual(MLXReplayPrefill.boundaries(restoredPrefix: 100, finalBoundary: 2010,
            promptSnapshotBackoffTokens: 31, retainCoarseAnchor: true), [1892, 1980])
        for final in [1, 20, 31, 32, 256, 300, 1024, 2048, 32768, Int.max] {
            for restored in [0, final / 2, final - 1, final] {
                for backoff in [0, 1, 2, 31, 256] {
                    for limit in [0, 1, 8] {
                        let plain = MLXReplayPrefill.boundaries(restoredPrefix: restored,
                            finalBoundary: final, maximumCheckpoints: limit,
                            promptSnapshotBackoffTokens: backoff)
                        let anchored = MLXReplayPrefill.boundaries(restoredPrefix: restored,
                            finalBoundary: final, maximumCheckpoints: limit,
                            promptSnapshotBackoffTokens: backoff, retainCoarseAnchor: true)
                        if backoff == 0 || plain.isEmpty || limit < 2 {
                            XCTAssertEqual(anchored, plain)
                        } else {
                            XCTAssertEqual(anchored.last, plain.last)
                            XCTAssertLessThanOrEqual(anchored.count, 2)
                            let grid = MLXReplayPrefill.boundaries(restoredPrefix: restored,
                                finalBoundary: final, maximumCheckpoints: limit)
                            let expectedAnchor = grid.last(where: { $0 < plain[0] })
                            XCTAssertEqual(anchored, expectedAnchor.map { [$0] + plain } ?? plain)
                            if anchored.count == 2 {
                                XCTAssertLessThan(anchored[0], anchored[1])
                                XCTAssertEqual(anchored[0], grid.last(where: { $0 < plain[0] }))
                            }
                        }
                        XCTAssertTrue(anchored.allSatisfy { $0 > restored && $0 < final })
                        XCTAssertEqual(Set(anchored).count, anchored.count)
                    }
                }
            }
        }
    }

    func testVLMTextNearEndReplayRequiresExplicitAROptInAndPreservesTrunkPolicy() {
        for type in [Qwen4ExpVL.self, Qwen4ExpModel.self, Gemma4VLM.self] as [Any.Type] {
            for enabled in [false, true] {
                for prefix in [false, true] {
                    for mtp in [false, true] {
                        for (value, count) in [(nil, 31), ("0", 0), ("17", 17), ("bad", 31)] as [(String?, Int)] {
                            let eligible = prefix && (type == Qwen4ExpModel.self
                                || (type == Qwen4ExpVL.self && enabled && !mtp))
                            XCTAssertEqual(BatchScheduler.qwenARReplayBackoffTokenCount(
                                modelType: type, prefixCacheEnabled: prefix, ownsMTP: mtp,
                                vlmTextReplayEnabled: enabled, value: value), eligible ? count : 0)
                        }
                    }
                }
            }
        }
        XCTAssertFalse(BatchScheduler.shouldCaptureReplayBoundary(
            prefixCacheEnabled: true, hasRecurrentLayers: true,
            isMultimodal: true, inputTokenCount: 2048))
    }

    func testContinuousVLMTextAdmissionRequiresMixedPositionARAndExplicitOptIn() {
        for type in [Qwen4ExpVL.self, Qwen4ExpModel.self, Gemma4VLM.self] as [Any.Type] {
            for enabled in [false, true] {
                for requestOwned in [false, true] {
                    for mtp in [false, true] {
                        XCTAssertEqual(BatchScheduler.supportsContinuousVLMTextAdmission(
                            modelType: type, requestOwnedBatch: requestOwned,
                            ownsMTP: mtp, enabled: enabled),
                            type == Qwen4ExpVL.self && enabled && requestOwned && !mtp)
                    }
                }
            }
        }
        // This narrow exception does not remove the architecture's media barrier.
        XCTAssertTrue(BatchScheduler.requiresFixedDecodeCohorts(
            for: Qwen4ExpVL.self, continuousUniformGroups: true))
    }

    func testContinuousVLMTextAdmissionRejectsStateSpeculationAndTransferredCaches() {
        for eligible in [false, true] {
            for state in [false, true] {
                for speculation in [false, true] {
                    for caches in [false, true] {
                        let permitted = BatchScheduler.permitsContinuousVLMTextSlot(
                            permitsDecodeGroup: eligible, hasModelState: state,
                            hasSpeculativeSession: speculation, hasRequestCaches: caches)
                        XCTAssertEqual(permitted, eligible && !state && !speculation && caches)
                        XCTAssertEqual(BatchScheduler.shouldDeferStaggeredAdmissions(
                            requiresFixedDecodeCohorts: true, activeSlotCount: 1,
                            permitsVLMTextAdmission: permitted), !permitted)
                    }
                }
            }
        }
        XCTAssertTrue(BatchScheduler.shouldDeferStaggeredAdmissions(
            requiresFixedDecodeCohorts: true, activeSlotCount: 1))
        XCTAssertFalse(BatchScheduler.shouldDeferStaggeredAdmissions(
            requiresFixedDecodeCohorts: true, activeSlotCount: 0))
    }

    func testContinuousVLMTextAdmissionRejectsMediaAndNonSingletonShapes() {
        let image = LMInput.ProcessedImage(pixels: MLXArray.zeros([1, 3, 2, 2]))
        let video = LMInput.ProcessedVideo(pixels: MLXArray.zeros([1, 3, 2, 2]))
        for shape in [[], [3], [1, 3], [2, 3], [1, 1, 3]] {
            let text = LMInput.Text(tokens: MLXArray.zeros(shape, dtype: .int32))
            XCTAssertEqual(BatchScheduler.isOrdinaryVLMTextAdmission(
                input: LMInput(text: text), usesSpeculativeSession: false),
                shape == [3] || shape == [1, 3])
            XCTAssertFalse(BatchScheduler.isOrdinaryVLMTextAdmission(
                input: LMInput(text: text), usesSpeculativeSession: true))
            XCTAssertFalse(BatchScheduler.isOrdinaryVLMTextAdmission(
                input: LMInput(text: text, image: image), usesSpeculativeSession: false))
            XCTAssertFalse(BatchScheduler.isOrdinaryVLMTextAdmission(
                input: LMInput(text: text, video: video), usesSpeculativeSession: false))
        }
    }

    func testContinuousVLMTextAdmissionPreservesFIFOBarrierAndCapacity() {
        let queue = [true, true, false, true]
        for (limit, expected) in [(-1, 0), (0, 0), (1, 1), (2, 2), (4, 2), (Int.max, 2)] {
            XCTAssertEqual(BatchScheduler.eligibleAdmissionPrefixCount(
                queue, limit: limit, isEligible: { $0 }), expected)
        }
        XCTAssertEqual(BatchScheduler.eligibleAdmissionPrefixCount(
            [false, true], limit: 2, isEligible: { $0 }), 0)
        XCTAssertEqual(BatchScheduler.eligibleAdmissionPrefixCount(
            [Bool](), limit: 2, isEligible: { $0 }), 0)
        var visited = [Int]()
        XCTAssertEqual(BatchScheduler.eligibleAdmissionPrefixCount([0, 1, 2], limit: 1) {
            visited.append($0)
            return true
        }, 1)
        XCTAssertEqual(visited, [0], "Do not inspect beyond available capacity")
    }

    func testQwenDecodeGroupsRemainOptInAndExcludeMediaOrModelState() {
        for enabled in [false, true] {
            for type in [Qwen4ExpModel.self, Qwen4ExpVL.self] as [Any.Type] {
                XCTAssertEqual(BatchScheduler.supportsQwenDecodeGroups(
                    modelType: type, enabled: enabled), enabled)
            }
            XCTAssertFalse(BatchScheduler.supportsQwenDecodeGroups(
                modelType: Gemma4Model.self, enabled: enabled))
            for independent in [false, true] {
                for media in [false, true] {
                    for state in [false, true] {
                        XCTAssertEqual(BatchScheduler.permitsQwenDecodeGroup(
                            enabled: enabled, independentCaches: independent,
                            isMultimodal: media, hasModelState: state),
                            enabled && independent && !media && !state)
                    }
                }
            }
        }
        // Grouping eligible text rows does not qualify continuous VLM admission
        // or interleaved visual prefill; preserve these independent guards.
        XCTAssertTrue(BatchScheduler.requiresFixedDecodeCohorts(
            for: Qwen4ExpVL.self, continuousUniformGroups: true))
        XCTAssertFalse(BatchScheduler.supportsPrefillInterleave(
            modelType: Qwen4ExpVL.self, continuousGroups: true, ownsMTP: false, enabled: true))
    }

    func testPrefillInterleaveRequiresQualifiedTextARAdapterAndExplicitOptIn() {
        for enabled in [false, true] {
            for continuous in [false, true] {
                for mtp in [false, true] {
                    XCTAssertEqual(BatchScheduler.supportsPrefillInterleave(
                        modelType: Qwen4ExpModel.self, continuousGroups: continuous,
                        ownsMTP: mtp, enabled: enabled), enabled && continuous && !mtp)
                    XCTAssertFalse(BatchScheduler.supportsPrefillInterleave(
                        modelType: Gemma4Model.self, continuousGroups: continuous,
                        ownsMTP: mtp, enabled: enabled))
                    XCTAssertFalse(BatchScheduler.supportsPrefillInterleave(
                        modelType: Qwen4ExpVL.self, continuousGroups: continuous,
                        ownsMTP: mtp, enabled: enabled))
                }
            }
        }
    }

    func testVLMTextPrefillInterleaveAlsoRequiresQualifiedContinuousAdmission() {
        for enabled in [false, true] {
            for continuous in [false, true] {
                for mtp in [false, true] {
                    for admission in [false, true] {
                        XCTAssertEqual(BatchScheduler.supportsPrefillInterleave(
                            modelType: Qwen4ExpVL.self, continuousGroups: continuous,
                            ownsMTP: mtp, enabled: enabled, permitsVLMTextAdmission: admission),
                            enabled && continuous && !mtp && admission)
                        XCTAssertEqual(BatchScheduler.supportsPrefillInterleave(
                            modelType: Qwen4ExpModel.self, continuousGroups: continuous,
                            ownsMTP: mtp, enabled: enabled, permitsVLMTextAdmission: admission),
                            enabled && continuous && !mtp)
                        XCTAssertFalse(BatchScheduler.supportsPrefillInterleave(
                            modelType: Gemma4VLM.self, continuousGroups: continuous,
                            ownsMTP: mtp, enabled: enabled, permitsVLMTextAdmission: admission))
                    }
                }
            }
        }
    }

    func testContinuousAdmissionIsBoundedAndRespectsAvailableSlots() {
        XCTAssertEqual(BatchScheduler.continuousAdmissionLimit(
            maxConcurrent: 15, activeCount: 0, enabled: true), 15)
        XCTAssertEqual(BatchScheduler.continuousAdmissionLimit(
            maxConcurrent: 15, activeCount: 2, enabled: true), 1)
        XCTAssertEqual(BatchScheduler.continuousAdmissionLimit(
            maxConcurrent: 15, activeCount: 14, enabled: true), 1)
        XCTAssertEqual(BatchScheduler.continuousAdmissionLimit(
            maxConcurrent: 15, activeCount: 15, enabled: true), 0)
        XCTAssertEqual(BatchScheduler.continuousAdmissionLimit(
            maxConcurrent: 15, activeCount: 16, enabled: true), 0)
        XCTAssertEqual(BatchScheduler.continuousAdmissionLimit(
            maxConcurrent: 15, activeCount: 2, enabled: false), 15)
        XCTAssertEqual(BatchScheduler.continuousAdmissionLimit(
            maxConcurrent: 15, activeCount: 2, enabled: true, tokenBudget: 1024), 13)
        XCTAssertEqual(BatchScheduler.continuousAdmissionLimit(
            maxConcurrent: 15, activeCount: 15, enabled: true, tokenBudget: 1024), 0)
    }

    func testInitialQwenMTPBurstCollectsPartialPeerCohortsOnly() {
        for count in 0...5 {
            XCTAssertEqual(BatchScheduler.shouldCollectInitialBurst(
                requestCount: count, maxConcurrent: 4, allRequestsUseQwenMTP: true,
                admissionWindowNanoseconds: 8_000_000), count > 0 && count < 4)
            XCTAssertEqual(BatchScheduler.shouldCollectInitialBurst(
                requestCount: count, maxConcurrent: 4, allRequestsUseQwenMTP: false,
                admissionWindowNanoseconds: 8_000_000), count == 1)
        }
        XCTAssertFalse(BatchScheduler.shouldCollectInitialBurst(
            requestCount: 3, maxConcurrent: 4, allRequestsUseQwenMTP: true,
            admissionWindowNanoseconds: 0))
        XCTAssertFalse(BatchScheduler.shouldCollectInitialBurst(
            requestCount: 1, maxConcurrent: 1, allRequestsUseQwenMTP: true,
            admissionWindowNanoseconds: 8_000_000))
    }

    func testContinuousPrefillBudgetPreservesFIFOAndAllowsOversizedHeadProgress() {
        XCTAssertEqual(BatchScheduler.continuousAdmissionPrefixCount(
            estimatedTokenCounts: [], tokenBudget: 1024), 0)
        XCTAssertEqual(BatchScheduler.continuousAdmissionPrefixCount(
            estimatedTokenCounts: [1, 1, 1, 1], tokenBudget: 1024), 4)
        XCTAssertEqual(BatchScheduler.continuousAdmissionPrefixCount(
            estimatedTokenCounts: [128, 128, 256, 1024, 1], tokenBudget: 512), 3)
        // A cheap request cannot jump over a larger FIFO head.
        XCTAssertEqual(BatchScheduler.continuousAdmissionPrefixCount(
            estimatedTokenCounts: [1, 2048, 1], tokenBudget: 1024), 1)
        XCTAssertEqual(BatchScheduler.continuousAdmissionPrefixCount(
            estimatedTokenCounts: [8192, 1], tokenBudget: 1024), 1)
    }

    func testContinuousPrefillBudgetHandlesSingleModeAndIntegerBounds() {
        XCTAssertEqual(BatchScheduler.continuousAdmissionPrefixCount(
            estimatedTokenCounts: [1, 1, 1], tokenBudget: 1), 1)
        XCTAssertEqual(BatchScheduler.continuousAdmissionPrefixCount(
            estimatedTokenCounts: [0, -1, 1], tokenBudget: 2), 2)
        XCTAssertEqual(BatchScheduler.continuousAdmissionPrefixCount(
            estimatedTokenCounts: [Int.max, Int.max], tokenBudget: Int.max), 1)
        XCTAssertEqual(BatchScheduler.continuousAdmissionPrefixCount(
            estimatedTokenCounts: [0, 0], tokenBudget: Int.min), 1)
    }

    func testContinuousGroupAdmissionDoesNotDisableOtherModelsSafetyBarrier() {
        XCTAssertTrue(BatchScheduler.requiresFixedDecodeCohorts(for: Qwen4ExpModel.self))
        XCTAssertFalse(BatchScheduler.requiresFixedDecodeCohorts(
            for: Qwen4ExpModel.self, continuousUniformGroups: true))
        XCTAssertTrue(BatchScheduler.requiresFixedDecodeCohorts(
            for: Gemma4Model.self, continuousUniformGroups: true))
        XCTAssertTrue(BatchScheduler.requiresFixedDecodeCohorts(for: Qwen4ExpVL.self))
        XCTAssertTrue(BatchScheduler.requiresFixedDecodeCohorts(
            for: Qwen4ExpVL.self, continuousUniformGroups: true))
    }

    func testContinuousYieldExperimentPreservesDefaultAndOtherModelsSchedule() {
        for step in 1...128 {
            XCTAssertEqual(BatchScheduler.shouldYieldIndependentDecode(
                stepCount: step, continuousGroups: true), step.isMultiple(of: 64))
            for interval in [-1, 0, 1, 8, 16, 64, 128] {
                XCTAssertEqual(BatchScheduler.shouldYieldIndependentDecode(
                    stepCount: step, continuousGroups: false, interval: interval),
                    step.isMultiple(of: 64))
                XCTAssertEqual(BatchScheduler.shouldYieldIndependentDecode(
                    stepCount: step, continuousGroups: true, interval: interval),
                    step.isMultiple(of: min(64, max(1, interval))))
            }
        }
    }

    func testQwenMTPSchedulerRequiresExplicitOptInAndMatchingArchitecture() {
        for enabled in [false, true] {
            for hasGenerator in [false, true] {
                XCTAssertEqual(BatchScheduler.supportsQwenMTPScheduler(
                    modelType: Qwen4ExpModel.self, hasGenerator: hasGenerator, enabled: enabled),
                    enabled && hasGenerator)
                XCTAssertEqual(BatchScheduler.supportsQwenMTPScheduler(
                    modelType: Qwen4ExpVL.self, hasGenerator: hasGenerator, enabled: enabled),
                    enabled && hasGenerator)
                XCTAssertFalse(BatchScheduler.supportsQwenMTPScheduler(
                    modelType: Gemma4Model.self, hasGenerator: hasGenerator, enabled: enabled))
                XCTAssertFalse(BatchScheduler.supportsQwenMTPScheduler(
                    modelType: Gemma4VLM.self, hasGenerator: hasGenerator, enabled: enabled))
            }
        }
    }

    func testQwenMTPSessionRejectsPreparedMediaEvenWhenCallerRequestsMTP() {
        let text = LMInput.Text(tokens: MLXArray([1, 2, 3]))
        let image = LMInput.ProcessedImage(pixels: MLXArray.zeros([1, 3, 2, 2]))
        let video = LMInput.ProcessedVideo(pixels: MLXArray.zeros([1, 3, 2, 2]))
        for requested in [false, true] {
            XCTAssertEqual(BatchScheduler.canUseQwenMTPSession(
                requested: requested, input: LMInput(text: text)), requested)
            XCTAssertFalse(BatchScheduler.canUseQwenMTPSession(
                requested: requested, input: LMInput(text: text, image: image)))
            XCTAssertFalse(BatchScheduler.canUseQwenMTPSession(
                requested: requested, input: LMInput(text: text, video: video)))
            XCTAssertFalse(BatchScheduler.canUseQwenMTPSession(
                requested: requested, input: LMInput(text: text, image: image, video: video)))
        }
    }

    func testCompatibleGroupsUseActualOffsetsAndStableRowOrder() {
        let caches = [7, 9, 7, 8, 9].enumerated().map {
            [TestUniformDecodeCache(offset: $0.element, value: Float($0.offset)) as KVCache]
        }
        XCTAssertEqual(UniformDecodeGroup.compatibleIndices(caches), [[0, 2], [1, 4]])
    }

    func testCompatibleGroupsRejectUnknownEmptyBatchedAndDifferentGeometry() {
        let normal = TestUniformDecodeCache(offset: 7, value: 1)
        let batch = TestUniformDecodeCache(offset: 7, value: 2)
        batch.extendUniformBatch(with: normal)
        let empty = TestUniformDecodeCache(offset: 7, value: 3)
        empty.state = []
        let candidates: [[KVCache]] = [
            [normal], [batch], [empty], [], [TestOnlyArraysCache(size: 1)],
            [TestUniformDecodeCache(offset: 7, value: 4, width: 3)],
            [TestUniformDecodeCache(offset: 7, value: 5, dtype: .bfloat16)],
            [OtherUniformDecodeCache(offset: 7, value: 6)],
            [normal, normal],
        ]
        XCTAssertEqual(UniformDecodeGroup.compatibleIndices(candidates), [])
        XCTAssertNil(UniformDecodeGroup(slotIDs: [UUID(), UUID()], requestCaches: [[normal], [batch]]))
        XCTAssertNil(UniformDecodeGroup(slotIDs: [UUID(), UUID()], requestCaches: [[normal], [empty]]))
    }

    func testPersistentGroupFiltersRowsWithoutRestoringStaleCaches() throws {
        let ids = [UUID(), UUID(), UUID()]
        let originals = [10, 20, 30].map { [TestUniformDecodeCache(offset: 7, value: Float($0)) as KVCache] }
        let group = try XCTUnwrap(UniformDecodeGroup(slotIDs: ids, requestCaches: originals))
        XCTAssertEqual(group.caches[0].state[0].asArray(Float.self), [10, 10, 20, 20, 30, 30])
        let mergedCache = try XCTUnwrap(group.caches[0] as? TestUniformDecodeCache)
        mergedCache.state = [mergedCache.state[0] + 1]
        group.remove(ids[1])
        XCTAssertEqual(group.slotIDs, [ids[0], ids[2]])
        XCTAssertEqual(group.caches[0].state[0].asArray(Float.self), [11, 11, 31, 31])
        group.remove(ids[0])
        XCTAssertEqual(group.slotIDs, [ids[2]])
        XCTAssertEqual(group.caches[0].state[0].asArray(Float.self), [31, 31])
        group.remove(UUID())
        XCTAssertEqual(group.slotIDs, [ids[2]])
        group.remove(ids[2]) // empty group must not call a cache's nonempty filter
        XCTAssertTrue(group.slotIDs.isEmpty)
        XCTAssertEqual(originals[0][0].state[0].asArray(Float.self), [10, 10])
        XCTAssertEqual(originals[1][0].state[0].asArray(Float.self), [20, 20])
        XCTAssertEqual(originals[2][0].state[0].asArray(Float.self), [30, 30])
    }

    func testGroupConstructionRejectsMismatchedMembershipAndOffsets() {
        let cache = TestUniformDecodeCache(offset: 7, value: 1)
        let id = UUID()
        XCTAssertNil(UniformDecodeGroup(slotIDs: [], requestCaches: []))
        XCTAssertNil(UniformDecodeGroup(slotIDs: [id], requestCaches: [[cache]]))
        XCTAssertNil(UniformDecodeGroup(slotIDs: [id, id], requestCaches: [[cache], [cache]]))
        XCTAssertNil(UniformDecodeGroup(slotIDs: [id, UUID()], requestCaches: [[cache]]))
        XCTAssertNil(UniformDecodeGroup(slotIDs: [id, UUID()], requestCaches: [
            [cache], [TestUniformDecodeCache(offset: 8, value: 1)]]))
    }

    func testUniformCacheCohortUsesDenseDecodeForEqualTextOffsets() {
        XCTAssertFalse(BatchScheduler.requiresIndependentUniformCacheCohort(
            promptTokenCounts: [128, 128],
            hasMultimodalInput: false))
    }

    func testUniformCacheCohortRetainsNativeCachesForMixedOffsets() {
        XCTAssertTrue(BatchScheduler.requiresIndependentUniformCacheCohort(
            promptTokenCounts: [127, 128],
            hasMultimodalInput: false))
    }

    func testUniformCacheCohortRetainsNativeCachesForMultimodalPositions() {
        XCTAssertTrue(BatchScheduler.requiresIndependentUniformCacheCohort(
            promptTokenCounts: [128, 128],
            hasMultimodalInput: true))
    }

    func testMultimodalInputNeverUsesTextOnlyRecurrentReplayBoundary() {
        XCTAssertFalse(BatchScheduler.shouldCaptureReplayBoundary(
            prefixCacheEnabled: true,
            hasRecurrentLayers: true,
            isMultimodal: true,
            inputTokenCount: 128))
        XCTAssertTrue(BatchScheduler.shouldCaptureReplayBoundary(
            prefixCacheEnabled: true,
            hasRecurrentLayers: true,
            isMultimodal: false,
            inputTokenCount: 128))
    }

    func testHostTokenIDsRemainPairedWithCurrentPipelinedTensor() {
        for tokens in [[11, 12], [21, 22], [31, 32]] {
            let tensor = MLXArray(tokens).reshaped(2, 1)
            XCTAssertEqual(BatchScheduler.pairedHostTokenIDs(
                for: tensor,
                modelConsumesHostTokenIDs: true), tokens)
        }
        XCTAssertNil(BatchScheduler.pairedHostTokenIDs(
            for: MLXArray([41]).reshaped(1, 1),
            modelConsumesHostTokenIDs: false))
    }

    func testEstablishedArrayCachesRemainDenseBatchable() {
        XCTAssertTrue(BatchScheduler.supportsDenseBatchMerge(ArraysCache(size: 2)))
        XCTAssertTrue(BatchScheduler.supportsDenseBatchMerge(MambaCache()))
    }

    func testUnknownArrayCacheSubclassMustOptIntoBatching() {
        XCTAssertFalse(BatchScheduler.supportsDenseBatchMerge(
            TestOnlyArraysCache(size: 2)))
    }
}
