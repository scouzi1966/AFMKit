import XCTest
import AFMKitCore
@testable import AFMKitDwarfStar

final class AFMDwarfStarProviderTests: XCTestCase {
    func testTextStopPolicySuppressesDelimiterSplitAcrossChunks() {
        var buffer = ""

        let first = AFMDwarfStarTextStopPolicy.consume(
            buffer: &buffer,
            piece: "AFM_STREAM_PREFIX<AFM_STREAM_",
            stopSequences: ["<AFM_STREAM_STOP>"]
        )
        let second = AFMDwarfStarTextStopPolicy.consume(
            buffer: &buffer,
            piece: "STOP>trailing text",
            stopSequences: ["<AFM_STREAM_STOP>"]
        )

        XCTAssertEqual(first.visibleText, "AFM_STREAM_PREFIX")
        XCTAssertFalse(first.stopped)
        XCTAssertEqual(second.visibleText, "")
        XCTAssertTrue(second.stopped)
        XCTAssertEqual(buffer, "")
    }

    func testTextStopPolicyEmitsTextThatCannotBecomeStopPrefix() {
        var buffer = ""

        let result = AFMDwarfStarTextStopPolicy.consume(
            buffer: &buffer,
            piece: "ordinary output",
            stopSequences: ["<AFM_STOP>"]
        )

        XCTAssertEqual(result.visibleText, "ordinary output")
        XCTAssertFalse(result.stopped)
        XCTAssertEqual(buffer, "")
    }

    func testParsedOutputStopPolicySuppressesToolCallAfterSplitStop() throws {
        var parser = AFMDwarfStarToolCodec.StreamParser()
        var responseBuffer = ""
        let toolBlock = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"weather\"></｜DSML｜invoke></｜DSML｜tool_calls>"

        let first = AFMDwarfStarParsedOutputStopPolicy.consume(
            buffer: &responseBuffer,
            outputs: try parser.consume("AFM_PREFIXST"),
            stopSequences: ["STOP"]
        )
        let second = AFMDwarfStarParsedOutputStopPolicy.consume(
            buffer: &responseBuffer,
            outputs: try parser.consume("OP" + toolBlock),
            stopSequences: ["STOP"]
        )

        XCTAssertEqual(first, .init(outputs: [.text("AFM_PREFIX")], stopped: false))
        XCTAssertEqual(second, .init(outputs: [], stopped: true))
        XCTAssertEqual(responseBuffer, "")
    }

    func testParsedOutputStopPolicyFlushesPartialPrefixBeforeToolBoundary() throws {
        var parser = AFMDwarfStarToolCodec.StreamParser()
        var responseBuffer = ""
        let toolBlock = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"weather\"></｜DSML｜invoke></｜DSML｜tool_calls>"

        let first = AFMDwarfStarParsedOutputStopPolicy.consume(
            buffer: &responseBuffer,
            outputs: try parser.consume("AFM_PREFIXST"),
            stopSequences: ["START"]
        )
        let second = AFMDwarfStarParsedOutputStopPolicy.consume(
            buffer: &responseBuffer,
            outputs: try parser.consume(toolBlock),
            stopSequences: ["START"]
        )

        XCTAssertEqual(first.outputs, [.text("AFM_PREFIX")])
        XCTAssertEqual(
            second.outputs,
            [
                .text("ST"),
                .toolCalls([AFMToolCall(id: "call_1", name: "weather", arguments: "{}")]),
            ]
        )
        XCTAssertFalse(second.stopped)
        XCTAssertEqual(responseBuffer, "")
    }

    func testParsedOutputStopPolicyFlushesPartialPrefixBeforeReasoningBoundary() throws {
        var parser = AFMDwarfStarToolCodec.StreamParser()
        var responseBuffer = ""

        _ = AFMDwarfStarParsedOutputStopPolicy.consume(
            buffer: &responseBuffer,
            outputs: try parser.consume("AFM_PREFIXST"),
            stopSequences: ["START"]
        )
        let result = AFMDwarfStarParsedOutputStopPolicy.consume(
            buffer: &responseBuffer,
            outputs: try parser.consume("<think>private"),
            stopSequences: ["START"]
        )

        XCTAssertEqual(result.outputs, [.text("ST"), .reasoning("private")])
        XCTAssertFalse(result.stopped)
        XCTAssertEqual(responseBuffer, "")
    }

    func testProviderStatePublicationPreservesMutationOrderWhenObserverBlocks() async throws {
        let coordinator = AFMDwarfStarProviderStateCoordinator()
        let observer = BlockingProviderStateObserver()
        _ = coordinator.register(observer)
        XCTAssertTrue(observer.waitForUpdateCount(1))
        observer.blockNextUpdate()

        let reservationID = UUID()
        DispatchQueue.global().async {
            coordinator.admissionStarted(reservationID)
        }
        XCTAssertTrue(observer.waitUntilBlocked())
        coordinator.admissionFailed(reservationID)
        observer.resumeBlockedUpdate()
        XCTAssertTrue(observer.waitForUpdateCount(3))

        XCTAssertEqual(Array(observer.waitingRequestHistory.suffix(2)), [1, 0])
    }

    func testLeaseKeepsAdmissionAliveUntilReservationIsReleased() async throws {
        let coordinator = AFMDwarfStarProviderStateCoordinator()
        let observer = ProviderStateObserver()
        weak var weakAdmission: AFMDwarfStarGenerationAdmission?
        var lease: AFMGenerationLease?

        do {
            let admission = AFMDwarfStarGenerationAdmission(
                maximumConcurrentRequests: 1,
                telemetryObserver: observer,
                providerStateCoordinator: coordinator
            )
            weakAdmission = admission
            lease = try await admission.admitGeneration(timeout: .seconds(1))
        }

        XCTAssertNotNil(weakAdmission)
        lease?.release()
        lease = nil
        try await waitForProviderState(observer) {
            $0.runningRequests == 0 && $0.waitingRequests == 0
        }
        XCTAssertNil(weakAdmission)
    }

    func testCombinedProviderStateBroadcastsAdmissionAndSchedulerTransitions() async throws {
        let coordinator = AFMDwarfStarProviderStateCoordinator()
        let firstObserver = ProviderStateObserver()
        let secondObserver = ProviderStateObserver()
        let firstAdmission = AFMDwarfStarGenerationAdmission(
            maximumConcurrentRequests: 1,
            telemetryObserver: firstObserver,
            providerStateCoordinator: coordinator
        )
        let secondAdmission = AFMDwarfStarGenerationAdmission(
            maximumConcurrentRequests: 1,
            telemetryObserver: secondObserver,
            providerStateCoordinator: coordinator
        )

        let firstLease = try await firstAdmission.admitGeneration(timeout: .seconds(1))
        try await waitForProviderState(secondObserver) {
            $0.runningRequests == 1 && $0.waitingRequests == 0
        }

        coordinator.schedulerChanged(
            running: 1,
            waiting: 0,
            activeLogicalCachePositions: 12,
            logicalCacheCapacity: 100
        )
        try await waitForProviderState(firstObserver) {
            $0.runningRequests == 1 && $0.activeLogicalCachePositions == 12
        }

        let secondLease = try await secondAdmission.admitGeneration(timeout: .seconds(1))
        try await waitForProviderState(firstObserver) {
            $0.runningRequests == 2 && $0.waitingRequests == 0
        }
        try await waitForProviderState(secondObserver) {
            $0.runningRequests == 2 && $0.waitingRequests == 0
        }

        coordinator.schedulerChanged(
            running: 1,
            waiting: 1,
            activeLogicalCachePositions: 24,
            logicalCacheCapacity: 100
        )
        try await waitForProviderState(firstObserver) {
            $0.runningRequests == 1 && $0.waitingRequests == 1
        }
        try await waitForProviderState(secondObserver) {
            $0.runningRequests == 1 && $0.waitingRequests == 1
        }

        firstLease.release()
        secondLease.release()
        coordinator.schedulerChanged(
            running: 0,
            waiting: 0,
            activeLogicalCachePositions: 0,
            logicalCacheCapacity: 100
        )
        try await waitForProviderState(firstObserver) {
            $0.runningRequests == 0 && $0.waitingRequests == 0
        }
        try await waitForProviderState(secondObserver) {
            $0.runningRequests == 0 && $0.waitingRequests == 0
        }
    }

    func testAdmissionWaitersRemainVisibleInCombinedProviderState() async throws {
        let observer = ProviderStateObserver()
        let admission = AFMDwarfStarGenerationAdmission(
            maximumConcurrentRequests: 1,
            telemetryObserver: observer,
            providerStateCoordinator: AFMDwarfStarProviderStateCoordinator()
        )
        let occupied = try await admission.admitGeneration(timeout: .seconds(1))
        let waiter = Task {
            try await admission.admitGeneration(timeout: .seconds(1))
        }

        try await waitForProviderState(observer) {
            $0.runningRequests == 1 && $0.waitingRequests == 1
        }
        waiter.cancel()
        do {
            _ = try await waiter.value
            XCTFail("cancelled waiter must not be admitted")
        } catch let error as AFMGenerationAdmissionError {
            XCTAssertEqual(error, .cancelled)
        }
        try await waitForProviderState(observer) {
            $0.runningRequests == 1 && $0.waitingRequests == 0
        }

        occupied.release()
        try await waitForProviderState(observer) {
            $0.runningRequests == 0 && $0.waitingRequests == 0
        }
    }

    private func waitForProviderState(
        _ observer: ProviderStateObserver,
        predicate: (AFMInferenceProviderState) -> Bool
    ) async throws {
        let deadline = ContinuousClock.now + .seconds(1)
        while ContinuousClock.now < deadline {
            if predicate(observer.latestState) { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("provider state did not reach expected value")
    }

    func testLeaseOwnerReleasesWhenModelLifetimeEnds() async throws {
        let leaseID = UUID()
        let releases = LeaseReleaseProbe()
        weak var weakLease: AFMDwarfStarRuntimeLease?

        do {
            let lease = AFMDwarfStarRuntimeLease(id: leaseID) { releasedID in
                await releases.record(releasedID)
            }
            weakLease = lease
            XCTAssertNotNil(weakLease)
        }

        for _ in 0..<100 {
            if !(await releases.ids()).isEmpty { break }
            try await Task.sleep(for: .milliseconds(1))
        }
        XCTAssertNil(weakLease)
        let releasedIDs = await releases.ids()
        XCTAssertEqual(releasedIDs, [leaseID])
    }

    func testTwoExecutorsRetainSharedRuntimeUntilFinalLeaseEnds() {
        let firstExecutor = UUID()
        let secondExecutor = UUID()
        let identity = "deepseek.gguf|32768|metal"
        var leases = AFMDwarfStarRuntimeLeaseRegistry()

        XCTAssertEqual(
            leases.acquisition(
                for: firstExecutor,
                runtimeIdentity: identity,
                residentRuntimeMatches: false
            ),
            .loadRuntime
        )
        leases.commit(leaseID: firstExecutor, runtimeIdentity: identity)
        XCTAssertEqual(
            leases.acquisition(
                for: secondExecutor,
                runtimeIdentity: identity,
                residentRuntimeMatches: true
            ),
            .shareResident
        )
        leases.commit(leaseID: secondExecutor, runtimeIdentity: identity)

        XCTAssertFalse(leases.release(leaseID: firstExecutor))
        XCTAssertEqual(leases.leaseIDs, [secondExecutor])
        XCTAssertFalse(leases.release(leaseID: firstExecutor))
        XCTAssertEqual(leases.runtimeIdentity, identity)
        XCTAssertEqual(leases.leaseIDs, [secondExecutor])
        XCTAssertTrue(leases.release(leaseID: secondExecutor))
        XCTAssertTrue(leases.leaseIDs.isEmpty)
        XCTAssertNil(leases.runtimeIdentity)
    }

    func testLoadedExecutorCannotBeRetargetedOrDisplaced() {
        let activeExecutor = UUID()
        let competingExecutor = UUID()
        var leases = AFMDwarfStarRuntimeLeaseRegistry()
        leases.commit(leaseID: activeExecutor, runtimeIdentity: "model-a")

        XCTAssertEqual(
            leases.acquisition(
                for: activeExecutor,
                runtimeIdentity: "model-b",
                residentRuntimeMatches: false
            ),
            .blockedByDifferentRuntime
        )
        XCTAssertEqual(
            leases.acquisition(
                for: competingExecutor,
                runtimeIdentity: "model-b",
                residentRuntimeMatches: false
            ),
            .blockedByDifferentRuntime
        )
    }

    func testProviderContractDescribesInProcessDeviceRuntime() {
        let descriptor = AFMDwarfStarProviderFactory().descriptor

        XCTAssertEqual(descriptor.id, "dwarfstar")
        XCTAssertEqual(descriptor.privacyBoundary, .device)
        XCTAssertEqual(descriptor.metadata["runtime"], .string("in-process-ds4"))
        XCTAssertEqual(descriptor.metadata["execution"], .string("fixed-metal-schedule"))
        XCTAssertEqual(descriptor.metadata["checkpointFormat"], .string("native-gguf"))
        XCTAssertTrue(descriptor.configurationKeys.contains("modelPath"))
        XCTAssertTrue(descriptor.configurationKeys.contains("enablePrefixCaching"))
        XCTAssertTrue(descriptor.configurationKeys.contains("maxConcurrent"))
        XCTAssertTrue(descriptor.configurationKeys.contains("dsparkSupportPath"))
        XCTAssertTrue(descriptor.configurationKeys.contains("dsparkDraftTokens"))
        XCTAssertTrue(descriptor.configurationKeys.contains("dsparkConfidenceThreshold"))
        XCTAssertTrue(descriptor.configurationKeys.contains("dsparkStrict"))
        XCTAssertFalse(descriptor.configurationKeys.contains("templateGGUF"))
        XCTAssertFalse(descriptor.configurationKeys.contains("projectionMetadataPath"))
        XCTAssertFalse(descriptor.configurationKeys.contains("externalMapGGUF"))
    }

    func testBundledMetalRuntimeContainsEveryRequiredSource() throws {
        let root = try XCTUnwrap(AFMDwarfStarRuntime.metalSourceDirectory)
        let requiredSources = [
            "flash_attn.metal", "dense.metal", "moe.metal", "dsv4_hc.metal",
            "unary.metal", "dsv4_kv.metal", "dsv4_rope.metal", "dsv4_misc.metal",
            "argsort.metal", "cpy.metal", "concat.metal", "get_rows.metal",
            "sum_rows.metal", "softmax.metal", "repeat.metal", "glu.metal",
            "norm.metal", "bin.metal", "set_rows.metal",
        ]

        for source in requiredSources {
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: root.appendingPathComponent(source).path),
                "missing bundled DwarfStar Metal source \(source)"
            )
        }
    }

    func testMissingModelIsUnavailableWithoutLoadingRuntime() async {
        let model = AFMDwarfStarModel(
            modelID: "missing",
            modelPath: "/path/that/does/not/exist.gguf",
            configuration: AFMDwarfStarRuntimeConfiguration(
                contextWindow: 32_768,
                prefillChunk: 0,
                powerPercent: 100,
                dsparkSupportPath: nil,
                dsparkDraftTokens: 5,
                dsparkConfidenceThreshold: 0.7,
                dsparkStrict: false,
                enablePrefixCaching: false,
                maxConcurrent: 1
            )
        )

        let availability = await model.availability()
        XCTAssertFalse(availability.isAvailable)
        XCTAssertEqual(model.descriptor.providerID, "dwarfstar")
        XCTAssertTrue(model.descriptor.capabilities.contains(.reasoning))
        XCTAssertTrue(model.descriptor.capabilities.contains(.streaming))
        XCTAssertTrue(model.descriptor.capabilities.contains(.toolCalling))
    }

    func testModelDescriptorPublishesResidentSessionConfiguration() {
        let model = AFMDwarfStarModel(
            modelID: "configured",
            modelPath: "/missing.gguf",
            configuration: AFMDwarfStarRuntimeConfiguration(
                contextWindow: 32_768,
                prefillChunk: 0,
                powerPercent: 100,
                dsparkSupportPath: "/support.gguf",
                dsparkDraftTokens: 8,
                dsparkConfidenceThreshold: 0.9,
                dsparkStrict: true,
                enablePrefixCaching: true,
                maxConcurrent: 4
            )
        )

        XCTAssertEqual(model.descriptor.metadata["enablePrefixCaching"], .bool(true))
        XCTAssertEqual(model.descriptor.metadata["checkpointFormat"], .string("native-gguf"))
        XCTAssertEqual(model.descriptor.metadata["maxConcurrent"], .integer(4))
        XCTAssertEqual(model.descriptor.metadata["dsparkEnabled"], .bool(true))
        XCTAssertEqual(model.descriptor.metadata["dsparkDraftTokens"], .integer(8))
        XCTAssertEqual(model.descriptor.metadata["dsparkConfidenceThreshold"], .number(0.9))
        XCTAssertEqual(model.descriptor.metadata["dsparkStrict"], .bool(true))
    }

    func testReasoningModeDefaultsToChatWithoutThinkingControls() {
        XCTAssertEqual(AFMDwarfStarReasoningMode.resolve(metadata: [:]), .chat)
    }

    func testReasoningModeUsesOfficialReasoningEffort() {
        XCTAssertEqual(
            AFMDwarfStarReasoningMode.resolve(metadata: [
                "chatTemplateKwargs": .object(["reasoning_effort": .string("max")])
            ]),
            .max)
    }

    func testReasoningModeTreatsEnableThinkingAsLowEffort() {
        XCTAssertEqual(
            AFMDwarfStarReasoningMode.resolve(metadata: [
                "chatTemplateKwargs": .object(["enable_thinking": .bool(true)])
            ]),
            .low)
    }

    func testNoThinkingOverridesReasoningEffort() {
        XCTAssertEqual(
            AFMDwarfStarReasoningMode.resolve(metadata: [
                "chatTemplateKwargs": .object([
                    "enable_thinking": .bool(false),
                    "reasoning_effort": .string("max")
                ])
            ]),
            .chat)
    }

    func testReasoningModesMapToNativeDwarfStarModes() {
        XCTAssertEqual(AFMDwarfStarReasoningMode.chat.thinkMode.rawValue, 0)
        XCTAssertEqual(AFMDwarfStarReasoningMode.low.thinkMode.rawValue, 1)
        XCTAssertEqual(AFMDwarfStarReasoningMode.high.thinkMode.rawValue, 1)
        XCTAssertEqual(AFMDwarfStarReasoningMode.max.thinkMode.rawValue, 2)
    }

    func testSlotPolicyUsesFirstAvailableSlotWithoutPrefixCaching() {
        XCTAssertEqual(
            AFMDwarfStarSlotPolicy.bestSlot(
                commonPrefixes: [nil, 12, 30],
                prefixCachingEnabled: false),
            1)
    }

    func testSlotPolicyPrefersLongestReusablePrefixWithStableTieBreak() {
        XCTAssertEqual(
            AFMDwarfStarSlotPolicy.bestSlot(
                commonPrefixes: [12, nil, 30, 30],
                prefixCachingEnabled: true),
            2)
    }

    func testSchedulerUsesLargePrefillQuantumWhenNoDecodeIsActive() {
        XCTAssertEqual(
            AFMDwarfStarSchedulingPolicy.prefillQuantum(activeDecodeCount: 0),
            2_048)
    }

    func testSchedulerUsesBoundedPrefillQuantumDuringContinuousDecode() {
        XCTAssertEqual(
            AFMDwarfStarSchedulingPolicy.prefillQuantum(activeDecodeCount: 3),
            128)
    }

    func testSchedulerEstablishesCheckpointBeforeMixedPrefill() {
        XCTAssertFalse(
            AFMDwarfStarSchedulingPolicy.canMixPrefill(
                currentPosition: 0, activeDecodeCount: 3))
        XCTAssertTrue(
            AFMDwarfStarSchedulingPolicy.canMixPrefill(
                currentPosition: 128, activeDecodeCount: 3))
        XCTAssertFalse(
            AFMDwarfStarSchedulingPolicy.canMixPrefill(
                currentPosition: 128, activeDecodeCount: 0))
    }

    func testDSparkAvailabilityUsesGeneralizedDraftDepth() {
        XCTAssertFalse(
            AFMDwarfStarSpeculativePolicy.isAvailable(
                requested: false,
                draftTokenCount: 5))
        XCTAssertFalse(
            AFMDwarfStarSpeculativePolicy.isAvailable(
                requested: true,
                draftTokenCount: 0))
        XCTAssertTrue(
            AFMDwarfStarSpeculativePolicy.isAvailable(
                requested: true,
                draftTokenCount: 5))
    }

    func testSchedulerRotatesAcrossWaitingPrefills() {
        XCTAssertEqual(
            AFMDwarfStarSchedulingPolicy.nextPrefillSlot(
                lastSlot: 0,
                waiting: [true, true, false, true]),
            1)
        XCTAssertEqual(
            AFMDwarfStarSchedulingPolicy.nextPrefillSlot(
                lastSlot: 1,
                waiting: [true, true, false, true]),
            3)
        XCTAssertEqual(
            AFMDwarfStarSchedulingPolicy.nextPrefillSlot(
                lastSlot: 3,
                waiting: [true, true, false, true]),
            0)
    }

    func testSchedulerReturnsNilWhenNoPromptNeedsPrefill() {
        XCTAssertNil(
            AFMDwarfStarSchedulingPolicy.nextPrefillSlot(
                lastSlot: 2,
                waiting: [false, false, false]))
    }

    func testPrefixCacheCheckpointIdentitySeparatesModelRevisions() {
        let original = AFMDwarfStarPrefixCachePolicy.checkpointKey(
            path: "/models/deepseek.gguf", size: 100, modified: 10)
        XCTAssertEqual(
            original,
            AFMDwarfStarPrefixCachePolicy.checkpointKey(
                path: "/models/deepseek.gguf", size: 100, modified: 10))
        XCTAssertNotEqual(
            original,
            AFMDwarfStarPrefixCachePolicy.checkpointKey(
                path: "/models/deepseek.gguf", size: 101, modified: 10))
        XCTAssertNotEqual(
            original,
            AFMDwarfStarPrefixCachePolicy.checkpointKey(
                path: "/models/deepseek.gguf", size: 100, modified: 11))
    }

    func testPrefixCacheBudgetUsesSafeDefaultAndEnvironmentOverride() {
        XCTAssertEqual(AFMDwarfStarPrefixCachePolicy.budgetMB(environment: [:]), 4_096)
        XCTAssertEqual(
            AFMDwarfStarPrefixCachePolicy.budgetMB(
                environment: ["AFM_DWARFSTAR_PREFIX_CACHE_MB": "8192"]),
            8_192)
        XCTAssertEqual(
            AFMDwarfStarPrefixCachePolicy.budgetMB(
                environment: ["AFM_DWARFSTAR_PREFIX_CACHE_MB": "0"]),
            4_096)
    }

    func testToolPromptPublishesSchemasInDeepSeekDSMLFormat() throws {
        let prompt = try AFMDwarfStarToolCodec.systemPrompt(for: [
            AFMToolDefinition(
                name: "weather",
                description: "Look up weather.",
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "city": .object(["type": .string("string")])
                    ])
                ])
            )
        ])

        XCTAssertTrue(prompt.contains("<｜DSML｜tool_calls>"))
        XCTAssertTrue(prompt.contains("\"name\":\"weather\""))
        XCTAssertTrue(prompt.contains("\"city\""))
    }

    func testToolParserHandlesSplitMarkersAndParallelCalls() throws {
        var parser = AFMDwarfStarToolCodec.StreamParser()
        XCTAssertEqual(try parser.consume("I will check. <｜DSML｜tool_"), [.text("I will check. ")])
        XCTAssertEqual(try parser.consume("calls>\n<｜DSML｜invoke name=\"weather\">\n"), [])
        XCTAssertEqual(
            try parser.consume(
                "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n"
                    + "</｜DSML｜invoke>\n<｜DSML｜invoke name=\"clock\">\n"
                    + "<｜DSML｜parameter name=\"offset\" string=\"false\">-4</｜DSML｜parameter>\n"
                    + "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
            ),
            [
                .toolCalls([
                    AFMToolCall(
                        id: "call_1",
                        name: "weather",
                        arguments: "{\"city\":\"Paris\"}"
                    ),
                    AFMToolCall(
                        id: "call_2",
                        name: "clock",
                        arguments: "{\"offset\":-4}"
                    )
                ])
            ]
        )
    }

    func testToolMarkupInsideReasoningIsNotExecuted() throws {
        var parser = AFMDwarfStarToolCodec.StreamParser(startsInReasoning: true)
        let fakeCall = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"unsafe\"></｜DSML｜invoke></｜DSML｜tool_calls>"

        XCTAssertEqual(
            try parser.consume("consider \(fakeCall)</think>answer"),
            [.reasoning("consider \(fakeCall)"), .text("answer")]
        )
    }

    func testCompletedToolCallTerminatesParserAndDiscardsTrailingOutput() throws {
        var parser = AFMDwarfStarToolCodec.StreamParser()
        let call = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"weather\"></｜DSML｜invoke></｜DSML｜tool_calls>"

        let outputs = try parser.consume(call + "must not become response text")
        guard case .toolCalls(let calls) = try XCTUnwrap(outputs.first) else {
            return XCTFail("Expected a completed tool call")
        }
        XCTAssertEqual(calls.map(\.name), ["weather"])
        XCTAssertEqual(outputs.count, 1)
        XCTAssertEqual(try parser.consume("also ignored"), [])
        XCTAssertEqual(parser.finish(), [])
    }

    func testReasoningMarkersCanSpanChunks() throws {
        var parser = AFMDwarfStarToolCodec.StreamParser(startsInReasoning: true)

        XCTAssertEqual(try parser.consume("analysis</thi"), [.reasoning("analysis")])
        XCTAssertEqual(try parser.consume("nk>final"), [.text("final")])
    }

    func testResponseCanEnterAndLeaveExplicitReasoning() throws {
        var parser = AFMDwarfStarToolCodec.StreamParser()

        XCTAssertEqual(
            try parser.consume("before<think>private</think>after"),
            [.text("before"), .reasoning("private"), .text("after")]
        )
    }

    func testAssistantToolCallsRoundTripThroughDSML() throws {
        let message = AFMMessage(
            role: .assistant,
            content: [.text("Checking")],
            toolCalls: [
                AFMToolCall(
                    id: "original",
                    name: "weather",
                    arguments: "{\"city\":\"Montréal\",\"days\":2}"
                )
            ]
        )
        let rendered = try AFMDwarfStarToolCodec.assistantContent(for: message)

        XCTAssertTrue(rendered.hasPrefix("Checking\n\n<｜DSML｜tool_calls>"))
        XCTAssertTrue(rendered.contains("name=\"weather\""))
        XCTAssertTrue(rendered.contains("name=\"city\" string=\"true\">Montréal"))
        XCTAssertTrue(rendered.contains("name=\"days\" string=\"false\">2"))
    }

    func testAssistantToolReplayUsesCanonicalSeparatorAndBoundary() throws {
        let message = AFMMessage(
            role: .assistant,
            content: [.text("Checking")],
            toolCalls: [
                AFMToolCall(id: "call", name: "weather", arguments: "{\"city\":\"Paris\"}")
            ]
        )

        let suffix = try AFMDwarfStarToolCodec.assistantReplaySuffix(for: message)
        XCTAssertTrue(suffix.hasPrefix("\n\n<｜DSML｜tool_calls>"))
        XCTAssertTrue(suffix.hasSuffix("<｜end▁of▁sentence｜>"))
    }

    func testAssistantTextReplayEndsWithConversationBoundary() throws {
        let message = AFMMessage(role: .assistant, content: [.text("Done")])
        XCTAssertEqual(
            try AFMDwarfStarToolCodec.assistantReplaySuffix(for: message),
            "<｜end▁of▁sentence｜>"
        )
    }

}

private final class ProviderStateObserver: AFMInferenceTelemetryObserving, @unchecked Sendable {
    private let lock = NSLock()
    private var state = AFMInferenceProviderState(runningRequests: 0, waitingRequests: 0)

    var latestState: AFMInferenceProviderState { lock.withLock { state } }

    func requestAccepted(at timestamp: Double) -> AFMInferenceRequestToken {
        AFMInferenceRequestToken()
    }
    func requestStarted(_ token: AFMInferenceRequestToken, at timestamp: Double) {}
    func outputToken(_ token: AFMInferenceRequestToken, at timestamp: Double) {}
    func prefixCacheObserved(queriedTokens: Int, hitTokens: Int) {}
    func speculativeRound(draftTokens: Int, acceptedTokens: Int) {}
    func preemptionObserved() {}
    func updateProviderState(_ state: AFMInferenceProviderState) {
        lock.withLock { self.state = state }
    }
    func requestFinished(
        _ token: AFMInferenceRequestToken,
        observation: AFMInferenceRequestFinishObservation
    ) -> Bool { true }
    func requestFailed(
        _ token: AFMInferenceRequestToken,
        reason: AFMInferenceFailureReason,
        at timestamp: Double
    ) -> Bool { true }
}

private final class BlockingProviderStateObserver:
    AFMInferenceTelemetryObserving,
    @unchecked Sendable
{
    private struct State {
        var waitingRequestHistory: [Int] = []
        var shouldBlockNext = false
    }

    private let lock = NSLock()
    private var state = State()
    private let blocked = DispatchSemaphore(value: 0)
    private let resume = DispatchSemaphore(value: 0)

    var waitingRequestHistory: [Int] {
        lock.withLock { state.waitingRequestHistory }
    }

    func blockNextUpdate() {
        lock.withLock { state.shouldBlockNext = true }
    }

    func waitUntilBlocked() -> Bool {
        blocked.wait(timeout: .now() + 1) == .success
    }

    func resumeBlockedUpdate() {
        resume.signal()
    }

    func waitForUpdateCount(_ count: Int) -> Bool {
        let deadline = Date().addingTimeInterval(1)
        while Date() < deadline {
            if lock.withLock({ state.waitingRequestHistory.count >= count }) {
                return true
            }
            Thread.sleep(forTimeInterval: 0.005)
        }
        return false
    }

    func requestAccepted(at timestamp: Double) -> AFMInferenceRequestToken {
        AFMInferenceRequestToken()
    }
    func requestStarted(_ token: AFMInferenceRequestToken, at timestamp: Double) {}
    func outputToken(_ token: AFMInferenceRequestToken, at timestamp: Double) {}
    func prefixCacheObserved(queriedTokens: Int, hitTokens: Int) {}
    func speculativeRound(draftTokens: Int, acceptedTokens: Int) {}
    func preemptionObserved() {}
    func updateProviderState(_ providerState: AFMInferenceProviderState) {
        let shouldBlock = lock.withLock { () -> Bool in
            let shouldBlock = state.shouldBlockNext
            state.shouldBlockNext = false
            return shouldBlock
        }
        if shouldBlock {
            blocked.signal()
            _ = resume.wait(timeout: .now() + 1)
        }
        lock.withLock {
            state.waitingRequestHistory.append(providerState.waitingRequests)
        }
    }
    func requestFinished(
        _ token: AFMInferenceRequestToken,
        observation: AFMInferenceRequestFinishObservation
    ) -> Bool { true }
    func requestFailed(
        _ token: AFMInferenceRequestToken,
        reason: AFMInferenceFailureReason,
        at timestamp: Double
    ) -> Bool { true }
}

private actor LeaseReleaseProbe {
    private var releasedIDs: [UUID] = []

    func record(_ id: UUID) {
        releasedIDs.append(id)
    }

    func ids() -> [UUID] {
        releasedIDs
    }
}
