import Foundation
import XCTest
import AFMKitCore
@testable import AFMKitSplash

final class SplashProviderTests: XCTestCase, @unchecked Sendable {
    private struct FixtureTokenizer: SplashTokenizing {
        func prompt(_ request: AFMRequest) throws -> [Int] { [7, 8] }
        func decode(_ tokens: [Int]) -> String { String(tokens.compactMap(UnicodeScalar.init).map(Character.init)) }
        func encode(_ text: String) -> [Int] { text.unicodeScalars.map { Int($0.value) } }
    }

    private var directory: URL!
    override func setUpWithError() throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(".build-splash-tests")
        directory = root.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("engine"), withIntermediateDirectories: true)
        let fixture = try XCTUnwrap(Bundle.module.url(forResource: "native-peer", withExtension: "py", subdirectory: "Fixtures"))
        let executable = directory.appendingPathComponent("engine/splash")
        try FileManager.default.copyItem(at: fixture, to: executable)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)
    }
    override func tearDownWithError() throws { try FileManager.default.removeItem(at: directory) }

    private func model(_ mode: String = "normal", timeout: Double = 3) -> AFMSplashModel {
        var config = AFMSplashConfiguration(runtime: .init(root: directory), modelDirectory: directory.appendingPathComponent(mode))
        config.requestTimeout = timeout
        config.startupTimeout = timeout
        return AFMSplashModel(id: "fixture", configuration: config, loadTokenizer: { _ in FixtureTokenizer() }, validateRuntime: {})
    }
    private var request: AFMRequest { .init(messages: [.init(role: .user, text: "fixture")]) }

    func testPinnedReleaseIdentity() throws {
        let pin = try AFMSplashRelease.pinned()
        XCTAssertEqual(pin.version, "1.0")
        XCTAssertEqual(pin.revision, "c675ed23e6942b5353961246e68b08cd63fb4ee9")
        XCTAssertEqual(pin.protocolVersion, Int(SplashWire.version))
    }

    func testFindsHomebrewRuntimeAfterResolvingExecutableSymlink() throws {
        let cellar = directory.appendingPathComponent("Cellar/afm/version")
        let bin = cellar.appendingPathComponent("bin/afm")
        let runtime = cellar.appendingPathComponent("libexec/splash-runtime")
        try FileManager.default.createDirectory(at: bin.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: runtime, withIntermediateDirectories: true)
        try Data().write(to: bin)
        try Data("{}".utf8).write(to: runtime.appendingPathComponent("release.json"))
        let link = directory.appendingPathComponent("afm-link")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: bin)
        XCTAssertEqual(try AFMSplashRuntime.bundled(beside: link.path).root.path, runtime.path)
    }

    func testLocalSwiftTokenizerUsesPackageTemplateWithoutThinking() async throws {
        let tokenizerDirectory = directory.appendingPathComponent("tokenizer")
        try FileManager.default.createDirectory(at: tokenizerDirectory, withIntermediateDirectories: true)
        try Data(#"{"tokenizer_class":"Qwen2Tokenizer","unk_token":"[UNK]"}"#.utf8)
            .write(to: tokenizerDirectory.appendingPathComponent("tokenizer_config.json"))
        try Data(#"{"version":"1.0","model":{"type":"BPE","vocab":{"A":0,"B":1,"C":2,"[UNK]":3},"merges":[],"unk_token":"[UNK]"},"decoder":{"type":"ByteLevel"}}"#.utf8)
            .write(to: tokenizerDirectory.appendingPathComponent("tokenizer.json"))
        try Data("{{ messages[0]['content'] }}{% if add_generation_prompt %}B{% endif %}{% if enable_thinking %}C{% endif %}".utf8)
            .write(to: tokenizerDirectory.appendingPathComponent("chat_template.jinja"))
        let tokenizer = try await SplashTokenizer.load(directory)
        XCTAssertEqual(try tokenizer.prompt(.init(messages: [.init(role: .user, text: "A")])), [0, 1])
        XCTAssertEqual(tokenizer.decode([0, 1]), "AB")
    }

    func testRequestWireLayoutAndSampling() throws {
        let data = try SplashWire.request(id: 5, tokens: [7, 8], options: .init(temperature: 0.5, maximumResponseTokens: 4, topP: 0.9, topK: 8, seed: 42), timeout: 2)
        // Request frames are outbound only; the inbound parser rejects them.
        var header = SplashWire.Reader(data: Data(data.prefix(24)))
        XCTAssertEqual(try header.bytes(4), Data("SPLH".utf8))
        XCTAssertEqual(try header.integer(UInt16.self), 5)
        XCTAssertEqual(try header.integer(UInt16.self), 24)
        XCTAssertEqual(try header.integer(UInt16.self), 1)
        XCTAssertEqual(try header.integer(UInt16.self), 0)
        XCTAssertEqual(try header.integer(UInt64.self), 68)
        var r = SplashWire.Reader(data: Data(data.dropFirst(24)))
        XCTAssertEqual(try r.integer(UInt64.self), 5)
        XCTAssertEqual(try r.integer(UInt8.self), 1)
        XCTAssertEqual(try r.integer(UInt8.self), 1)
        XCTAssertEqual(try r.integer(UInt8.self), 0)
        _ = try r.integer(UInt64.self)
        XCTAssertEqual(try r.integer(UInt64.self), 2_000_000)
        XCTAssertEqual(try r.integer(UInt32.self), 4)
        XCTAssertEqual(try r.integer(UInt32.self), 2)
        XCTAssertEqual(try r.integer(UInt32.self), 0)
        XCTAssertEqual(Float(bitPattern: try r.integer(UInt32.self)), 0.5)
        XCTAssertEqual(Float(bitPattern: try r.integer(UInt32.self)), 0.9)
        XCTAssertEqual(try r.integer(UInt32.self), 8)
        XCTAssertEqual(try r.integer(UInt64.self), 42)
        XCTAssertEqual(try r.integer(UInt8.self), 0)
        XCTAssertEqual(try r.integer(UInt32.self), 7)
        XCTAssertEqual(try r.integer(UInt32.self), 8)
        try r.end()
    }

    func testRejectsInvalidWireHeaders() throws {
        let frame = Data(SplashWire.frame(type: 0x100, payload: Data(count: 24)).prefix(24))
        XCTAssertNoThrow(try SplashWire.header(frame))
        for index in [0, 4, 6, 10, 20] {
            var invalid = frame
            invalid[index] = 255
            XCTAssertThrowsError(try SplashWire.header(invalid))
        }
        XCTAssertThrowsError(try SplashWire.header(Data(frame.prefix(10))))
    }

    func testRejectsInvalidSamplingAndUnsupportedCapabilities() throws {
        XCTAssertThrowsError(try SplashWire.request(id: 1, tokens: [1], options: .init(topK: 33), timeout: 1))
        XCTAssertThrowsError(try SplashWire.request(id: 1, tokens: [-1], options: .init(), timeout: 1))
        var request = request
        request.options.reasoningEnabled = true
        XCTAssertThrowsError(try AFMSplashModel.validate(request))
        request.options.reasoningEnabled = false
        request.messages[0].content = [.data(mimeType: "image/png", value: Data())]
        XCTAssertThrowsError(try AFMSplashModel.validate(request))
    }

    func testFactoryRegistersWithoutMLXOrServer() throws {
        let registry = AFMProviderRegistry()
        try registry.register(AFMSplashProviderFactory())
        let model = try registry.makeModel(providerID: "splash", modelID: "fixture", configuration: .init(values: ["modelPath": .string("/unused"), "runtimePath": .string(directory.path)]))
        XCTAssertEqual(model.descriptor.providerID, "splash")
        XCTAssertEqual(model.descriptor.privacyBoundary, .device)
        XCTAssertFalse(model.descriptor.capabilities.contains(.toolCalling))
        XCTAssertThrowsError(try registry.makeModel(providerID: "splash", modelID: "fixture"))
    }

    func testFragmentedNativeFramesAndRepeatedRequests() async throws {
        let model = model()
        for _ in 0..<2 {
            let response = try await model.respond(to: request)
            XCTAssertEqual(response.text, "AB")
            XCTAssertEqual(response.usage.inputTokens, 2)
            XCTAssertEqual(response.usage.outputTokens, 2)
            XCTAssertEqual(response.finishReason, .stop)
        }
        await model.unload()
        let response = try await model.respond(to: request)
        XCTAssertEqual(response.text, "AB")
        await model.unload()
    }

    func testStreamsIncrementalText() async throws {
        let model = model()
        var deltas: [String] = []
        for try await event in model.streamResponse(to: request) {
            if case .responseText(_, let text, _) = event { deltas.append(text) }
        }
        XCTAssertEqual(deltas, ["A", "B"])
        await model.unload()
    }

    func testEngineErrorAndBadRequestIDFailCleanly() async throws {
        for mode in ["error", "bad-id", "truncated"] {
            let model = model(mode)
            do { _ = try await model.respond(to: request); XCTFail("Expected \(mode) failure") }
            catch { XCTAssertTrue(error.localizedDescription.contains("Splash")) }
            await model.unload()
        }
    }

    func testStartupAndGenerationTimeouts() async throws {
        for mode in ["startup-timeout", "wait"] {
            let model = model(mode, timeout: 0.2)
            do { _ = try await model.respond(to: request); XCTFail("Expected timeout") }
            catch { XCTAssertTrue(error.localizedDescription.contains("timed out"), "\(error)") }
            await model.unload()
        }
    }

    func testCancellationTerminatesPendingGeneration() async throws {
        let model = model("wait")
        _ = try await model.load()
        let request = request
        let task = Task { try await model.respond(to: request) }
        try await Task.sleep(for: .milliseconds(100))
        task.cancel()
        do { _ = try await task.value; XCTFail("Expected cancellation") }
        catch is CancellationError {} catch { XCTFail("Unexpected cancellation error: \(error)") }
        await model.unload()
    }
}
