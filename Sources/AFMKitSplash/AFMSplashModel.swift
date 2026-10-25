import Foundation
import AFMKitCore

public struct AFMSplashConfiguration: Sendable {
    public var runtime: AFMSplashRuntime
    public var modelDirectory: URL
    public var maximumContext: Int?
    public var maximumMemoryBytes: UInt64?
    public var startupTimeout: TimeInterval = 120
    public var requestTimeout: TimeInterval = 600

    public init(runtime: AFMSplashRuntime, modelDirectory: URL, maximumContext: Int? = nil, maximumMemoryBytes: UInt64? = nil) {
        self.runtime = runtime
        self.modelDirectory = modelDirectory
        self.maximumContext = maximumContext
        self.maximumMemoryBytes = maximumMemoryBytes
    }

    func validate() throws {
        guard startupTimeout.isFinite, startupTimeout > 0, startupTimeout <= 3600,
              requestTimeout.isFinite, requestTimeout > 0, requestTimeout <= 86400,
              maximumContext == nil || (1...SplashWire.maximumTokens).contains(maximumContext!),
              maximumMemoryBytes == nil || maximumMemoryBytes! > 0 else {
            throw AFMError.invalidRequest("Invalid Splash context, memory limit, or timeout")
        }
    }
}

public struct AFMSplashProviderFactory: AFMProviderFactory {
    public static let providerID: AFMProviderID = "splash"
    public init() {}
    public var descriptor: AFMProviderDescriptor {
        .init(id: Self.providerID, displayName: "Splash native", privacyBoundary: .device,
              configurationKeys: ["modelPath", "runtimePath", "maxContext", "maxMemoryBytes"])
    }
    public func modelDescriptors() async throws -> [AFMModelDescriptor] { [] }
    public func makeModel(id: AFMModelID, configuration: AFMProviderConfiguration) throws -> AnyAFMModel {
        guard case .string(let path) = configuration.values["modelPath"] else {
            throw AFMError.invalidRequest("Splash requires modelPath pointing to an existing Splash package (target/, draft/, tokenizer/)")
        }
        let runtime: AFMSplashRuntime
        if case .string(let path) = configuration.values["runtimePath"] { runtime = .init(root: URL(fileURLWithPath: path)) }
        else { runtime = try .bundled() }
        var config = AFMSplashConfiguration(runtime: runtime, modelDirectory: URL(fileURLWithPath: path))
        if case .integer(let value) = configuration.values["maxContext"] { config.maximumContext = value }
        if case .integer(let value) = configuration.values["maxMemoryBytes"] {
            guard value > 0 else { throw AFMError.invalidRequest("maxMemoryBytes must be positive") }
            config.maximumMemoryBytes = UInt64(value)
        }
        try config.validate()
        return AnyAFMModel(AFMSplashModel(id: id, configuration: config))
    }
}

public final class AFMSplashModel: AFMModel, AFMTextTokenizing, @unchecked Sendable {
    public let descriptor: AFMModelDescriptor
    private let configuration: AFMSplashConfiguration
    private let worker: SplashWorker
    private let loadTokenizer: @Sendable (URL) async throws -> any SplashTokenizing
    private let validateRuntime: @Sendable () throws -> Void

    public convenience init(id: AFMModelID, configuration: AFMSplashConfiguration) {
        self.init(id: id, configuration: configuration, loadTokenizer: { try await SplashTokenizer.load($0) }, validateRuntime: {
            try AFMSplashRuntime.checkPlatform()
            try configuration.runtime.validate()
        })
    }

    // Injection is internal so CPU tests can use an inert protocol peer, never a model.
    init(id: AFMModelID, configuration: AFMSplashConfiguration,
         loadTokenizer: @escaping @Sendable (URL) async throws -> any SplashTokenizing,
         validateRuntime: @escaping @Sendable () throws -> Void) {
        self.configuration = configuration
        self.loadTokenizer = loadTokenizer
        self.validateRuntime = validateRuntime
        worker = SplashWorker(configuration: configuration)
        descriptor = .init(providerID: AFMSplashProviderFactory.providerID, modelID: id,
                           displayName: id.rawValue, capabilities: [.text, .streaming, .speculativeDecoding],
                           contextWindow: configuration.maximumContext, privacyBoundary: .device, requiresNetwork: false)
    }

    public func availability() async -> AFMModelAvailability {
        do { try configuration.validate(); try validateRuntime(); return .available }
        catch { return .unavailable(reason: error.localizedDescription) }
    }

    public func load(progress: (@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor {
        if try await worker.perform({ _ in self.worker.isLoaded }) {
            progress?(1)
            return descriptor
        }
        try configuration.validate()
        try validateRuntime()
        let tokenizer = try await loadTokenizer(configuration.modelDirectory)
        try await worker.perform { cancellation in try self.worker.load(tokenizer, cancellation: cancellation) }
        progress?(1)
        return descriptor
    }

    public func tokenize(text: String) async throws -> [Int] {
        let tokenizer = try await loadTokenizer(configuration.modelDirectory)
        return tokenizer.encode(text)
    }

    public func respond(to request: AFMRequest) async throws -> AFMModelResponse {
        var response = AFMModelResponse()
        for try await event in streamResponse(to: request) {
            switch event {
            case .responseText(let action, let text, _):
                if action == .replace { response.text = text } else { response.text += text }
            case .usage(let usage): response.usage = usage
            case .completed(let reason): response.finishReason = reason
            default: break
            }
        }
        try Task.checkCancellation()
        return response
    }

    public func streamResponse(to request: AFMRequest) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try Self.validate(request)
                    _ = try await self.load(progress: nil)
                    try await self.worker.perform { cancellation in
                        try self.worker.generate(request, cancellation: cancellation) { continuation.yield($0) }
                    }
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public func unload() async { await worker.unload() }

    static func validate(_ request: AFMRequest) throws {
        let o = request.options
        guard request.tools.isEmpty, o.responseConstraint == nil, o.logprobs != true,
              o.topLogprobs == nil, o.minP == nil || o.minP == 0,
              o.repetitionPenalty == nil || o.repetitionPenalty == 1,
              o.presencePenalty == nil || o.presencePenalty == 0,
              !o.ignoreEndOfSequence, o.reasoningEnabled != true,
              o.stopSequences.isEmpty else {
            throw AFMError.unsupportedCapability("Splash native adapter currently supports text chat, streaming, temperature, top-p, top-k, seed, and max tokens; use afm splash for tools, reasoning, constraints, custom stops, or multimodal requests")
        }
        guard !request.messages.isEmpty, request.messages.allSatisfy({ message in
            message.role != .tool && message.toolCalls.isEmpty && message.toolCallID == nil
                && message.content.allSatisfy { if case .text = $0 { return true }; return false }
        }) else { throw AFMError.unsupportedCapability("Splash native adapter requires text-only messages without tool calls") }
    }
}

private final class SplashWorker: @unchecked Sendable {
    let configuration: AFMSplashConfiguration
    private let queue = DispatchQueue(label: "afm.splash.native")
    private let lock = NSLock()
    private var activeCancellation: SplashCancellation?
    // Everything below is accessed exclusively on queue.
    private var transport: SplashTransport?
    private var tokenizer: (any SplashTokenizing)?
    private var context = 0
    private var nextID: UInt64 = 0
    var isLoaded: Bool { transport != nil }

    init(configuration: AFMSplashConfiguration) { self.configuration = configuration }

    func perform<T: Sendable>(_ body: @escaping @Sendable (SplashCancellation) throws -> T) async throws -> T {
        let cancellation = SplashCancellation()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                queue.async {
                    self.lock.withLock { self.activeCancellation = cancellation }
                    defer { self.lock.withLock { self.activeCancellation = nil } }
                    do { try cancellation.check(); continuation.resume(returning: try body(cancellation)) }
                    catch { continuation.resume(throwing: error) }
                }
            }
        } onCancel: { cancellation.cancel() }
    }

    func unload() async {
        lock.withLock { activeCancellation?.cancel() }
        await withCheckedContinuation { continuation in
            queue.async { self.transport?.close(); self.transport = nil; self.tokenizer = nil; continuation.resume() }
        }
    }

    func load(_ tokenizer: any SplashTokenizing, cancellation: SplashCancellation) throws {
        if transport != nil { return }
        let process = try SplashTransport(executable: configuration.runtime.nativeExecutable, arguments: [
            "serve-native", configuration.modelDirectory.appendingPathComponent("target").path,
            configuration.modelDirectory.appendingPathComponent("draft").path,
            configuration.maximumContext.map(String.init) ?? "auto",
            configuration.maximumMemoryBytes.map(String.init) ?? "auto"])
        do {
            let frame = try process.frame(deadline: Date().addingTimeInterval(configuration.startupTimeout), cancellation: cancellation)
            if frame.type == 0x105 { throw try SplashWire.remoteError(frame.payload) }
            guard frame.type == 0x100 else { throw SplashWire.invalid("Expected Ready") }
            var r = SplashWire.Reader(data: frame.payload)
            let instance = try r.integer(UInt64.self)
            let capacity = try r.integer(UInt32.self)
            context = Int(try r.integer(UInt32.self))
            _ = try r.integer(UInt64.self)
            try r.end()
            guard instance > 0, capacity > 0, context > 0 else { throw SplashWire.invalid("Invalid Ready limits") }
            self.tokenizer = tokenizer
            transport = process
        } catch { process.close(); throw error }
    }

    func generate(_ request: AFMRequest, cancellation: SplashCancellation,
                  emit: @Sendable (AFMGenerationEvent) -> Void) throws {
        guard let transport, let tokenizer else { throw AFMError.unavailable("Splash is not loaded") }
        let prompt = try tokenizer.prompt(request)
        let outputLimit = request.options.maximumResponseTokens ?? SplashWire.defaultOutputTokens
        guard prompt.count < context, outputLimit > 0, outputLimit <= context - prompt.count else {
            throw AFMError.invalidRequest("Splash prompt plus max tokens exceeds context window \(context)")
        }
        nextID += 1
        let id = nextID
        let deadline = Date().addingTimeInterval(configuration.requestTimeout)
        let data = try SplashWire.request(id: id, tokens: prompt, options: request.options, timeout: configuration.requestTimeout)
        var tokens: [Int] = []
        var emitted = ""
        var cached = 0
        do {
            try transport.write(data, deadline: deadline, cancellation: cancellation)
            while true {
                let frame = try transport.frame(deadline: deadline, cancellation: cancellation)
                if frame.type == 0x105 { throw try SplashWire.remoteError(frame.payload) }
                var r = SplashWire.Reader(data: frame.payload)
                guard try r.integer(UInt64.self) == id else { throw SplashWire.invalid("Mismatched request ID") }
                switch frame.type {
                case 0x101:
                    let disposition = try r.integer(UInt8.self)
                    _ = try r.integer(UInt32.self)
                    cached = Int(try r.integer(UInt32.self))
                    _ = try r.integer(UInt32.self)
                    try r.end()
                    guard disposition <= 1, cached <= prompt.count else { throw SplashWire.invalid("Invalid Start") }
                case 0x102:
                    let offset = Int(try r.integer(UInt32.self))
                    let count = Int(try r.integer(UInt32.self))
                    guard offset == tokens.count, count > 0, count <= SplashWire.maximumBatch,
                          count <= outputLimit - tokens.count else { throw SplashWire.invalid("Invalid token batch or sequence offset") }
                    for _ in 0..<count { tokens.append(Int(try r.integer(UInt32.self))) }
                    try r.end()
                    let text = tokenizer.decode(tokens)
                    // Byte-level BPE can temporarily end in a replacement character.
                    if !text.hasSuffix("\u{FFFD}") {
                        guard text.hasPrefix(emitted) else { throw SplashWire.invalid("Tokenizer rewrote emitted text") }
                        let delta = String(text.dropFirst(emitted.count))
                        if !delta.isEmpty { emit(.responseText(action: .append, text: delta, tokenCount: count)) }
                        emitted = text
                    }
                case 0x104:
                    let reason = try r.integer(UInt8.self)
                    let input = Int(try r.integer(UInt32.self))
                    let output = Int(try r.integer(UInt32.self))
                    _ = try r.integer(UInt64.self); _ = try r.integer(UInt64.self); _ = try r.integer(UInt64.self)
                    try r.end()
                    guard reason <= 2, input == prompt.count, output == tokens.count else { throw SplashWire.invalid("Invalid Done accounting") }
                    let text = tokenizer.decode(tokens)
                    guard text.hasPrefix(emitted) else { throw SplashWire.invalid("Tokenizer rewrote final text") }
                    if text != emitted { emit(.responseText(action: .append, text: String(text.dropFirst(emitted.count)), tokenCount: 0)) }
                    emit(.usage(.init(inputTokens: input, cachedInputTokens: cached, outputTokens: output)))
                    emit(.completed(reason == 0 ? .stop : reason == 1 ? .length : .cancelled))
                    return
                case 0x106: throw AFMError.generationFailed("Splash native capacity exhausted; retry after current work completes")
                default: throw SplashWire.invalid("Unexpected frame type \(frame.type)")
                }
            }
        } catch {
            if error is CancellationError {
                try? transport.write(SplashWire.cancel(id: id), deadline: Date().addingTimeInterval(1), cancellation: SplashCancellation())
            }
            // Discard a cancelled/failed stream; never mix late tokens into the next request.
            transport.close(); self.transport = nil
            throw error
        }
    }
}
