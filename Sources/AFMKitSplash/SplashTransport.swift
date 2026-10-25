import Foundation
import Darwin
import AFMKitCore

final class SplashCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    func cancel() { lock.withLock { cancelled = true } }
    func check() throws { if lock.withLock({ cancelled }) { throw CancellationError() } }
}

// Confined to the provider's serial worker queue. Polling keeps blocking pipe I/O
// off Swift's cooperative executor and bounds cancellation/timeout latency.
final class SplashTransport {
    static let pollMilliseconds: Int32 = 100
    static let terminationGrace: TimeInterval = 2
    static let terminationPoll: TimeInterval = 0.01
    let process: Process
    private let input = Pipe()
    private let output = Pipe()
    private var closed = false
    private var started = false

    init(executable: URL, arguments: [String]) throws {
        process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.standardInput = input
        process.standardOutput = output
        process.standardError = FileHandle.standardError
        // A crashed engine must return EPIPE, never kill the AFM host.
        _ = fcntl(input.fileHandleForWriting.fileDescriptor, F_SETNOSIGPIPE, 1)
        try process.run()
        started = true
        try? input.fileHandleForReading.close()
        try? output.fileHandleForWriting.close()
        _ = fcntl(input.fileHandleForWriting.fileDescriptor, F_SETFL, O_NONBLOCK)
        _ = fcntl(output.fileHandleForReading.fileDescriptor, F_SETFL, O_NONBLOCK)
    }

    deinit { close() }

    func close() {
        guard !closed else { return }
        closed = true
        try? input.fileHandleForWriting.close()
        try? output.fileHandleForReading.close()
        if started && process.isRunning {
            process.terminate()
            let deadline = Date().addingTimeInterval(Self.terminationGrace)
            while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: Self.terminationPoll) }
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
        }
        // Foundation reaps its child asynchronously. waitUntilExit can wait for
        // run-loop delivery indefinitely from this worker queue on macOS 27.
    }

    private func wait(_ descriptor: Int32, events: Int16, deadline: Date, cancellation: SplashCancellation) throws {
        try cancellation.check()
        guard Date() < deadline else { throw AFMError.generationFailed("Splash runtime timed out") }
        var descriptor = pollfd(fd: descriptor, events: events, revents: 0)
        let result = poll(&descriptor, 1, Self.pollMilliseconds)
        guard result >= 0 || errno == EINTR else { throw AFMError.generationFailed("Splash pipe poll failed") }
    }

    func write(_ data: Data, deadline: Date, cancellation: SplashCancellation) throws {
        var offset = 0
        while offset < data.count {
            try wait(input.fileHandleForWriting.fileDescriptor, events: Int16(POLLOUT), deadline: deadline, cancellation: cancellation)
            let count = data.withUnsafeBytes {
                Darwin.write(input.fileHandleForWriting.fileDescriptor, $0.baseAddress!.advanced(by: offset), data.count - offset)
            }
            if count < 0 && (errno == EINTR || errno == EAGAIN) { continue }
            guard count > 0 else { throw AFMError.generationFailed("Splash engine closed its input pipe") }
            offset += count
        }
    }

    private func read(_ count: Int, deadline: Date, cancellation: SplashCancellation) throws -> Data {
        var data = Data(count: count)
        var offset = 0
        while offset < count {
            try wait(output.fileHandleForReading.fileDescriptor, events: Int16(POLLIN), deadline: deadline, cancellation: cancellation)
            let received = data.withUnsafeMutableBytes {
                Darwin.read(output.fileHandleForReading.fileDescriptor, $0.baseAddress!.advanced(by: offset), count - offset)
            }
            if received < 0 && (errno == EINTR || errno == EAGAIN) { continue }
            guard received > 0 else { throw SplashWire.invalid("Engine exited or truncated a frame") }
            offset += received
        }
        return data
    }

    func frame(deadline: Date, cancellation: SplashCancellation) throws -> SplashWire.Frame {
        let (type, count) = try SplashWire.header(read(SplashWire.headerBytes, deadline: deadline, cancellation: cancellation))
        return SplashWire.Frame(type: type, payload: try read(count, deadline: deadline, cancellation: cancellation))
    }
}
