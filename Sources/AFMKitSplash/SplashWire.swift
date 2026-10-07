import Foundation
import AFMKitCore

// Splash 1.0 runtime/engine/Protocol.hpp, little-endian protocol v5.
// No native headers, MLX, Metal, Python, or HTTP dependencies.
enum SplashWire {
    static let version: UInt16 = 5
    static let headerBytes = 24
    static let maximumPayload = 256 * 1024 * 1024
    static let maximumTokens = 1 << 20
    static let maximumBatch = 4096
    static let defaultOutputTokens = 1024

    struct Frame {
        let type: UInt16
        let payload: Data
    }

    struct Reader {
        let data: Data
        var offset = 0
        mutating func integer<T: FixedWidthInteger>(_ type: T.Type = T.self) throws -> T {
            let size = MemoryLayout<T>.size
            guard offset + size <= data.count else { throw invalid("Truncated payload") }
            defer { offset += size }
            return data.withUnsafeBytes { T(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: T.self)) }
        }
        mutating func bytes(_ count: Int) throws -> Data {
            guard count >= 0, count <= data.count - offset else { throw invalid("Invalid payload length") }
            defer { offset += count }
            return data.subdata(in: offset..<offset + count)
        }
        func end() throws {
            guard offset == data.count else { throw invalid("Unexpected trailing payload") }
        }
    }

    static func invalid(_ message: String) -> AFMError { .generationFailed("Splash protocol: \(message)") }

    static func header(_ data: Data) throws -> (UInt16, Int) {
        guard data.count == headerBytes else { throw invalid("Truncated header") }
        var r = Reader(data: data)
        guard try r.bytes(4) == Data("SPLH".utf8),
              try r.integer(UInt16.self) == version,
              try r.integer(UInt16.self) == UInt16(headerBytes) else {
            throw invalid("Expected Splash 1.0 protocol v5")
        }
        let type = try r.integer(UInt16.self)
        guard try r.integer(UInt16.self) == 0 else { throw invalid("Nonzero flags") }
        let length = try r.integer(UInt64.self)
        guard length <= maximumPayload, try r.integer(UInt32.self) == 0 else {
            throw invalid("Invalid frame size or reserved field")
        }
        // This text-only client never accepts image/mask/request payloads from
        // the child. Reject impossible sizes before allocating the payload.
        let bounds: ClosedRange<UInt64>
        switch type {
        case 0x100: bounds = 24...24
        case 0x101: bounds = 21...21
        case 0x102: bounds = 16...UInt64(16 + maximumBatch * 4)
        case 0x104: bounds = 41...41
        case 0x105: bounds = 18...UInt64(18 + 2 * 1024 * 1024)
        case 0x106: bounds = 24...24
        default: throw invalid("Unexpected inbound frame type \(type)")
        }
        guard bounds.contains(length) else { throw invalid("Invalid payload size for frame \(type)") }
        return (type, Int(length))
    }

    static func frame(type: UInt16, payload: Data) -> Data {
        var data = Data("SPLH".utf8)
        data.word(version); data.word(UInt16(headerBytes)); data.word(type)
        data.word(UInt16(0)); data.word(UInt64(payload.count)); data.word(UInt32(0))
        data.append(payload)
        return data
    }

    static func request(id: UInt64, tokens: [Int], options: AFMGenerationOptions, timeout: TimeInterval) throws -> Data {
        let maxTokens = options.maximumResponseTokens ?? defaultOutputTokens
        let temperature = options.temperature ?? 0
        let topP = options.topP ?? 1
        let topK = options.topK ?? (temperature > 0 ? 32 : 0)
        guard !tokens.isEmpty, tokens.count <= maximumTokens,
              tokens.allSatisfy({ $0 >= 0 && UInt64($0) <= UInt32.max }),
              (1...maximumTokens).contains(maxTokens),
              temperature.isFinite, temperature >= 0, temperature <= Double(Float.greatestFiniteMagnitude),
              topP.isFinite, topP > 0, topP <= 1, (0...32).contains(topK),
              temperature == 0 || topK > 0,
              (options.seed ?? 0) >= 0 else { throw AFMError.invalidRequest("Invalid Splash tokens or sampling parameters (top-k must be 0...32)") }
        var payload = Data()
        payload.word(id); payload.word(UInt8(1)); payload.word(UInt8(temperature == 0 ? 0 : 1)); payload.word(UInt8(0))
        let micros = UInt64(timeout * 1_000_000)
        payload.word(UInt64(Date().timeIntervalSince1970 * 1_000_000) + micros)
        payload.word(micros); payload.word(UInt32(maxTokens)); payload.word(UInt32(tokens.count))
        payload.word(UInt32(0)) // text-only: no image spans
        payload.word(Float(temperature).bitPattern); payload.word(Float(topP).bitPattern)
        payload.word(UInt32(topK)); payload.word(UInt64(options.seed ?? 0)); payload.word(UInt8(0))
        tokens.forEach { payload.word(UInt32($0)) }
        return frame(type: 1, payload: payload)
    }

    static func cancel(id: UInt64) -> Data {
        var payload = Data(); payload.word(id)
        return frame(type: 2, payload: payload)
    }

    static func remoteError(_ payload: Data) throws -> AFMError {
        var r = Reader(data: payload)
        let classification = try r.integer(UInt8.self)
        let retryable = try r.integer(UInt8.self)
        _ = try r.integer(UInt64.self)
        let codeCount = Int(try r.integer(UInt32.self))
        let messageCount = Int(try r.integer(UInt32.self))
        guard (1...3).contains(classification), retryable <= 1 else { throw invalid("Invalid error classification") }
        let code = String(decoding: try r.bytes(codeCount), as: UTF8.self)
        let message = String(decoding: try r.bytes(messageCount), as: UTF8.self)
        try r.end()
        return .generationFailed("Splash \(code): \(message)")
    }
}

extension Data {
    mutating func word<T: FixedWidthInteger>(_ value: T) {
        var value = value.littleEndian
        Swift.withUnsafeBytes(of: &value) { append(contentsOf: $0) }
    }
}
