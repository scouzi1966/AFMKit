import Foundation
import CryptoKit
import AFMKitCore

public struct AFMSplashRelease: Decodable, Sendable {
    public let version: String
    public let revision: String
    public let protocolVersion: Int
    public let url: String
    public let sha256: String
    public let directory: String

    public static func pinned() throws -> Self {
        guard let url = Bundle.module.url(forResource: "splash-release", withExtension: "json") else {
            throw AFMError.unavailable("AFMKitSplash release metadata is missing")
        }
        return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }
}

public struct AFMSplashRuntime: Sendable {
    public let root: URL
    public var nativeExecutable: URL { root.appendingPathComponent("engine/splash") }
    public var pythonExecutable: URL { root.appendingPathComponent("python/bin/python3") }
    public var launcher: URL { root.appendingPathComponent("install/launcher.py") }

    public init(root: URL) { self.root = root }

    public static func bundled(beside executable: String = Bundle.main.executablePath ?? CommandLine.arguments[0]) throws -> Self {
        let directory = URL(fileURLWithPath: executable).resolvingSymlinksInPath().deletingLastPathComponent()
        let candidates = [directory.appendingPathComponent("splash-runtime"),
                          directory.appendingPathComponent("../libexec/afm/splash-runtime").standardizedFileURL,
                          directory.appendingPathComponent("../libexec/splash-runtime").standardizedFileURL]
        guard let root = candidates.first(where: { FileManager.default.fileExists(atPath: $0.appendingPathComponent("release.json").path) }) else {
            throw AFMError.unavailable("Bundled Splash is missing. Rebuild with make build or reinstall AFM with its splash-runtime directory.")
        }
        return Self(root: root)
    }

    public static func checkPlatform() throws {
        #if arch(arm64) && os(macOS)
        guard #available(macOS 26.4, *) else {
            throw AFMError.unavailable("Splash requires macOS 26.4 or newer on Apple Silicon (M3 or newer).")
        }
        #else
        throw AFMError.unavailable("Splash requires macOS 26.4 or newer on Apple Silicon (M3 or newer).")
        #endif
    }

    public func validate() throws {
        let pin = try AFMSplashRelease.pinned()
        struct Identity: Decodable {
            let version: String
            let binary_sha256: String
            let metallib_sha256: String
        }
        let identity = try JSONDecoder().decode(Identity.self, from: Data(contentsOf: root.appendingPathComponent("release.json")))
        let installedPin = try JSONDecoder().decode(AFMSplashRelease.self, from: Data(contentsOf: root.appendingPathComponent("afm-release-pin.json")))
        guard identity.version == pin.version, installedPin.sha256 == pin.sha256,
              installedPin.revision == pin.revision else {
            throw AFMError.unavailable("Splash release mismatch; this AFM build requires Splash \(pin.version) (\(pin.revision)). Reinstall the bundled runtime.")
        }
        for (file, hash) in [(nativeExecutable, identity.binary_sha256), (root.appendingPathComponent("engine/splash.metallib"), identity.metallib_sha256)] {
            let digest = SHA256.hash(data: try Data(contentsOf: file, options: .mappedIfSafe)).map { String(format: "%02x", $0) }.joined()
            guard digest == hash else { throw AFMError.unavailable("Splash runtime checksum mismatch: \(file.lastPathComponent)") }
        }
    }
}
