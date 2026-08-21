// swift-tools-version: 6.1
import Foundation
import PackageDescription

let publicPackageDependency: Package.Dependency
let mlxSwiftDependency: Package.Dependency
let mlxSwiftLMDependency: Package.Dependency
let mlxSwiftPackageIdentity: String
let mlxSwiftLMPackageIdentity: String
let releaseDependencyPins: [Package.Dependency] = [
    .package(url: "https://github.com/swift-server/async-http-client", exact: "1.36.0"),
    .package(url: "https://github.com/mattt/EventSource.git", exact: "1.5.1"),
    .package(url: "https://github.com/apple/swift-algorithms.git", exact: "1.2.1"),
    .package(url: "https://github.com/apple/swift-argument-parser", exact: "1.8.2"),
    .package(url: "https://github.com/apple/swift-asn1.git", exact: "1.7.1"),
    .package(url: "https://github.com/apple/swift-async-algorithms.git", exact: "1.1.5"),
    .package(url: "https://github.com/apple/swift-atomics.git", exact: "1.3.1"),
    .package(url: "https://github.com/apple/swift-certificates.git", exact: "1.19.4"),
    .package(url: "https://github.com/apple/swift-collections.git", exact: "1.6.0"),
    .package(url: "https://github.com/apple/swift-configuration.git", exact: "1.2.0"),
    .package(url: "https://github.com/apple/swift-crypto.git", exact: "4.5.1"),
    .package(url: "https://github.com/apple/swift-distributed-tracing.git", exact: "1.4.1"),
    .package(url: "https://github.com/apple/swift-http-structured-headers.git", exact: "1.7.0"),
    .package(url: "https://github.com/apple/swift-http-types.git", exact: "1.6.0"),
    .package(url: "https://github.com/huggingface/swift-jinja.git", exact: "2.4.2"),
    .package(url: "https://github.com/apple/swift-log.git", exact: "1.15.0"),
    .package(url: "https://github.com/apple/swift-nio.git", exact: "2.101.3"),
    .package(url: "https://github.com/apple/swift-nio-extras.git", exact: "1.34.3"),
    .package(url: "https://github.com/apple/swift-nio-http2.git", exact: "1.45.0"),
    .package(url: "https://github.com/apple/swift-nio-ssl.git", exact: "2.37.2"),
    .package(url: "https://github.com/apple/swift-nio-transport-services.git", exact: "1.28.0"),
    .package(url: "https://github.com/apple/swift-numerics", exact: "1.1.1"),
    .package(url: "https://github.com/apple/swift-service-context.git", exact: "1.3.0"),
    .package(url: "https://github.com/swift-server/swift-service-lifecycle", exact: "2.11.0"),
    .package(url: "https://github.com/apple/swift-system.git", exact: "1.8.1"),
    .package(url: "https://github.com/ibireme/yyjson.git", exact: "0.12.0")
]
let releaseGraphProductPins: [Target.Dependency] = [
    .product(name: "AsyncHTTPClient", package: "async-http-client"),
    .product(name: "EventSource", package: "eventsource"),
    .product(name: "Algorithms", package: "swift-algorithms"),
    .product(name: "ArgumentParser", package: "swift-argument-parser"),
    .product(name: "SwiftASN1", package: "swift-asn1"),
    .product(name: "AsyncAlgorithms", package: "swift-async-algorithms"),
    .product(name: "Atomics", package: "swift-atomics"),
    .product(name: "X509", package: "swift-certificates"),
    .product(name: "Collections", package: "swift-collections"),
    .product(name: "Configuration", package: "swift-configuration"),
    .product(name: "Crypto", package: "swift-crypto"),
    .product(name: "Instrumentation", package: "swift-distributed-tracing"),
    .product(name: "StructuredFieldValues", package: "swift-http-structured-headers"),
    .product(name: "HTTPTypes", package: "swift-http-types"),
    .product(name: "Jinja", package: "swift-jinja"),
    .product(name: "Logging", package: "swift-log"),
    .product(name: "NIOCore", package: "swift-nio"),
    .product(name: "NIOExtras", package: "swift-nio-extras"),
    .product(name: "NIOHTTP2", package: "swift-nio-http2"),
    .product(name: "NIOSSL", package: "swift-nio-ssl"),
    .product(name: "NIOTransportServices", package: "swift-nio-transport-services"),
    .product(name: "Numerics", package: "swift-numerics"),
    .product(name: "ServiceContextModule", package: "swift-service-context"),
    .product(name: "ServiceLifecycle", package: "swift-service-lifecycle"),
    .product(name: "SystemPackage", package: "swift-system"),
    .product(name: "yyjson", package: "yyjson")
]

if let publicPackagePath = ProcessInfo.processInfo.environment["AFMKIT_PUBLIC_PATH"],
   !publicPackagePath.isEmpty {
    publicPackageDependency = .package(name: "AFMKit", path: publicPackagePath)
} else {
    publicPackageDependency = .package(name: "AFMKit", path: "../..")
}

if let localMLXSwiftPath = ProcessInfo.processInfo.environment["AFMKIT_MLX_SWIFT_PATH"],
   !localMLXSwiftPath.isEmpty {
    mlxSwiftDependency = .package(path: localMLXSwiftPath)
    mlxSwiftPackageIdentity = URL(fileURLWithPath: localMLXSwiftPath).lastPathComponent.lowercased()
} else {
    mlxSwiftDependency = .package(
        url: "https://github.com/scouzi1966/mlx-swift-afm",
        exact: "0.31.6-afm.1"
    )
    mlxSwiftPackageIdentity = "mlx-swift-afm"
}

if let localMLXSwiftLMPath = ProcessInfo.processInfo.environment["AFMKIT_MLX_SWIFT_LM_PATH"],
   !localMLXSwiftLMPath.isEmpty {
    mlxSwiftLMDependency = .package(path: localMLXSwiftLMPath)
    mlxSwiftLMPackageIdentity = URL(fileURLWithPath: localMLXSwiftLMPath).lastPathComponent.lowercased()
} else {
    mlxSwiftLMDependency = .package(
        url: "https://github.com/scouzi1966/mlx-swift-lm.git",
        exact: "0.31.6-afm.3"
    )
    mlxSwiftLMPackageIdentity = "mlx-swift-lm"
}

var products: [Product] = [
    .library(name: "AFMKitMLX", targets: ["AFMKitMLX"])
]
var targets: [Target] = [
    .target(
        name: "AFMXGrammar",
        dependencies: [],
        path: "Sources/CXGrammar",
        sources: [
            "error_handler.cpp",
            "grammar_compiler.cpp",
            "grammar_matcher.cpp",
            "tokenizer_info.cpp",
            "xgrammar/cpp"
        ],
        cxxSettings: [
            .headerSearchPath("xgrammar/include"),
            .headerSearchPath("xgrammar/cpp"),
            .headerSearchPath("xgrammar/3rdparty/dlpack/include"),
            .headerSearchPath("xgrammar/3rdparty/picojson"),
            .define("XGRAMMAR_ENABLE_LOG_DEBUG", to: "0"),
            .define("XGRAMMAR_ENABLE_CPPTRACE", to: "0")
        ]
    ),
    .target(
        name: "AFMKitMLXReleaseGraph",
        dependencies: releaseGraphProductPins
    ),
    .target(
        name: "AFMKitMLX",
        dependencies: [
            .product(name: "AFMKitCore", package: "AFMKit"),
            .product(name: "AFMOpenAICompat", package: "AFMKit"),
            "AFMXGrammar",
            "AFMKitMLXReleaseGraph",
            .product(name: "MLX", package: mlxSwiftPackageIdentity),
            .product(name: "MLXLLM", package: mlxSwiftLMPackageIdentity),
            .product(name: "MLXVLM", package: mlxSwiftLMPackageIdentity),
            .product(name: "MLXLMCommon", package: mlxSwiftLMPackageIdentity),
            .product(name: "Tokenizers", package: "swift-transformers"),
            .product(name: "Hub", package: "swift-transformers"),
            .product(name: "HuggingFace", package: "swift-huggingface")
        ],
        resources: [.copy("Resources/default.metallib")],
        linkerSettings: [
            .linkedFramework("Security"),
            .linkedFramework("IOKit"),
            .linkedLibrary("IOReport"),
            .linkedLibrary("sqlite3")
        ]
    )
]

#if compiler(>=6.4)
products.append(
    .library(
        name: "AFMKitFoundationModelsMLX",
        targets: ["AFMKitFoundationModelsMLX"]
    )
)
targets.append(
    .target(
        name: "AFMKitFoundationModelsMLX",
        dependencies: [
            .product(name: "AFMKitCore", package: "AFMKit"),
            "AFMKitMLX"
        ]
    )
)
#endif

targets.append(
    .testTarget(
        name: "AFMKitMLXTests",
        dependencies: [
            "AFMKitMLX",
            .product(name: "AFMKitCore", package: "AFMKit"),
            .product(name: "AFMOpenAICompat", package: "AFMKit"),
            .product(name: "MLXLMCommon", package: mlxSwiftLMPackageIdentity)
        ]
    )
)

#if compiler(>=6.4)
targets.append(
    .testTarget(
        name: "AFMKitFoundationModelsMLXTests",
        dependencies: [
            .product(name: "AFMKitCore", package: "AFMKit"),
            "AFMKitFoundationModelsMLX"
        ]
    )
)
#endif

let package = Package(
    name: "AFMKitMLX",
    platforms: [
        .macOS("26.0")
    ],
    products: products,
    dependencies: [
        publicPackageDependency,
        mlxSwiftLMDependency,
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            exact: "1.3.3"
        ),
        .package(
            url: "https://github.com/huggingface/swift-huggingface.git",
            exact: "0.9.0",
            traits: ["Xet"]
        ),
        .package(
            url: "https://github.com/huggingface/swift-xet.git",
            exact: "0.2.3"
        ),
        mlxSwiftDependency
    ] + releaseDependencyPins,
    targets: targets,
    cxxLanguageStandard: .gnucxx17
)
