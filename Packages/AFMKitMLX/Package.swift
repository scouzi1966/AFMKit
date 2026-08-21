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

let package = Package(
    name: "AFMKitMLX",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(
            name: "AFMKitMLX",
            targets: ["AFMKitMLX"]
        ),
        .library(
            name: "AFMKitFoundationModelsMLX",
            targets: ["AFMKitFoundationModelsMLX"]
        )
    ],
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
    targets: [
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
            name: "AFMKitMLX",
            dependencies: [
                .product(name: "AFMKitCore", package: "AFMKit"),
                .product(name: "AFMOpenAICompat", package: "AFMKit"),
                "AFMXGrammar",
                .product(name: "MLX", package: mlxSwiftPackageIdentity),
                .product(name: "MLXLLM", package: mlxSwiftLMPackageIdentity),
                .product(name: "MLXVLM", package: mlxSwiftLMPackageIdentity),
                .product(name: "MLXLMCommon", package: mlxSwiftLMPackageIdentity),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface")
            ],
            resources: [
                .copy("Resources/default.metallib")
            ],
            linkerSettings: [
                .linkedFramework("Security"),
                .linkedFramework("IOKit"),
                .linkedLibrary("IOReport"),
                .linkedLibrary("sqlite3")
            ]
        ),
        .target(
            name: "AFMKitFoundationModelsMLX",
            dependencies: [
                .product(name: "AFMKitCore", package: "AFMKit"),
                "AFMKitMLX"
            ]
        ),
        .testTarget(
            name: "AFMKitMLXTests",
            dependencies: [
                "AFMKitMLX",
                .product(name: "AFMKitCore", package: "AFMKit"),
                .product(name: "AFMOpenAICompat", package: "AFMKit"),
                .product(name: "MLXLMCommon", package: mlxSwiftLMPackageIdentity)
            ]
        ),
        .testTarget(
            name: "AFMKitFoundationModelsMLXTests",
            dependencies: [
                .product(name: "AFMKitCore", package: "AFMKit"),
                "AFMKitFoundationModelsMLX"
            ]
        )
    ],
    cxxLanguageStandard: .gnucxx17
)
