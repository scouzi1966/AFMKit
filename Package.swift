// swift-tools-version: 6.1
import PackageDescription
import Foundation

let packageDirectory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let mlxSwiftDependency: Package.Dependency
let mlxSwiftLMDependency: Package.Dependency
let mlxSwiftPackageIdentity = "mlx-swift-afm"
let mlxSwiftLMPackageIdentity: String

if let localMLXSwiftPath = ProcessInfo.processInfo.environment["AFMKIT_MLX_SWIFT_PATH"],
   !localMLXSwiftPath.isEmpty {
    mlxSwiftDependency = .package(path: localMLXSwiftPath)
} else {
    mlxSwiftDependency = .package(
        url: "https://github.com/scouzi1966/mlx-swift-afm",
        exact: "0.31.6-afm.1"
    )
}

if let localMLXSwiftLMPath = ProcessInfo.processInfo.environment["AFMKIT_MLX_SWIFT_LM_PATH"],
   !localMLXSwiftLMPath.isEmpty {
    mlxSwiftLMDependency = .package(path: localMLXSwiftLMPath)
    mlxSwiftLMPackageIdentity = "mlx-swift-lm-afm"
} else {
    mlxSwiftLMDependency = .package(
        url: "https://github.com/scouzi1966/mlx-swift-lm.git",
        exact: "0.31.6-afm.2"
    )
    mlxSwiftLMPackageIdentity = "mlx-swift-lm"
}

let package = Package(
    name: "AFMKit",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(
            name: "AFMKitCore",
            targets: ["AFMKitCore"]
        ),
        .library(
            name: "AFMOpenAICompat",
            targets: ["AFMOpenAICompat"]
        ),
        .library(
            name: "AFMKitApple",
            targets: ["AFMKitApple"]
        ),
        .library(
            name: "AFMKitMLX",
            targets: ["AFMKitMLX"]
        )
    ],
    dependencies: [
        mlxSwiftLMDependency,
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            from: "1.3.0"
        ),
        .package(
            url: "https://github.com/huggingface/swift-huggingface.git",
            from: "0.8.1",
            traits: ["Xet"]
        ),
        .package(
            url: "https://github.com/mlc-ai/xgrammar",
            revision: "c1570cdb4f8c867a4dbd07b7ff90581f4a2a432b"
        ),
        mlxSwiftDependency
    ],
    targets: [
        .target(
            name: "AFMKitCore",
            dependencies: []
        ),
        .target(
            name: "AFMOpenAICompat",
            dependencies: []
        ),
        .target(
            name: "AFMKitApple",
            dependencies: ["AFMKitCore", "AFMOpenAICompat"]
        ),
        .target(
            name: "AFMXGrammar",
            dependencies: [
                .product(name: "XGrammar", package: "xgrammar")
            ],
            path: "Sources/CXGrammar",
            exclude: ["xgrammar"],
            cxxSettings: [
                .headerSearchPath("xgrammar/3rdparty/dlpack/include")
            ]
        ),
        .target(
            name: "AFMKitMLX",
            dependencies: [
                "AFMKitCore",
                "AFMOpenAICompat",
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
            swiftSettings: [
                .unsafeFlags(["-cross-module-optimization"], .when(configuration: .release)),
                .unsafeFlags(["-O"], .when(configuration: .release)),
                .unsafeFlags(
                    ["-file-prefix-map", "\(packageDirectory)/="],
                    .when(configuration: .release)
                )
            ],
            linkerSettings: [
                .linkedFramework("Security"),
                .linkedFramework("IOKit"),
                .linkedLibrary("IOReport"),
                .linkedLibrary("sqlite3")
            ]
        ),
        .testTarget(
            name: "AFMKitCoreTests",
            dependencies: ["AFMKitCore"]
        ),
        .testTarget(
            name: "AFMOpenAICompatTests",
            dependencies: ["AFMOpenAICompat"]
        ),
        .testTarget(
            name: "AFMKitAppleTests",
            dependencies: ["AFMKitApple", "AFMKitCore"]
        ),
        .testTarget(
            name: "AFMKitMLXTests",
            dependencies: [
                "AFMKitMLX",
                "AFMKitCore",
                "AFMOpenAICompat",
                .product(name: "MLXLMCommon", package: mlxSwiftLMPackageIdentity)
            ]
        )
    ],
    cxxLanguageStandard: .gnucxx17
)
