// swift-tools-version: 6.1
import PackageDescription

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

let package = Package(
    name: "AFMKitPrivateQualification",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(name: "AFMKitCore", targets: ["AFMKitCore"]),
        .library(name: "AFMOpenAICompat", targets: ["AFMOpenAICompat"]),
        .library(name: "AFMKitInference", targets: ["AFMKitInference"]),
        .library(name: "AFMKitApple", targets: ["AFMKitApple"]),
        .library(name: "AFMKitEmbeddings", targets: ["AFMKitEmbeddings"]),
        .library(name: "AFMKitSpeech", targets: ["AFMKitSpeech"]),
        .library(name: "AFMKitSpeechSynthesis", targets: ["AFMKitSpeechSynthesis"]),
        .library(name: "AFMKitVision", targets: ["AFMKitVision"]),
        .library(name: "AFMKitServices", targets: ["AFMKitServices"]),
        .library(name: "AFMEvalKit", targets: ["AFMEvalKit"]),
        .library(name: "AFMKitDwarfStar", targets: ["AFMKitDwarfStar"]),
        .library(
            name: "AFMKitFoundationModelsDwarfStar",
            targets: ["AFMKitFoundationModelsDwarfStar"]
        ),
        .library(name: "AFMKitMLX", targets: ["AFMKitMLX"]),
        .library(
            name: "AFMKitFoundationModelsMLX",
            targets: ["AFMKitFoundationModelsMLX"]
        )
    ],
    dependencies: [
        .package(
            url: "https://github.com/scouzi1966/mlx-swift-lm.git",
            exact: "0.31.6-afm.3"
        ),
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
        .package(
            url: "https://github.com/scouzi1966/mlx-swift-afm",
            exact: "0.31.6-afm.1"
        )
    ] + releaseDependencyPins,
    targets: [
        .executableTarget(
            name: "AFMKitPrivateDependencySeed",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift-afm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Xet", package: "swift-xet")
            ],
            path: "TrustedSeed"
        ),
        .target(
            name: "AFMKitCore",
            path: "Candidate/Sources/AFMKitCore"
        ),
        .target(
            name: "AFMOpenAICompat",
            path: "Candidate/Sources/AFMOpenAICompat"
        ),
        .target(
            name: "AFMKitInference",
            dependencies: ["AFMKitCore", "AFMOpenAICompat"],
            path: "Candidate/Sources/AFMKitInference"
        ),
        .target(
            name: "AFMEvalKit",
            dependencies: ["AFMOpenAICompat"],
            path: "Candidate/Sources/AFMEvalKit"
        ),
        .target(
            name: "AFMKitApple",
            dependencies: ["AFMKitCore", "AFMOpenAICompat"],
            path: "Candidate/Sources/AFMKitApple",
            linkerSettings: [
                .linkedFramework("Security"),
                .linkedFramework("ImageIO"),
                .linkedFramework("UniformTypeIdentifiers")
            ]
        ),
        .target(
            name: "AFMKitEmbeddings",
            path: "Candidate/Sources/AFMKitEmbeddings",
            linkerSettings: [.linkedFramework("NaturalLanguage")]
        ),
        .target(
            name: "AFMKitSpeech",
            dependencies: ["AFMKitCore"],
            path: "Candidate/Sources/AFMKitSpeech",
            linkerSettings: [.linkedFramework("Speech")]
        ),
        .target(
            name: "AFMKitSpeechSynthesis",
            dependencies: ["AFMKitCore"],
            path: "Candidate/Sources/AFMKitSpeechSynthesis",
            linkerSettings: [.linkedFramework("AVFoundation")]
        ),
        .target(
            name: "AFMKitVision",
            path: "Candidate/Sources/AFMKitVision",
            linkerSettings: [
                .linkedFramework("Vision"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("ImageIO"),
                .linkedFramework("PDFKit"),
                .linkedFramework("Quartz")
            ]
        ),
        .target(
            name: "AFMKitServices",
            dependencies: [
                "AFMKitEmbeddings",
                "AFMKitSpeech",
                "AFMKitSpeechSynthesis",
                "AFMKitVision"
            ],
            path: "Candidate/Sources/AFMKitServices"
        ),
        .target(
            name: "AFMXGrammar",
            path: "Candidate/Packages/AFMKitMLX/Sources/CXGrammar",
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
                "AFMKitCore",
                "AFMOpenAICompat",
                "AFMXGrammar",
                .product(name: "MLX", package: "mlx-swift-afm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface")
            ],
            path: "Candidate/Packages/AFMKitMLX/Sources/AFMKitMLX",
            resources: [.copy("Resources/default.metallib")],
            linkerSettings: [
                .linkedFramework("Security"),
                .linkedFramework("IOKit"),
                .linkedLibrary("IOReport"),
                .linkedLibrary("sqlite3")
            ]
        ),
        .target(
            name: "AFMKitFoundationModelsMLX",
            dependencies: ["AFMKitApple", "AFMKitCore", "AFMKitMLX"],
            path: "Candidate/Packages/AFMKitMLX/Sources/AFMKitFoundationModelsMLX"
        ),
        .target(
            name: "CDwarfStar",
            path: "Candidate/Packages/AFMKitDwarfStar/Sources/CDwarfStar",
            sources: [
                "AFMDwarfStarBridge.c",
                "CDwarfStarKVStore.c",
                "CDwarfStarEngine.c",
                "CDwarfStarDistributed.c",
                "CDwarfStarTensorParallel.c",
                "CDwarfStarSSD.c",
                "CDwarfStarMetal.m",
                "CDwarfStarLayerPack.c",
                "CDwarfStarGPUUnavailable.cpp"
            ],
            publicHeadersPath: "include",
            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("Metal")
            ]
        ),
        .target(
            name: "AFMKitDwarfStar",
            dependencies: [
                "AFMKitCore",
                "CDwarfStar",
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Xet", package: "swift-xet")
            ],
            path: "Candidate/Packages/AFMKitDwarfStar/Sources/AFMKitDwarfStar",
            resources: [.copy("../../vendor/ds4/metal")]
        ),
        .target(
            name: "AFMKitFoundationModelsDwarfStar",
            dependencies: ["AFMKitApple", "AFMKitCore", "AFMKitDwarfStar"],
            path: "Candidate/Packages/AFMKitDwarfStar/Sources/AFMKitFoundationModelsDwarfStar"
        ),
        .testTarget(
            name: "AFMKitCoreTests",
            dependencies: ["AFMKitCore"],
            path: "Candidate/Tests/AFMKitCoreTests"
        ),
        .testTarget(
            name: "AFMOpenAICompatTests",
            dependencies: ["AFMOpenAICompat"],
            path: "Candidate/Tests/AFMOpenAICompatTests"
        ),
        .testTarget(
            name: "AFMKitInferenceTests",
            dependencies: ["AFMKitInference", "AFMKitCore", "AFMOpenAICompat"],
            path: "Candidate/Tests/AFMKitInferenceTests"
        ),
        .testTarget(
            name: "AFMKitAppleTests",
            dependencies: ["AFMKitApple", "AFMKitCore"],
            path: "Candidate/Tests/AFMKitAppleTests"
        ),
        .testTarget(
            name: "AFMKitEmbeddingsTests",
            dependencies: ["AFMKitEmbeddings"],
            path: "Candidate/Tests/AFMKitEmbeddingsTests"
        ),
        .testTarget(
            name: "AFMKitSpeechTests",
            dependencies: ["AFMKitSpeech"],
            path: "Candidate/Tests/AFMKitSpeechTests"
        ),
        .testTarget(
            name: "AFMKitSpeechSynthesisTests",
            dependencies: ["AFMKitSpeechSynthesis"],
            path: "Candidate/Tests/AFMKitSpeechSynthesisTests"
        ),
        .testTarget(
            name: "AFMKitVisionTests",
            dependencies: ["AFMKitVision"],
            path: "Candidate/Tests/AFMKitVisionTests"
        ),
        .testTarget(
            name: "AFMKitServicesTests",
            dependencies: ["AFMKitServices"],
            path: "Candidate/Tests/AFMKitServicesTests"
        ),
        .testTarget(
            name: "AFMEvalKitTests",
            dependencies: ["AFMEvalKit"],
            path: "Candidate/Tests/AFMEvalKitTests"
        ),
        .testTarget(
            name: "AFMKitMLXTests",
            dependencies: [
                "AFMKitMLX",
                "AFMKitCore",
                "AFMOpenAICompat",
                .product(name: "MLXLMCommon", package: "mlx-swift-lm")
            ],
            path: "Candidate/Packages/AFMKitMLX/Tests/AFMKitMLXTests"
        ),
        .testTarget(
            name: "AFMKitFoundationModelsMLXTests",
            dependencies: ["AFMKitApple", "AFMKitCore", "AFMKitFoundationModelsMLX"],
            path: "Candidate/Packages/AFMKitMLX/Tests/AFMKitFoundationModelsMLXTests"
        ),
        .testTarget(
            name: "AFMKitDwarfStarTests",
            dependencies: ["AFMKitCore", "AFMKitDwarfStar", "CDwarfStar"],
            path: "Candidate/Packages/AFMKitDwarfStar/Tests/AFMKitDwarfStarTests"
        ),
        .testTarget(
            name: "AFMKitFoundationModelsDwarfStarTests",
            dependencies: [
                "AFMKitApple",
                "AFMKitCore",
                "AFMKitDwarfStar",
                "AFMKitFoundationModelsDwarfStar"
            ],
            path: "Candidate/Packages/AFMKitDwarfStar/Tests/AFMKitFoundationModelsDwarfStarTests"
        )
    ],
    cxxLanguageStandard: .gnucxx17
)
