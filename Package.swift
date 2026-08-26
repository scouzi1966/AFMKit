// swift-tools-version: 6.1
import PackageDescription

// Keep no_cuda.cpp so Apple-platform builds link MLX's CUDA capability stubs.
// SwiftPM ignores CUDA translation units, but the C++ implementation and
// header-only subtrees must be excluded explicitly.
let mlxNoCudaExcludes = [
    "mlx/mlx/backend/cuda/allocator.cpp",
    "mlx/mlx/backend/cuda/compiled.cpp",
    "mlx/mlx/backend/cuda/conv.cpp",
    "mlx/mlx/backend/cuda/cublas_utils.cpp",
    "mlx/mlx/backend/cuda/cudnn_utils.cpp",
    "mlx/mlx/backend/cuda/custom_kernel.cpp",
    "mlx/mlx/backend/cuda/delayload.cpp",
    "mlx/mlx/backend/cuda/device.cpp",
    "mlx/mlx/backend/cuda/device_info.cpp",
    "mlx/mlx/backend/cuda/eval.cpp",
    "mlx/mlx/backend/cuda/fence.cpp",
    "mlx/mlx/backend/cuda/indexing.cpp",
    "mlx/mlx/backend/cuda/jit_module.cpp",
    "mlx/mlx/backend/cuda/load.cpp",
    "mlx/mlx/backend/cuda/matmul.cpp",
    "mlx/mlx/backend/cuda/primitives.cpp",
    "mlx/mlx/backend/cuda/scaled_dot_product_attention.cpp",
    "mlx/mlx/backend/cuda/slicing.cpp",
    "mlx/mlx/backend/cuda/utils.cpp",
    "mlx/mlx/backend/cuda/worker.cpp",
    "mlx/mlx/backend/cuda/binary",
    "mlx/mlx/backend/cuda/conv",
    "mlx/mlx/backend/cuda/copy",
    "mlx/mlx/backend/cuda/device",
    "mlx/mlx/backend/cuda/gemms",
    "mlx/mlx/backend/cuda/quantized",
    "mlx/mlx/backend/cuda/reduce",
    "mlx/mlx/backend/cuda/steel",
    "mlx/mlx/backend/cuda/unary"
]

let mlxPlatformExcludes = [
    "mlx/mlx/backend/cpu/compiled.cpp",
    "mlx/mlx/backend/no_gpu",
    "mlx/mlx/backend/no_cpu",
    "mlx/mlx/backend/metal/no_metal.cpp",
    "mlx/mlx/backend/cpu/gemms/simd_fp16.cpp",
    "mlx/mlx/backend/cpu/gemms/simd_bf16.cpp"
] + mlxNoCudaExcludes

let mlxCTarget = Target.target(
    name: "Cmlx",
    path: "vendor/MLX/mlx-swift/Source/Cmlx",
    exclude: mlxPlatformExcludes + [
        "vendor-README.md",
        "mlx-c/examples",
        "mlx-c/mlx/c/distributed.cpp",
        "mlx-c/mlx/c/distributed_group.cpp",
        "json",
        "fmt/test",
        "fmt/doc",
        "fmt/support",
        "fmt/src/os.cc",
        "fmt/src/fmt.cc",
        "mlx/mlx/backend/no_cpu/compiled.cpp",
        "mlx/ACKNOWLEDGMENTS.md",
        "mlx/CMakeLists.txt",
        "mlx/CODE_OF_CONDUCT.md",
        "mlx/CONTRIBUTING.md",
        "mlx/LICENSE",
        "mlx/MANIFEST.in",
        "mlx/README.md",
        "mlx/benchmarks",
        "mlx/cmake",
        "mlx/docs",
        "mlx/examples",
        "mlx/mlx.pc.in",
        "mlx/pyproject.toml",
        "mlx/python",
        "mlx/setup.py",
        "mlx/tests",
        "mlx/mlx/io/no_safetensors.cpp",
        "mlx/mlx/io/gguf.cpp",
        "mlx/mlx/io/gguf_quants.cpp",
        "mlx/mlx/backend/metal/kernels",
        "mlx/mlx/backend/metal/nojit_kernels.cpp",
        "mlx/mlx/distributed/mpi/mpi.cpp",
        "mlx/mlx/distributed/ring/ring.cpp",
        "mlx/mlx/distributed/nccl/nccl.cpp",
        "mlx/mlx/distributed/jaccl/jaccl.cpp",
        "mlx/mlx/distributed/jaccl/lib",
        "mlx/mlx/distributed/jaccl/jaccl.h"
    ],
    cSettings: [
        .headerSearchPath("mlx"),
        .headerSearchPath("mlx-c"),
        .headerSearchPath("mlx-generated/cuda")
    ],
    cxxSettings: [
        .headerSearchPath("metal-cpp"),
        .headerSearchPath("mlx"),
        .headerSearchPath("mlx-c"),
        .headerSearchPath("json/single_include/nlohmann"),
        .headerSearchPath("fmt/include"),
        .define("MLX_USE_ACCELERATE"),
        .define("ACCELERATE_NEW_LAPACK"),
        .define("_METAL_"),
        .define("SWIFTPM_BUNDLE", to: "\"AFMKit_Cmlx\""),
        .define("METAL_PATH", to: "\"default.metallib\""),
        .define("MLX_VERSION", to: "\"0.32.2\"")
    ],
    linkerSettings: [
        .linkedFramework("Foundation"),
        .linkedFramework("Metal"),
        .linkedFramework("Accelerate")
    ],
    plugins: [.plugin(name: "CudaBuild")]
)

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

// DwarfStar still uses a compatibility target to keep its complete release graph
// reachable. AFMKitMLX intentionally does not: downstream MLX consumers should
// compile only the products required by the MLX runtime. The exact package pins
// above remain audited against Package.resolved by the release dependency gate.
let dwarfStarReleaseGraphProductPins: [Target.Dependency] = [
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

var products: [Product] = [
    .library(name: "AFMKitCore", targets: ["AFMKitCore"]),
    .library(name: "AFMOpenAICompat", targets: ["AFMOpenAICompat"]),
    .library(name: "AFMKitInference", targets: ["AFMKitInference"]),
    .library(name: "AFMKitEmbeddings", targets: ["AFMKitEmbeddings"]),
    .library(name: "AFMKitSpeech", targets: ["AFMKitSpeech"]),
    .library(name: "AFMKitSpeechSynthesis", targets: ["AFMKitSpeechSynthesis"]),
    .library(name: "AFMKitVision", targets: ["AFMKitVision"]),
    .library(name: "AFMKitServices", targets: ["AFMKitServices"]),
    .library(name: "AFMEvalKit", targets: ["AFMEvalKit"]),
    .library(name: "AFMKitMLX", targets: ["AFMKitMLX"]),
    .library(name: "AFMKitMLXAudio", targets: ["AFMKitMLXAudio"]),
    .library(name: "AFMKitDwarfStar", targets: ["AFMKitDwarfStar"])
]

var targets: [Target] = [
    .executableTarget(
        name: "encuda",
        dependencies: [.product(name: "ArgumentParser", package: "swift-argument-parser")],
        path: "vendor/MLX/mlx-swift/Source/Encuda"
    ),
    .plugin(
        name: "CudaBuild",
        capability: .buildTool(),
        dependencies: [.target(name: "encuda")],
        path: "vendor/MLX/mlx-swift/Plugins/CudaBuild"
    ),
    mlxCTarget,
    .target(
        name: "MLX",
        dependencies: ["Cmlx", .product(name: "Numerics", package: "swift-numerics")],
        path: "vendor/MLX/mlx-swift/Source/MLX",
        exclude: ["GPU+CUDA.swift"],
        swiftSettings: [.enableExperimentalFeature("StrictConcurrency"), .swiftLanguageMode(.v5)]
    ),
    .target(
        name: "MLXRandom",
        dependencies: ["MLX"],
        path: "vendor/MLX/mlx-swift/Source/MLXRandom",
        swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
    ),
    .target(
        name: "MLXFast",
        dependencies: ["MLX", "Cmlx"],
        path: "vendor/MLX/mlx-swift/Source/MLXFast",
        swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
    ),
    .target(
        name: "MLXNN",
        dependencies: ["MLX"],
        path: "vendor/MLX/mlx-swift/Source/MLXNN",
        swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
    ),
    .target(
        name: "MLXOptimizers",
        dependencies: ["MLX", "MLXNN"],
        path: "vendor/MLX/mlx-swift/Source/MLXOptimizers",
        swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
    ),
    .target(
        name: "MLXFFT",
        dependencies: ["MLX"],
        path: "vendor/MLX/mlx-swift/Source/MLXFFT",
        swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
    ),
    .target(
        name: "MLXLinalg",
        dependencies: ["MLX"],
        path: "vendor/MLX/mlx-swift/Source/MLXLinalg",
        swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
    ),
    .target(
        name: "MLXLMCommon",
        dependencies: [
            "MLX", "MLXNN", "MLXOptimizers", "MLXFast",
            .product(name: "Transformers", package: "swift-transformers")
        ],
        path: "vendor/MLX/mlx-swift-lm/Libraries/MLXLMCommon",
        exclude: ["README.md"],
        swiftSettings: [.enableExperimentalFeature("StrictConcurrency"), .swiftLanguageMode(.v5)]
    ),
    .target(
        name: "MLXLLM",
        dependencies: [
            "MLXLMCommon", "MLX", "MLXNN", "MLXOptimizers", "MLXFast",
            .product(name: "Transformers", package: "swift-transformers")
        ],
        path: "vendor/MLX/mlx-swift-lm/Libraries/MLXLLM",
        exclude: ["README.md"],
        swiftSettings: [.enableExperimentalFeature("StrictConcurrency"), .swiftLanguageMode(.v5)]
    ),
    .target(
        name: "MLXVLM",
        dependencies: [
            "MLXLMCommon", "MLXLLM", "MLX", "MLXNN", "MLXOptimizers", "MLXFast",
            .product(name: "Transformers", package: "swift-transformers")
        ],
        path: "vendor/MLX/mlx-swift-lm/Libraries/MLXVLM",
        exclude: ["README.md"],
        swiftSettings: [.enableExperimentalFeature("StrictConcurrency"), .swiftLanguageMode(.v5)]
    ),
    .target(
        name: "MLXEmbedders",
        dependencies: [
            "MLX", "MLXNN", "MLXLMCommon",
            .product(name: "Transformers", package: "swift-transformers")
        ],
        path: "vendor/MLX/mlx-swift-lm/Libraries/Embedders",
        exclude: ["README.md"],
        swiftSettings: [.swiftLanguageMode(.v5)]
    ),
    .target(name: "AFMKitCore", dependencies: []),
    .target(name: "AFMOpenAICompat", dependencies: []),
    .target(name: "AFMKitInference", dependencies: ["AFMKitCore", "AFMOpenAICompat"]),
    .target(name: "AFMKitEmbeddings", dependencies: [], linkerSettings: [.linkedFramework("NaturalLanguage")]),
    .target(name: "AFMKitSpeech", dependencies: ["AFMKitCore"], linkerSettings: [.linkedFramework("Speech")]),
    .target(name: "AFMKitSpeechSynthesis", dependencies: ["AFMKitCore"], linkerSettings: [.linkedFramework("AVFoundation")]),
    .target(
        name: "AFMKitVision",
        dependencies: [],
        linkerSettings: [
            .linkedFramework("Vision"),
            .linkedFramework("CoreGraphics"),
            .linkedFramework("ImageIO"),
            .linkedFramework("PDFKit"),
            .linkedFramework("Quartz")
        ]
    ),
    .target(name: "AFMKitServices", dependencies: ["AFMKitCore", "AFMKitEmbeddings", "AFMKitSpeech", "AFMKitSpeechSynthesis", "AFMKitVision"]),
    .target(name: "AFMEvalKit", dependencies: ["AFMOpenAICompat"]),
    .target(
        name: "AFMXGrammar",
        dependencies: [],
        path: "Packages/AFMKitMLX/Sources/CXGrammar",
        sources: ["error_handler.cpp", "grammar_compiler.cpp", "grammar_matcher.cpp", "tokenizer_info.cpp", "xgrammar/cpp"],
        cxxSettings: [
            .headerSearchPath("xgrammar/include"),
            .headerSearchPath("xgrammar/cpp"),
            .headerSearchPath("xgrammar/3rdparty/dlpack/include"),
            .headerSearchPath("xgrammar/3rdparty/picojson"),
            .define("xgrammar", to: "afmkit_xgrammar"),
            .define("XGRAMMAR_ENABLE_LOG_DEBUG", to: "0"),
            .define("XGRAMMAR_ENABLE_CPPTRACE", to: "0")
        ]
    ),
    .target(
        name: "MLXAudioCore",
        dependencies: [
            "MLX", "MLXNN",
            .product(name: "HuggingFace", package: "swift-huggingface")
        ],
        path: "vendor/MLX/mlx-audio-swift/Sources/MLXAudioCore",
        linkerSettings: [.linkedFramework("AVFoundation")]
    ),
    .target(
        name: "MLXAudioCodecs",
        dependencies: [
            "MLXAudioCore", "MLX", "MLXNN", "MLXLMCommon",
            .product(name: "Tokenizers", package: "swift-transformers"),
            .product(name: "HuggingFace", package: "swift-huggingface")
        ],
        path: "vendor/MLX/mlx-audio-swift/Sources/MLXAudioCodecs",
        linkerSettings: [.linkedFramework("Accelerate"), .linkedFramework("AVFoundation")]
    ),
    .target(
        name: "MLXAudioTTS",
        dependencies: [
            "MLXAudioCore", "MLXAudioCodecs", "MLX", "MLXNN", "MLXLMCommon", "MLXLLM",
            .product(name: "Transformers", package: "swift-transformers"),
            .product(name: "Tokenizers", package: "swift-transformers"),
            .product(name: "Hub", package: "swift-transformers"),
            .product(name: "HuggingFace", package: "swift-huggingface")
        ],
        path: "vendor/MLX/mlx-audio-swift/Sources/MLXAudioTTS",
        exclude: [
            "Models/Marvis/README.md",
            "Models/Llama/README.md",
            "Models/PocketTTS/README.md",
            "Models/Qwen3/README.md",
            "Models/Soprano/README.md"
        ],
        linkerSettings: [.linkedFramework("Accelerate"), .linkedFramework("AVFoundation")]
    ),
    .target(
        name: "AFMKitMLX",
        dependencies: [
            "AFMKitCore", "AFMOpenAICompat", "AFMXGrammar",
            "MLX", "MLXLLM", "MLXVLM", "MLXLMCommon",
            .product(name: "Tokenizers", package: "swift-transformers"),
            .product(name: "Hub", package: "swift-transformers"),
            .product(name: "HuggingFace", package: "swift-huggingface")
        ],
        path: "Packages/AFMKitMLX/Sources/AFMKitMLX",
        resources: [.copy("Resources/default.metallib")],
        linkerSettings: [.linkedFramework("Security"), .linkedFramework("IOKit"), .linkedLibrary("IOReport"), .linkedLibrary("sqlite3")]
    ),
    .target(
        name: "AFMKitMLXAudio",
        dependencies: [
            "AFMKitMLX", "MLXAudioCore", "MLXAudioTTS", "MLX", "MLXLMCommon",
            .product(name: "HuggingFace", package: "swift-huggingface")
        ],
        path: "Packages/AFMKitMLXAudio/Sources/AFMKitMLXAudio",
        linkerSettings: [.linkedFramework("AVFoundation")]
    ),
    .target(
        name: "CDwarfStar",
        path: "Packages/AFMKitDwarfStar/Sources/CDwarfStar",
        sources: [
            "AFMDwarfStarBridge.c", "CDwarfStarKVStore.c", "CDwarfStarEngine.c", "CDwarfStarDistributed.c",
            "CDwarfStarTensorParallel.c", "CDwarfStarSSD.c", "CDwarfStarMetal.m", "CDwarfStarLayerPack.c",
            "CDwarfStarGPUUnavailable.cpp"
        ],
        publicHeadersPath: "include",
        linkerSettings: [.linkedFramework("Foundation"), .linkedFramework("Metal")]
    ),
    .target(name: "AFMKitDwarfStarReleaseGraph", dependencies: dwarfStarReleaseGraphProductPins, path: "Packages/AFMKitDwarfStar/Sources/AFMKitDwarfStarReleaseGraph"),
    .target(
        name: "AFMKitDwarfStar",
        dependencies: [
            "AFMKitCore", "CDwarfStar", "AFMKitDwarfStarReleaseGraph",
            .product(name: "HuggingFace", package: "swift-huggingface"),
            .product(name: "Xet", package: "swift-xet")
        ],
        path: "Packages/AFMKitDwarfStar/Sources/AFMKitDwarfStar",
        resources: [.copy("../../vendor/ds4/metal")]
    ),
    .testTarget(name: "AFMKitCoreTests", dependencies: ["AFMKitCore"]),
    .testTarget(name: "AFMOpenAICompatTests", dependencies: ["AFMOpenAICompat"]),
    .testTarget(name: "AFMKitInferenceTests", dependencies: ["AFMKitInference", "AFMKitCore", "AFMOpenAICompat"]),
    .testTarget(name: "AFMKitEmbeddingsTests", dependencies: ["AFMKitEmbeddings"]),
    .testTarget(name: "AFMKitSpeechTests", dependencies: ["AFMKitSpeech"]),
    .testTarget(name: "AFMKitSpeechSynthesisTests", dependencies: ["AFMKitSpeechSynthesis"]),
    .testTarget(name: "AFMKitVisionTests", dependencies: ["AFMKitVision"]),
    .testTarget(name: "AFMKitServicesTests", dependencies: ["AFMKitServices"]),
    .testTarget(name: "AFMEvalKitTests", dependencies: ["AFMEvalKit"]),
    .testTarget(
        name: "AFMKitMLXTests",
        dependencies: [
            "AFMKitMLX", "AFMKitCore", "AFMKitServices", "AFMOpenAICompat",
            "MLXLMCommon"
        ],
        path: "Packages/AFMKitMLX/Tests/AFMKitMLXTests"
    ),
    .testTarget(
        name: "AFMKitMLXAudioTests",
        dependencies: [
            "AFMKitMLXAudio", "AFMKitMLX", "MLXAudioCore",
            .product(name: "HuggingFace", package: "swift-huggingface")
        ],
        path: "Packages/AFMKitMLXAudio/Tests/AFMKitMLXAudioTests"
    ),
    .testTarget(name: "AFMKitDwarfStarTests", dependencies: ["AFMKitCore", "AFMKitDwarfStar", "CDwarfStar"], path: "Packages/AFMKitDwarfStar/Tests/AFMKitDwarfStarTests")
]

#if compiler(>=6.4)
products.append(.library(name: "AFMKitApple", targets: ["AFMKitApple"]))
products.append(.library(name: "AFMKitFoundationModelsMLX", targets: ["AFMKitFoundationModelsMLX"]))
products.append(.library(name: "AFMKitFoundationModelsDwarfStar", targets: ["AFMKitFoundationModelsDwarfStar"]))
targets.append(
    .target(
        name: "AFMKitApple",
        dependencies: ["AFMKitCore", "AFMOpenAICompat"],
        linkerSettings: [.linkedFramework("Security"), .linkedFramework("ImageIO"), .linkedFramework("UniformTypeIdentifiers")]
    )
)
targets.append(.target(name: "AFMKitFoundationModelsMLX", dependencies: ["AFMKitCore", "AFMKitApple", "AFMKitMLX"], path: "Packages/AFMKitMLX/Sources/AFMKitFoundationModelsMLX"))
targets.append(.target(name: "AFMKitFoundationModelsDwarfStar", dependencies: ["AFMKitApple", "AFMKitCore", "AFMKitDwarfStar"], path: "Packages/AFMKitDwarfStar/Sources/AFMKitFoundationModelsDwarfStar"))
targets.append(.testTarget(name: "AFMKitAppleTests", dependencies: ["AFMKitApple", "AFMKitCore"]))
targets.append(.testTarget(name: "AFMKitFoundationModelsMLXTests", dependencies: ["AFMKitApple", "AFMKitCore", "AFMKitFoundationModelsMLX"], path: "Packages/AFMKitMLX/Tests/AFMKitFoundationModelsMLXTests"))
targets.append(.testTarget(name: "AFMKitFoundationModelsDwarfStarTests", dependencies: ["AFMKitApple", "AFMKitCore", "AFMKitDwarfStar", "AFMKitFoundationModelsDwarfStar"], path: "Packages/AFMKitDwarfStar/Tests/AFMKitFoundationModelsDwarfStarTests"))
#endif

let package = Package(
    name: "AFMKit",
    platforms: [.macOS("26.0"), .iOS("16.0")],
    products: products,
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", exact: "0.9.0", traits: ["Xet"]),
        .package(url: "https://github.com/huggingface/swift-xet.git", exact: "0.2.3")
    ] + releaseDependencyPins,
    targets: targets,
    cxxLanguageStandard: .gnucxx20
)
