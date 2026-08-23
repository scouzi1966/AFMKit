// swift-tools-version: 6.1
import PackageDescription

let mlxNoCudaExcludes = [
    "mlx/mlx/backend/cuda/allocator.cpp", "mlx/mlx/backend/cuda/compiled.cpp",
    "mlx/mlx/backend/cuda/conv.cpp", "mlx/mlx/backend/cuda/cublas_utils.cpp",
    "mlx/mlx/backend/cuda/cudnn_utils.cpp", "mlx/mlx/backend/cuda/custom_kernel.cpp",
    "mlx/mlx/backend/cuda/delayload.cpp", "mlx/mlx/backend/cuda/device_info.cpp",
    "mlx/mlx/backend/cuda/device.cpp", "mlx/mlx/backend/cuda/eval.cpp",
    "mlx/mlx/backend/cuda/fence.cpp", "mlx/mlx/backend/cuda/indexing.cpp",
    "mlx/mlx/backend/cuda/jit_module.cpp", "mlx/mlx/backend/cuda/load.cpp",
    "mlx/mlx/backend/cuda/matmul.cpp", "mlx/mlx/backend/cuda/primitives.cpp",
    "mlx/mlx/backend/cuda/scaled_dot_product_attention.cpp",
    "mlx/mlx/backend/cuda/slicing.cpp", "mlx/mlx/backend/cuda/utils.cpp",
    "mlx/mlx/backend/cuda/worker.cpp", "mlx/mlx/backend/cuda/binary",
    "mlx/mlx/backend/cuda/conv", "mlx/mlx/backend/cuda/copy",
    "mlx/mlx/backend/cuda/device", "mlx/mlx/backend/cuda/gemms",
    "mlx/mlx/backend/cuda/quantized", "mlx/mlx/backend/cuda/reduce",
    "mlx/mlx/backend/cuda/steel", "mlx/mlx/backend/cuda/unary"
]

let mlxPlatformExcludes = [
    "mlx/mlx/backend/cpu/compiled.cpp", "mlx/mlx/backend/no_gpu",
    "mlx/mlx/backend/no_cpu", "mlx/mlx/backend/metal/no_metal.cpp",
    "mlx/mlx/backend/cpu/gemms/simd_fp16.cpp",
    "mlx/mlx/backend/cpu/gemms/simd_bf16.cpp"
] + mlxNoCudaExcludes

let mlxCTarget = Target.target(
    name: "Cmlx",
    path: "Candidate/vendor/MLX/mlx-swift/Source/Cmlx",
    exclude: mlxPlatformExcludes + [
        "vendor-README.md", "mlx-c/examples", "mlx-c/mlx/c/distributed.cpp",
        "mlx-c/mlx/c/distributed_group.cpp", "json", "fmt/test", "fmt/doc",
        "fmt/support", "fmt/src/os.cc", "fmt/src/fmt.cc",
        "mlx/mlx/backend/no_cpu/compiled.cpp", "mlx/ACKNOWLEDGMENTS.md",
        "mlx/CMakeLists.txt", "mlx/CODE_OF_CONDUCT.md", "mlx/CONTRIBUTING.md",
        "mlx/LICENSE", "mlx/MANIFEST.in", "mlx/README.md", "mlx/benchmarks",
        "mlx/cmake", "mlx/docs", "mlx/examples", "mlx/mlx.pc.in",
        "mlx/pyproject.toml", "mlx/python", "mlx/setup.py", "mlx/tests",
        "mlx/mlx/io/no_safetensors.cpp", "mlx/mlx/io/gguf.cpp",
        "mlx/mlx/io/gguf_quants.cpp", "mlx/mlx/backend/metal/kernels",
        "mlx/mlx/backend/metal/nojit_kernels.cpp",
        "mlx/mlx/distributed/mpi/mpi.cpp", "mlx/mlx/distributed/ring/ring.cpp",
        "mlx/mlx/distributed/nccl/nccl.cpp", "mlx/mlx/distributed/nccl/nccl_stub",
        "mlx/mlx/distributed/jaccl/jaccl.cpp", "mlx/mlx/distributed/jaccl/mesh.cpp",
        "mlx/mlx/distributed/jaccl/ring.cpp", "mlx/mlx/distributed/jaccl/utils.cpp"
    ],
    cSettings: [
        .headerSearchPath("mlx"), .headerSearchPath("mlx-c"),
        .headerSearchPath("mlx-generated/cuda")
    ],
    cxxSettings: [
        .headerSearchPath("metal-cpp"), .headerSearchPath("mlx"),
        .headerSearchPath("mlx-c"), .headerSearchPath("json/single_include/nlohmann"),
        .headerSearchPath("fmt/include"), .define("MLX_USE_ACCELERATE"),
        .define("ACCELERATE_NEW_LAPACK"), .define("_METAL_"),
        .define("SWIFTPM_BUNDLE", to: "\"AFMKitPrivateQualification_Cmlx\""),
        .define("METAL_PATH", to: "\"default.metallib\""),
        .define("MLX_VERSION", to: "\"0.31.1\"")
    ],
    linkerSettings: [
        .linkedFramework("Foundation"), .linkedFramework("Metal"),
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
    ] + releaseDependencyPins,
    targets: [
        .executableTarget(
            name: "encuda",
            dependencies: [.product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Candidate/vendor/MLX/mlx-swift/Source/Encuda"
        ),
        .plugin(
            name: "CudaBuild",
            capability: .buildTool(),
            dependencies: [.target(name: "encuda")],
            path: "Candidate/vendor/MLX/mlx-swift/Plugins/CudaBuild"
        ),
        mlxCTarget,
        .target(
            name: "MLX",
            dependencies: ["Cmlx", .product(name: "Numerics", package: "swift-numerics")],
            path: "Candidate/vendor/MLX/mlx-swift/Source/MLX",
            exclude: ["GPU+CUDA.swift"],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .target(
            name: "MLXRandom", dependencies: ["MLX"],
            path: "Candidate/vendor/MLX/mlx-swift/Source/MLXRandom",
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .target(
            name: "MLXFast", dependencies: ["MLX", "Cmlx"],
            path: "Candidate/vendor/MLX/mlx-swift/Source/MLXFast",
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .target(
            name: "MLXNN", dependencies: ["MLX"],
            path: "Candidate/vendor/MLX/mlx-swift/Source/MLXNN",
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .target(
            name: "MLXOptimizers", dependencies: ["MLX", "MLXNN"],
            path: "Candidate/vendor/MLX/mlx-swift/Source/MLXOptimizers",
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .target(
            name: "MLXFFT", dependencies: ["MLX"],
            path: "Candidate/vendor/MLX/mlx-swift/Source/MLXFFT",
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .target(
            name: "MLXLinalg", dependencies: ["MLX"],
            path: "Candidate/vendor/MLX/mlx-swift/Source/MLXLinalg",
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .target(
            name: "MLXLMCommon",
            dependencies: [
                "MLX", "MLXNN", "MLXOptimizers", "MLXFast",
                .product(name: "Transformers", package: "swift-transformers")
            ],
            path: "Candidate/vendor/MLX/mlx-swift-lm/Libraries/MLXLMCommon",
            exclude: ["README.md"],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency"), .swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXLLM",
            dependencies: [
                "MLXLMCommon", "MLX", "MLXNN", "MLXOptimizers", "MLXFast",
                .product(name: "Transformers", package: "swift-transformers")
            ],
            path: "Candidate/vendor/MLX/mlx-swift-lm/Libraries/MLXLLM",
            exclude: ["README.md"],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency"), .swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXVLM",
            dependencies: [
                "MLXLMCommon", "MLX", "MLXNN", "MLXOptimizers", "MLXFast",
                .product(name: "Transformers", package: "swift-transformers")
            ],
            path: "Candidate/vendor/MLX/mlx-swift-lm/Libraries/MLXVLM",
            exclude: ["README.md"],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency"), .swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXEmbedders",
            dependencies: [
                "MLX", "MLXNN", "MLXLMCommon",
                .product(name: "Transformers", package: "swift-transformers")
            ],
            path: "Candidate/vendor/MLX/mlx-swift-lm/Libraries/Embedders",
            exclude: ["README.md"],
            swiftSettings: [.swiftLanguageMode(.v5)]
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
                "MLX", "MLXLLM", "MLXVLM", "MLXLMCommon",
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
                "MLXLMCommon"
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
