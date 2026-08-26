// swift-tools-version: 5.12
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let mlxSwiftDependency = Package.Dependency.package(path: "../mlx-swift")
let mlxSwiftPackageIdentity = "mlx-swift"

let package = Package(
    name: "mlx-swift-lm",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "MLXLLM",
            targets: ["MLXLLM"]),
        .library(
            name: "MLXVLM",
            targets: ["MLXVLM"]),
        .library(
            name: "MLXLMCommon",
            targets: ["MLXLMCommon"]),
        .library(
            name: "MLXEmbedders",
            targets: ["MLXEmbedders"]),
    ],
    dependencies: [
        mlxSwiftDependency,
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            from: "1.3.0"
        ),
    ],
    targets: [
        .target(
            name: "MLXLLM",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: mlxSwiftPackageIdentity),
                .product(name: "MLXNN", package: mlxSwiftPackageIdentity),
                .product(name: "MLXOptimizers", package: mlxSwiftPackageIdentity),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "MLXFast", package: mlxSwiftPackageIdentity),
            ],
            path: "Libraries/MLXLLM",
            exclude: [
                "README.md"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXVLM",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                .product(name: "MLX", package: mlxSwiftPackageIdentity),
                .product(name: "MLXNN", package: mlxSwiftPackageIdentity),
                .product(name: "MLXOptimizers", package: mlxSwiftPackageIdentity),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "MLXFast", package: mlxSwiftPackageIdentity),
            ],
            path: "Libraries/MLXVLM",
            exclude: [
                "README.md"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXLMCommon",
            dependencies: [
                .product(name: "MLX", package: mlxSwiftPackageIdentity),
                .product(name: "MLXNN", package: mlxSwiftPackageIdentity),
                .product(name: "MLXOptimizers", package: mlxSwiftPackageIdentity),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "MLXFast", package: mlxSwiftPackageIdentity),
            ],
            path: "Libraries/MLXLMCommon",
            exclude: [
                "README.md"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXEmbedders",
            dependencies: [
                .product(name: "MLX", package: mlxSwiftPackageIdentity),
                .product(name: "MLXNN", package: mlxSwiftPackageIdentity),
                .product(name: "Transformers", package: "swift-transformers"),
                .target(name: "MLXLMCommon"),
            ],
            path: "Libraries/Embedders",
            exclude: [
                "README.md"
            ]
        ),
        .testTarget(
            name: "MLXLMTests",
            dependencies: [
                .product(name: "MLX", package: mlxSwiftPackageIdentity),
                .product(name: "MLXNN", package: mlxSwiftPackageIdentity),
                .product(name: "MLXOptimizers", package: mlxSwiftPackageIdentity),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "MLXFast", package: mlxSwiftPackageIdentity),
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
            ],
            path: "Tests/MLXLMTests",
            exclude: [
                "README.md"
            ],
            resources: [.process("Resources/1080p_30.mov"), .process("Resources/audio_only.mov")],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "MLXLMIntegrationTests",
            dependencies: [
                .product(name: "MLX", package: mlxSwiftPackageIdentity),
                .product(name: "MLXNN", package: mlxSwiftPackageIdentity),
                .product(name: "MLXOptimizers", package: mlxSwiftPackageIdentity),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "MLXFast", package: mlxSwiftPackageIdentity),
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
            ],
            path: "Tests/MLXLMIntegrationTests",
            exclude: [
                "README.md"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "Benchmarks",
            dependencies: [
                "MLXLLM",
                "MLXVLM",
                "MLXLMCommon",
            ],
            path: "Tests/Benchmarks",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
    ]
)

if Context.environment["MLX_SWIFT_BUILD_DOC"] == "1"
    || Context.environment["SPI_GENERATE_DOCS"] == "1"
{
    // docc builder
    package.dependencies.append(
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0")
    )
}
