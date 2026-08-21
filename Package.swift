// swift-tools-version: 6.1
import PackageDescription

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
        )
    ],
    dependencies: [],
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
            dependencies: ["AFMKitCore", "AFMOpenAICompat"],
            linkerSettings: [
                .linkedFramework("Security")
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
        )
    ]
)
