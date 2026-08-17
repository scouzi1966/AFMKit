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
        )
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
        .testTarget(
            name: "AFMKitCoreTests",
            dependencies: ["AFMKitCore"]
        ),
        .testTarget(
            name: "AFMOpenAICompatTests",
            dependencies: ["AFMOpenAICompat"]
        )
    ]
)
