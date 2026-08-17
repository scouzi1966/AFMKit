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
        )
    ],
    targets: [
        .target(
            name: "AFMKitCore",
            dependencies: []
        ),
        .testTarget(
            name: "AFMKitCoreTests",
            dependencies: ["AFMKitCore"]
        )
    ]
)
