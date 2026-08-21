// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "AFMKitQuickstart",
    platforms: [
        .macOS("26.0")
    ],
    dependencies: [
        .package(name: "AFMKit", path: "../.."),
        .package(name: "AFMKitMLX", path: "../../Packages/AFMKitMLX")
    ],
    targets: [
        .executableTarget(
            name: "AFMKitQuickstart",
            dependencies: [
                .product(name: "AFMKitCore", package: "AFMKit"),
                .product(name: "AFMKitApple", package: "AFMKit"),
                .product(name: "AFMKitMLX", package: "AFMKitMLX")
            ]
        )
    ]
)
