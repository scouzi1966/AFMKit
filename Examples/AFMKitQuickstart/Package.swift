// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "AFMKitQuickstart",
    platforms: [
        .macOS("26.0")
    ],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "AFMKitQuickstart",
            dependencies: [
                .product(name: "AFMKitCore", package: "AFMKit"),
                .product(name: "AFMKitApple", package: "AFMKit"),
                .product(name: "AFMKitMLX", package: "AFMKit")
            ]
        )
    ]
)
