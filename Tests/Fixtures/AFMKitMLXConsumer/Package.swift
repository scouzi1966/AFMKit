// swift-tools-version: 6.1
import Foundation
import PackageDescription

let afmKitPath = ProcessInfo.processInfo.environment["AFMKIT_PACKAGE_PATH"] ?? "../../.."

let package = Package(
    name: "AFMKitMLXConsumerFixture",
    platforms: [.macOS("26.0")],
    products: [
        .executable(name: "AFMKitMLXConsumer", targets: ["AFMKitMLXConsumer"])
    ],
    dependencies: [
        .package(name: "AFMKit", path: afmKitPath)
    ],
    targets: [
        .executableTarget(
            name: "AFMKitMLXConsumer",
            dependencies: [
                .product(name: "AFMKitCore", package: "AFMKit"),
                .product(name: "AFMKitMLX", package: "AFMKit")
            ]
        )
    ]
)
