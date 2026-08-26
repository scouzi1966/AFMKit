// swift-tools-version: 6.1
import Foundation
import PackageDescription

let afmKitPath = ProcessInfo.processInfo.environment["AFMKIT_PACKAGE_PATH"] ?? "../../.."

let package = Package(
    name: "AFMKitIOSCoreConsumerFixture",
    platforms: [.iOS("16.0")],
    products: [
        .library(
            name: "AFMKitIOSCoreConsumer",
            targets: ["AFMKitIOSCoreConsumer"]
        )
    ],
    dependencies: [
        .package(name: "AFMKit", path: afmKitPath)
    ],
    targets: [
        .target(
            name: "AFMKitIOSCoreConsumer",
            dependencies: [
                .product(name: "AFMKitCore", package: "AFMKit"),
                .product(name: "AFMOpenAICompat", package: "AFMKit"),
                .product(name: "AFMKitInference", package: "AFMKit")
            ]
        )
    ]
)
