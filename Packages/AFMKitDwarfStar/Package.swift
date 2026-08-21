// swift-tools-version: 6.1
import Foundation
import PackageDescription

let publicPackageDependency: Package.Dependency
let releaseDependencyPins: [Package.Dependency] = [
    .package(url: "https://github.com/swift-server/async-http-client", exact: "1.36.0"),
    .package(url: "https://github.com/mattt/EventSource.git", exact: "1.5.1"),
    .package(url: "https://github.com/apple/swift-algorithms.git", exact: "1.2.1"),
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
    .package(url: "https://github.com/apple/swift-log.git", exact: "1.15.0"),
    .package(url: "https://github.com/apple/swift-nio.git", exact: "2.101.3"),
    .package(url: "https://github.com/apple/swift-nio-extras.git", exact: "1.34.3"),
    .package(url: "https://github.com/apple/swift-nio-http2.git", exact: "1.45.0"),
    .package(url: "https://github.com/apple/swift-nio-ssl.git", exact: "2.37.2"),
    .package(url: "https://github.com/apple/swift-nio-transport-services.git", exact: "1.28.0"),
    .package(url: "https://github.com/apple/swift-numerics", exact: "1.1.1"),
    .package(url: "https://github.com/apple/swift-service-context.git", exact: "1.3.0"),
    .package(url: "https://github.com/swift-server/swift-service-lifecycle", exact: "2.11.0"),
    .package(url: "https://github.com/apple/swift-system.git", exact: "1.8.1")
]

if let publicPackagePath = ProcessInfo.processInfo.environment["AFMKIT_PUBLIC_PATH"],
   !publicPackagePath.isEmpty {
    publicPackageDependency = .package(name: "AFMKit", path: publicPackagePath)
} else {
    publicPackageDependency = .package(name: "AFMKit", path: "../..")
}

let package = Package(
    name: "AFMKitDwarfStar",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(
            name: "AFMKitDwarfStar",
            targets: ["AFMKitDwarfStar"]
        )
    ],
    dependencies: [
        publicPackageDependency,
        .package(
            url: "https://github.com/huggingface/swift-huggingface.git",
            exact: "0.9.0",
            traits: ["Xet"]
        ),
        .package(
            url: "https://github.com/huggingface/swift-xet.git",
            exact: "0.2.3"
        )
    ] + releaseDependencyPins,
    targets: [
        .target(
            name: "CDwarfStar",
            path: "Sources/CDwarfStar",
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
                .product(name: "AFMKitCore", package: "AFMKit"),
                "CDwarfStar",
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Xet", package: "swift-xet")
            ],
            resources: [
                .copy("../../vendor/ds4/metal")
            ]
        ),
        .testTarget(
            name: "AFMKitDwarfStarTests",
            dependencies: [
                .product(name: "AFMKitCore", package: "AFMKit"),
                "AFMKitDwarfStar",
                "CDwarfStar"
            ]
        )
    ],
    cxxLanguageStandard: .gnucxx17
)
