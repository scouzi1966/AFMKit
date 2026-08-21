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
let releaseGraphProductPins: [Target.Dependency] = [
    .product(name: "AsyncHTTPClient", package: "async-http-client"),
    .product(name: "EventSource", package: "eventsource"),
    .product(name: "Algorithms", package: "swift-algorithms"),
    .product(name: "SwiftASN1", package: "swift-asn1"),
    .product(name: "AsyncAlgorithms", package: "swift-async-algorithms"),
    .product(name: "Atomics", package: "swift-atomics"),
    .product(name: "X509", package: "swift-certificates"),
    .product(name: "Collections", package: "swift-collections"),
    .product(name: "Configuration", package: "swift-configuration"),
    .product(name: "Crypto", package: "swift-crypto"),
    .product(name: "Instrumentation", package: "swift-distributed-tracing"),
    .product(name: "StructuredFieldValues", package: "swift-http-structured-headers"),
    .product(name: "HTTPTypes", package: "swift-http-types"),
    .product(name: "Logging", package: "swift-log"),
    .product(name: "NIOCore", package: "swift-nio"),
    .product(name: "NIOExtras", package: "swift-nio-extras"),
    .product(name: "NIOHTTP2", package: "swift-nio-http2"),
    .product(name: "NIOSSL", package: "swift-nio-ssl"),
    .product(name: "NIOTransportServices", package: "swift-nio-transport-services"),
    .product(name: "Numerics", package: "swift-numerics"),
    .product(name: "ServiceContextModule", package: "swift-service-context"),
    .product(name: "ServiceLifecycle", package: "swift-service-lifecycle"),
    .product(name: "SystemPackage", package: "swift-system")
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
            name: "AFMKitDwarfStarReleaseGraph",
            dependencies: releaseGraphProductPins
        ),
        .target(
            name: "AFMKitDwarfStar",
            dependencies: [
                .product(name: "AFMKitCore", package: "AFMKit"),
                "CDwarfStar",
                "AFMKitDwarfStarReleaseGraph",
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
