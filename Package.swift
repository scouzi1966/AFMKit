// swift-tools-version: 6.1
import PackageDescription

var products: [Product] = [
    .library(name: "AFMKitCore", targets: ["AFMKitCore"]),
    .library(name: "AFMOpenAICompat", targets: ["AFMOpenAICompat"]),
    .library(name: "AFMKitInference", targets: ["AFMKitInference"])
]
var targets: [Target] = [
    .target(name: "AFMKitCore", dependencies: []),
    .target(name: "AFMOpenAICompat", dependencies: []),
    .target(
        name: "AFMKitInference",
        dependencies: ["AFMKitCore", "AFMOpenAICompat"]
    ),
    .testTarget(name: "AFMKitCoreTests", dependencies: ["AFMKitCore"]),
    .testTarget(name: "AFMOpenAICompatTests", dependencies: ["AFMOpenAICompat"]),
    .testTarget(
        name: "AFMKitInferenceTests",
        dependencies: ["AFMKitInference", "AFMKitCore", "AFMOpenAICompat"]
    )
]

#if compiler(>=6.4)
products.append(.library(name: "AFMKitApple", targets: ["AFMKitApple"]))
targets.append(
    .target(
        name: "AFMKitApple",
        dependencies: ["AFMKitCore", "AFMOpenAICompat"],
        linkerSettings: [.linkedFramework("Security")]
    )
)
targets.append(
    .testTarget(
        name: "AFMKitAppleTests",
        dependencies: ["AFMKitApple", "AFMKitCore"]
    )
)
#endif

let package = Package(
    name: "AFMKit",
    platforms: [
        .macOS("26.0")
    ],
    products: products,
    dependencies: [],
    targets: targets
)
