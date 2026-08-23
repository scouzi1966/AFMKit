// swift-tools-version: 6.1
import PackageDescription

var products: [Product] = [
    .library(name: "AFMKitCore", targets: ["AFMKitCore"]),
    .library(name: "AFMOpenAICompat", targets: ["AFMOpenAICompat"]),
    .library(name: "AFMKitInference", targets: ["AFMKitInference"]),
    .library(name: "AFMKitEmbeddings", targets: ["AFMKitEmbeddings"]),
    .library(name: "AFMKitSpeech", targets: ["AFMKitSpeech"]),
    .library(name: "AFMKitSpeechSynthesis", targets: ["AFMKitSpeechSynthesis"]),
    .library(name: "AFMKitVision", targets: ["AFMKitVision"]),
    .library(name: "AFMKitServices", targets: ["AFMKitServices"])
]
var targets: [Target] = [
    .target(name: "AFMKitCore", dependencies: []),
    .target(name: "AFMOpenAICompat", dependencies: []),
    .target(
        name: "AFMKitInference",
        dependencies: ["AFMKitCore", "AFMOpenAICompat"]
    ),
    .target(
        name: "AFMKitEmbeddings",
        dependencies: [],
        linkerSettings: [.linkedFramework("NaturalLanguage")]
    ),
    .target(
        name: "AFMKitSpeech",
        dependencies: ["AFMKitCore"],
        linkerSettings: [.linkedFramework("Speech")]
    ),
    .target(
        name: "AFMKitSpeechSynthesis",
        dependencies: ["AFMKitCore"],
        linkerSettings: [.linkedFramework("AVFoundation")]
    ),
    .target(
        name: "AFMKitVision",
        dependencies: [],
        linkerSettings: [
            .linkedFramework("Vision"),
            .linkedFramework("CoreGraphics"),
            .linkedFramework("ImageIO"),
            .linkedFramework("PDFKit"),
            .linkedFramework("Quartz")
        ]
    ),
    .target(
        name: "AFMKitServices",
        dependencies: [
            "AFMKitEmbeddings",
            "AFMKitSpeech",
            "AFMKitSpeechSynthesis",
            "AFMKitVision"
        ]
    ),
    .testTarget(name: "AFMKitCoreTests", dependencies: ["AFMKitCore"]),
    .testTarget(name: "AFMOpenAICompatTests", dependencies: ["AFMOpenAICompat"]),
    .testTarget(
        name: "AFMKitInferenceTests",
        dependencies: ["AFMKitInference", "AFMKitCore", "AFMOpenAICompat"]
    ),
    .testTarget(name: "AFMKitEmbeddingsTests", dependencies: ["AFMKitEmbeddings"]),
    .testTarget(name: "AFMKitSpeechTests", dependencies: ["AFMKitSpeech"]),
    .testTarget(name: "AFMKitSpeechSynthesisTests", dependencies: ["AFMKitSpeechSynthesis"]),
    .testTarget(name: "AFMKitVisionTests", dependencies: ["AFMKitVision"]),
    .testTarget(name: "AFMKitServicesTests", dependencies: ["AFMKitServices"])
]

#if compiler(>=6.4)
products.append(.library(name: "AFMKitApple", targets: ["AFMKitApple"]))
targets.append(
    .target(
        name: "AFMKitApple",
        dependencies: ["AFMKitCore", "AFMOpenAICompat"],
        linkerSettings: [
            .linkedFramework("Security"),
            .linkedFramework("CoreImage"),
            .linkedFramework("ImageIO"),
            .linkedFramework("UniformTypeIdentifiers")
        ]
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
