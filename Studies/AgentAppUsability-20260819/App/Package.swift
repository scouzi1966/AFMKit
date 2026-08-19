// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "DecisionBrief",
    platforms: [.macOS("26.0")],
    products: [
        .executable(name: "DecisionBrief", targets: ["DecisionBrief"])
    ],
    dependencies: [
        .package(name: "AFMKit", path: "../../..")
    ],
    targets: [
        .target(
            name: "DecisionBriefCore",
            dependencies: [
                .product(name: "AFMKitCore", package: "AFMKit"),
                .product(name: "AFMKitMLX", package: "AFMKit")
            ]
        ),
        .executableTarget(name: "DecisionBrief", dependencies: ["DecisionBriefCore"]),
        .testTarget(name: "DecisionBriefCoreTests", dependencies: ["DecisionBriefCore"])
    ]
)
