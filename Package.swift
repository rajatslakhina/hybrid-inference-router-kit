// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "HybridInference",
    // Only the platforms CI actually builds: macOS (swift test) and iOS (xcodebuild, Simulator SDK).
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "HybridInference", targets: ["HybridInference"]),
        .library(name: "HybridInferenceUI", targets: ["HybridInferenceUI"]),
    ],
    targets: [
        .target(name: "HybridInference"),
        .target(name: "HybridInferenceUI", dependencies: ["HybridInference"]),
        .testTarget(name: "HybridInferenceTests", dependencies: ["HybridInference", "HybridInferenceUI"]),
    ]
)
