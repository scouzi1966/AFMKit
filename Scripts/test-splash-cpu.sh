#!/usr/bin/env bash
# Compile only the real Splash/Core sources and their CPU fixtures.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER="${AFM_SWIFTPM_WRAPPER:?Set AFM_SWIFTPM_WRAPPER to maclocal-api/Scripts/swiftpm-reliable.sh}"
WORK="$ROOT/.build-splash-cpu"
mkdir -p "$WORK"
python3 - "$ROOT" "$WORK" <<'PY'
from pathlib import Path
import sys
root, work = map(Path, sys.argv[1:])
for name, source in [("Core", root / "Sources/AFMKitCore"), ("Splash", root / "Sources/AFMKitSplash"), ("Tests", root / "Tests/AFMKitSplashTests")]:
    path = work / name
    if not path.exists():
        path.symlink_to(source)
(work / "Package.swift").write_text('''// swift-tools-version: 6.1
import PackageDescription
let package = Package(name: "SplashCPUValidation", platforms: [.macOS("26.0")], dependencies: [
    .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3")
], targets: [
    .target(name: "AFMKitCore", path: "Core"),
    .target(name: "AFMKitSplash", dependencies: ["AFMKitCore", .product(name: "Tokenizers", package: "swift-transformers")], path: "Splash", resources: [.process("Resources")]),
    .testTarget(name: "AFMKitSplashTests", dependencies: ["AFMKitSplash", "AFMKitCore"], path: "Tests", exclude: ["test_staging.py"], resources: [.copy("Fixtures")])
])
''')
PY
"$WRAPPER" test --package-path "$WORK" --scratch-path "$WORK/build" --filter SplashProviderTests
PYTHONDONTWRITEBYTECODE=1 python3 "$ROOT/Tests/AFMKitSplashTests/test_staging.py"
