#!/bin/bash

afmkit_verify_qualified_toolchain() {
    local root="$1"
    local provenance="$root/docs/api-baselines/toolchain.env"
    local xcode_version_output actual_xcode_version actual_xcode_build
    local actual_sdk_version actual_sdk_build actual_swift_version actual_swift_sha
    local selected_developer_dir tool_path required_variable

    if [[ ! -f "$provenance" ]]; then
        echo "Missing API baseline toolchain provenance: $provenance" >&2
        return 1
    fi

    # shellcheck source=/dev/null
    source "$provenance"
    for required_variable in \
        API_BASELINE_XCODE_VERSION \
        API_BASELINE_XCODE_BUILD \
        API_BASELINE_MACOS_SDK_VERSION \
        API_BASELINE_MACOS_SDK_BUILD \
        API_BASELINE_SWIFT_VERSION \
        API_BASELINE_SWIFT_EXECUTABLE_SHA256; do
        if [[ -z "${!required_variable:-}" ]]; then
            echo "Missing $required_variable in $provenance" >&2
            return 1
        fi
    done

    AFMKIT_XCRUN_EXECUTABLE=/usr/bin/xcrun
    AFMKIT_XCODEBUILD_EXECUTABLE=/usr/bin/xcodebuild
    AFMKIT_XCODE_SELECT_EXECUTABLE=/usr/bin/xcode-select

    xcode_version_output="$($AFMKIT_XCODEBUILD_EXECUTABLE -version)"
    actual_xcode_version="$(printf '%s\n' "$xcode_version_output" | sed -n 's/^Xcode //p')"
    actual_xcode_build="$(printf '%s\n' "$xcode_version_output" | sed -n 's/^Build version //p')"
    actual_sdk_version="$($AFMKIT_XCRUN_EXECUTABLE --sdk macosx --show-sdk-version)"
    actual_sdk_build="$($AFMKIT_XCRUN_EXECUTABLE --sdk macosx --show-sdk-build-version)"

    AFMKIT_SWIFT_EXECUTABLE="$($AFMKIT_XCRUN_EXECUTABLE --toolchain XcodeDefault --find swift)"
    AFMKIT_SWIFT_SYMBOLGRAPH_EXECUTABLE="$($AFMKIT_XCRUN_EXECUTABLE --toolchain XcodeDefault --find swift-symbolgraph-extract)"
    selected_developer_dir="${DEVELOPER_DIR:-$($AFMKIT_XCODE_SELECT_EXECUTABLE -p)}"
    selected_developer_dir="$(cd "$selected_developer_dir" && pwd -P)"

    for tool_path in "$AFMKIT_SWIFT_EXECUTABLE" "$AFMKIT_SWIFT_SYMBOLGRAPH_EXECUTABLE"; do
        case "$tool_path" in
            "$selected_developer_dir"/Toolchains/XcodeDefault.xctoolchain/usr/bin/*) ;;
            *)
                echo "Qualified toolchain provenance failure: $tool_path is not from $selected_developer_dir/Toolchains/XcodeDefault.xctoolchain." >&2
                return 1
                ;;
        esac
    done

    actual_swift_version="$($AFMKIT_SWIFT_EXECUTABLE --version | sed -n '1p')"
    actual_swift_sha="$(/usr/bin/shasum -a 256 "$AFMKIT_SWIFT_EXECUTABLE" | awk '{print $1}')"

    if [[ "$actual_xcode_version" != "$API_BASELINE_XCODE_VERSION" ]] || \
       [[ "$actual_xcode_build" != "$API_BASELINE_XCODE_BUILD" ]] || \
       [[ "$actual_sdk_version" != "$API_BASELINE_MACOS_SDK_VERSION" ]] || \
       [[ "$actual_sdk_build" != "$API_BASELINE_MACOS_SDK_BUILD" ]] || \
       [[ "$actual_swift_version" != "$API_BASELINE_SWIFT_VERSION" ]] || \
       [[ "$actual_swift_sha" != "$API_BASELINE_SWIFT_EXECUTABLE_SHA256" ]]; then
        cat >&2 <<EOF
API baseline toolchain mismatch.
Required: Xcode $API_BASELINE_XCODE_VERSION ($API_BASELINE_XCODE_BUILD), macOS SDK $API_BASELINE_MACOS_SDK_VERSION ($API_BASELINE_MACOS_SDK_BUILD)
          $API_BASELINE_SWIFT_VERSION, executable SHA-256 $API_BASELINE_SWIFT_EXECUTABLE_SHA256
Current:  Xcode ${actual_xcode_version:-unknown} (${actual_xcode_build:-unknown}), macOS SDK ${actual_sdk_version:-unknown} (${actual_sdk_build:-unknown})
          ${actual_swift_version:-unknown}, executable SHA-256 ${actual_swift_sha:-unknown}
Select the qualified Xcode 27 toolchain, for example:
  export DEVELOPER_DIR=/Applications/Xcode-27.0.0-Beta.3.app/Contents/Developer
Then rerun the gate. To qualify a different toolchain, intentionally regenerate and review every API baseline and update:
  $provenance
EOF
        return 1
    fi

    export AFMKIT_XCRUN_EXECUTABLE
    export AFMKIT_XCODEBUILD_EXECUTABLE
    export AFMKIT_SWIFT_EXECUTABLE
    export AFMKIT_SWIFT_SYMBOLGRAPH_EXECUTABLE
}

afmkit_run_qualified_swift() {
    env \
        -u SWIFT_EXEC \
        -u SWIFT_DRIVER_SWIFT_FRONTEND_EXEC \
        -u TOOLCHAINS \
        "$AFMKIT_SWIFT_EXECUTABLE" "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    afmkit_verify_qualified_toolchain "$ROOT"
    printf 'Qualified Swift: %s\n' "$AFMKIT_SWIFT_EXECUTABLE"
fi
