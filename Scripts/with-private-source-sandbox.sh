#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: ${0##*/} build-root command [args ...]" >&2
    exit 64
fi
if [[ ! -x /usr/bin/sandbox-exec ]]; then
    echo "Private qualification requires macOS sandbox-exec for candidate compilation." >&2
    exit 78
fi

BUILD_ROOT="$1"
shift
PRIVATE_PATHS=(
    "$BUILD_ROOT/checkouts/mlx-swift-afm"
    "$BUILD_ROOT/checkouts/mlx-swift-lm"
    "$BUILD_ROOT/repositories"
)
for private_path in "${PRIVATE_PATHS[@]}"; do
    if [[ ! -e "$private_path" ]]; then
        echo "Expected private qualification path is missing: $private_path" >&2
        exit 1
    fi
done

umask 077
WRAPPER_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/afmkit-compiler-sandbox.XXXXXX")"
cleanup() {
    find "$WRAPPER_ROOT" -depth -delete
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

REAL_SWIFTC="$(/usr/bin/xcrun --toolchain XcodeDefault --find swiftc)"
REAL_CLANG="$(/usr/bin/xcrun --toolchain XcodeDefault --find clang)"
REAL_CLANGXX="$(/usr/bin/xcrun --toolchain XcodeDefault --find clang++)"
AFMKIT_COMPILER_SANDBOX_LOG="$WRAPPER_ROOT/invocations.log"
export REAL_SWIFTC REAL_CLANG REAL_CLANGXX AFMKIT_COMPILER_SANDBOX_LOG

/usr/bin/python3 - "$WRAPPER_ROOT" "${PRIVATE_PATHS[@]}" <<'PY'
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
private_paths = [str(pathlib.Path(path).resolve()) for path in sys.argv[2:]]

def sandbox_string(value: str) -> str:
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'

profile = ["(version 1)", "(allow default)"]
for path in private_paths:
    profile.append(f"(deny file-read* (subpath {sandbox_string(path)}))")
(root / "profile.sb").write_text("\n".join(profile) + "\n", encoding="utf-8")

wrapper = root / "compiler-wrapper"
wrapper.write_text(
    """#!/bin/bash
set -euo pipefail
case "${0##*/}" in
    swiftc) compiler="$REAL_SWIFTC" ;;
    clang) compiler="$REAL_CLANG" ;;
    clang++) compiler="$REAL_CLANGXX" ;;
    *) echo "Unknown compiler wrapper: ${0##*/}" >&2; exit 64 ;;
esac
printf '%s\n' "${0##*/}" >> "$AFMKIT_COMPILER_SANDBOX_LOG"
exec /usr/bin/sandbox-exec -f "$(dirname "$0")/profile.sb" "$compiler" "$@"
""",
    encoding="utf-8",
)
wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
for name in ("swiftc", "clang", "clang++"):
    (root / name).symlink_to(wrapper.name)
PY

set +e
SWIFT_EXEC="$WRAPPER_ROOT/swiftc" \
    CC="$WRAPPER_ROOT/clang" \
    CXX="$WRAPPER_ROOT/clang++" \
        "$@"
command_status=$?
set -e
if [[ "$command_status" -ne 0 ]]; then
    exit "$command_status"
fi

IFS=',' read -r -a required_compilers \
    <<< "${AFMKIT_REQUIRE_SANDBOX_COMPILERS:-}"
for compiler in "${required_compilers[@]}"; do
    [[ -z "$compiler" ]] || grep -qx "$compiler" "$AFMKIT_COMPILER_SANDBOX_LOG" \
        || { echo "Required sandboxed compiler was not invoked: $compiler" >&2; exit 1; }
done
