#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="$ROOT_DIR/VendorPatches/mlx-gguf/mlx-swift-gguf.patch"
MANIFEST_FILE="$ROOT_DIR/VendorPatches/mlx-gguf/manifest.json"
MODE="${1:---status}"

if [[ ! -f "$PATCH_FILE" ]]; then
    echo "GGUF patch is missing: $PATCH_FILE" >&2
    exit 2
fi

is_applied() {
    python3 - "$ROOT_DIR" "$MANIFEST_FILE" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text())
for relative, expected in manifest["applied_sha256"].items():
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(1)
PY
}

is_absent() {
    git -C "$ROOT_DIR" apply --check "$PATCH_FILE" >/dev/null 2>&1
}

case "$MODE" in
    --status)
        if is_applied; then
            echo "applied"
        elif is_absent; then
            echo "not-applied"
        else
            echo "drifted"
            exit 1
        fi
        ;;
    --check)
        if is_applied; then
            echo "MLX GGUF patch is applied exactly."
        else
            echo "MLX GGUF patch is absent or has drifted." >&2
            exit 1
        fi
        ;;
    --apply)
        if is_applied; then
            echo "MLX GGUF patch is already applied."
        elif is_absent; then
            git -C "$ROOT_DIR" apply "$PATCH_FILE"
            echo "Applied MLX GGUF patch."
        else
            echo "Refusing to apply MLX GGUF patch to a drifted vendor tree." >&2
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 [--status|--check|--apply]" >&2
        exit 2
        ;;
esac
