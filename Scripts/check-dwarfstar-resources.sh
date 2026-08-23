#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="$ROOT/vendor/ds4"
PACKAGE_COPY="$ROOT/Packages/AFMKitDwarfStar/vendor/ds4"
SNAPSHOT_FILES=(
    LICENSE
    ds4.c
    ds4.h
    ds4_distributed.c
    ds4_distributed.h
    ds4_gpu.h
    ds4_gpu_args.h
    ds4_gpu_mgpu.h
    ds4_kvstore.c
    ds4_kvstore.h
    ds4_layer_pack.c
    ds4_layer_pack.h
    ds4_metal.m
    ds4_ssd.c
    ds4_ssd.h
    ds4_streaming_hotlist.inc
    ds4_streaming_hotlist_glm52.inc
    ds4_tp.c
    ds4_tp.h
)

if find "$PACKAGE_COPY" -type l -print -quit | grep -q .; then
    echo "The distributable DwarfStar snapshot must not contain symlinks." >&2
    exit 1
fi

EXPECTED_FILES="$(printf '%s\n' "${SNAPSHOT_FILES[@]}" | sort)"
ACTUAL_FILES="$(find "$PACKAGE_COPY" -maxdepth 1 -type f -exec basename {} \; | sort)"
if [[ "$ACTUAL_FILES" != "$EXPECTED_FILES" ]]; then
    echo "The DwarfStar source snapshot contains unexpected or missing root files." >&2
    diff -u <(printf '%s\n' "$EXPECTED_FILES") <(printf '%s\n' "$ACTUAL_FILES") || true
    exit 1
fi

for source_file in "${SNAPSHOT_FILES[@]}"; do
    if ! cmp -s "$UPSTREAM/$source_file" "$PACKAGE_COPY/$source_file"; then
        echo "DwarfStar snapshot drifted from vendor/ds4/$source_file." >&2
        exit 1
    fi
done
if ! diff -qr "$UPSTREAM/metal" "$PACKAGE_COPY/metal"; then
    echo "DwarfStar Metal snapshot drifted from vendor/ds4/metal." >&2
    exit 1
fi

echo "DwarfStar package source and resources match the pinned ds4 submodule."
