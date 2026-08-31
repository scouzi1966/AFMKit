# MLX GGUF patch queue

This directory is the temporary replay mechanism for native GGUF support while
the C and Swift API additions progress through mlx-c/mlx-swift upstream review.
The manifest pins the exact source versions and security fixes used by AFMKit.

From the AFMKit root:

```sh
Scripts/sync-mlx-gguf-patch.sh --status
Scripts/sync-mlx-gguf-patch.sh --check
Scripts/sync-mlx-gguf-patch.sh --apply
```

`--status` distinguishes applied, absent, and drifted patches. `--check` is the
CI form and fails unless the vendored tree contains the exact reverse-applicable
patch. `--apply` is idempotent and refuses source drift rather than attempting a
fuzzy merge.

The Package.swift GGUF target and pinned `vendor/gguflib` sources remain
AFMKit-owned integration files and are intentionally outside the upstream
mlx-swift patch.
