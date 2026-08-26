# Vendored MLX compatibility stack

AFMKit vendors its patched MLX dependency stack so a tagged checkout is complete,
anonymous, and reproducible. These directories are source snapshots, not nested Git
repositories or submodules.

| Directory | Upstream project | Upstream base | AFM compatibility revision |
| --- | --- | --- | --- |
| `mlx-swift/Source/Cmlx/mlx` | `ml-explore/mlx` | `v0.32.2` (`1f8e74e3f12f31365464a6867c6579f0e9b29d85`) | This AFMKit revision |
| `mlx-swift/Source/Cmlx/mlx-c` | `ml-explore/mlx-c` | `0726ca922fc902c4c61ef9c27d94132be418e945` | `1692252c78e634a90ae09bd77a9f68929982b8a0` |
| `mlx-swift` | `ml-explore/mlx-swift` | `0bb916c67f4b9e5c682cbe02a42c701c93ab5021` | `6000b7b26b70be2713c74e9ec2adeb89be07b9e5` |
| `mlx-swift-lm` | `ml-explore/mlx-swift-lm` plus AFM model adaptations | — | `e0d7fa71bc5e422a416f191c297264f698391561` |

The AFM changes add DeepSeek V4 MXFP4/Q8 Metal primitives, their C and Swift
bindings, AFM model architectures, parsing, generation behavior, and
quantization-aware Qwen MTP loading. Preserve each project's included `LICENSE`
file and the upstream revision table when refreshing a snapshot.

MLX 0.32.2 requires C++20. AFMKit keeps XGrammar in the same immutable package
and carries a focused compatibility adjustment for its recursive grammar IR so
the complete provider stack remains resolvable from one stable AFMKit tag.

Do not edit the vendored sources without also updating the focused provider tests,
the committed Metal library when kernel sources change, and this provenance table.
