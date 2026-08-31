# FLUX.2 Klein Swift source provenance

The model implementation in this target is adapted from
[`xocialize/flux2-klein-swift`](https://github.com/xocialize/flux2-klein-swift)
at commit `b554c4520e4ff2649b09376aa6176dffe511ea93` and
[`xocialize/flux2-vae-mlx-swift`](https://github.com/xocialize/flux2-vae-mlx-swift)
at commit `889223384f656efa481fd8e726dcffb3681554ef`.

The upstream Swift code is MIT licensed, Copyright (c) 2026 Xocialize. The
FLUX.2-klein-4B model weights are Apache-2.0 licensed by Black Forest Labs.
AFMKit carries these sources in one target so image generation shares AFMKit's
single pinned MLX runtime instead of linking a second, conflicting MLX package.
