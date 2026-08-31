# gguflib vendoring

These sources are pinned from
[`antirez/gguf-tools`](https://github.com/antirez/gguf-tools) commit
`8fa6eb65236618e28fd7710a0fba565f7faa1848` and are licensed under BSD-2-Clause.

MLX's native GGUF implementation includes `gguflib.h`. The dedicated SwiftPM
target keeps that dependency explicit and makes GGUF-enabled builds
reproducible while support is proposed upstream in mlx-swift.
