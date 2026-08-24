# mlx-audio-swift provenance

This directory is a reviewed source subset of
[`Blaizzy/mlx-audio-swift`](https://github.com/Blaizzy/mlx-audio-swift):

- upstream tag: `v0.1.2`
- upstream commit: `fcbd04d`
- imported targets: `MLXAudioCore`, `MLXAudioCodecs`, `MLXAudioTTS`
- license: `LICENSE` in this directory

AFMKit excludes the upstream applications, command-line tools, UI, speech
recognition targets, examples, and test media. The imported targets compile
against AFMKit's vendored MLX and MLX LM targets so downstream applications
resolve one compatible MLX graph. Consumers use the public `AFMKitMLXAudio`
product rather than importing these implementation targets directly.
