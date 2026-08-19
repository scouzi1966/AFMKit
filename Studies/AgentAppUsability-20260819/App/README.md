# DecisionBrief App

DecisionBrief reads only `.txt`, `.md`, and `.markdown` files explicitly selected for the current session. It sends their in-memory contents only to the local AFMKit MLX provider using `mlx-community/Qwen3.8-27B-4bit`. It does not edit, export, retain, or transmit source content or generated briefs. AFMKit-managed model download and cache writes are the documented exception.

## Verify

```bash
swift package resolve
swift test -c release --disable-swift-testing
Scripts/build-release-app.sh
open .build/DecisionBrief.app
```

The bundling script creates an ad-hoc-signed Release app at `.build/DecisionBrief.app`. Build products remain ignored by the repository.
