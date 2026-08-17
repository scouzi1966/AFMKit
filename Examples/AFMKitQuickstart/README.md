# AFMKit MLX Quickstart

This executable demonstrates the intended downstream integration boundary. It imports AFMKit's
public provider contract and MLX adapter without importing Vapor, the maclocal-api server, its CLI,
or its WebUI.

From the AFMKit repository root:

```bash
swift run -c release \
  --package-path Examples/AFMKitQuickstart \
  afmkit-quickstart \
  mlx-community/Qwen3.8-27B-4bit \
  "Why is the sky blue?"
```

The example deliberately uses `AFMProviderRegistry` instead of constructing an MLX service. An app
can therefore keep its chat workflow provider-neutral and select another AFMKit provider by changing
registration and configuration rather than rewriting its generation loop.

The relative package dependency is for development inside this repository. A downstream app should
replace it with the tagged repository dependency after AFMKit's first package tag is published:

```swift
.package(url: "https://github.com/scouzi1966/AFMKit.git", from: "0.1.0")
```
