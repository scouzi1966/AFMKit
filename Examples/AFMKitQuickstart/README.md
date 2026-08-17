# AFMKit Provider Quickstart

This executable demonstrates AFMKit's intended downstream boundary. It imports the public provider
contract plus MLX and Apple adapters without importing Vapor, the maclocal-api server, its CLI, or
its WebUI.

From the AFMKit repository root, run an MLX model on macOS 26 or newer:

```bash
swift run -c release \
  --package-path Examples/AFMKitQuickstart \
  afmkit-quickstart \
  mlx mlx-community/Qwen3.8-27B-4bit \
  "Why is the sky blue?"
```

On macOS 27, the same generation loop can use Apple Intelligence on device:

```bash
swift run -c release \
  --package-path Examples/AFMKitQuickstart \
  afmkit-quickstart \
  apple-on-device "Why is the sky blue?"
```

Or Private Cloud Compute:

```bash
swift run -c release \
  --package-path Examples/AFMKitQuickstart \
  afmkit-quickstart \
  apple-pcc "Analyze two competing implementation plans."
```

The PCC executable must be signed with a provisioning profile that contains the managed
`com.apple.developer.private-cloud-compute` entitlement. AFMKit reads that entitlement from the
signed host process; it cannot add or emulate it.

The example deliberately uses `AFMProviderRegistry`. An app can therefore keep its chat workflow
provider-neutral and select another AFMKit provider by changing registration and configuration
rather than rewriting its generation loop.

The relative package dependency is for development inside this repository. A downstream app should
replace it with the tagged repository dependency after AFMKit's first package tag is published:

```swift
.package(url: "https://github.com/scouzi1966/AFMKit.git", from: "0.1.0")
```
