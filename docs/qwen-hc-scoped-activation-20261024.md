# Qwen HC activation arithmetic investigation

This branch is an experiment, not a replacement for the preserved performance
builds. It combines the already isolated request-budget safeguard (`2235ab88`)
with the earlier HC normalization correction (`6598268f`, replayed as
`886d21a7`). The first activation probe was test-only. After it passed, the
candidate was wired into Qwen HC SiLU, sigmoid injection/mixing, and its BF16
sigmoid table. Shared MLX arithmetic and non-BF16 paths remain unchanged.

## Evidence

The saved layer-0 reference capture contains 2,773,440 BF16 HC activation
inputs (`[1, 8667, 320]`). The normalized/down-projection inputs already agree.
There are 6,767 differing activation outputs. Every one has input `-6.84375`
(BF16 bits `0xc0db`): the reference activation is `-0.00726318359375` and the
previous AFM activation is `-0.00732421875`.

The reference's source explicitly identifies this same BF16 sigmoid boundary
in its `SWIGLU_SOURCE` discussion. Source: `ddalcu/mlx-serve`, MIT license,
commit `02bee553f48cd3bc7d82aba0f8073820bd924738`, `src/transformer.zig`.
Copying an expression is insufficient if compiler math modes or intermediate
rounding differ. The native `hcSiluCallback` uses MLX sigmoid and multiply.

A CPU calculation against the captured tensor confirmed the rounding
requirements: round the exponential, denominator addition, reciprocal, sigmoid
result, and multiplication to BF16 at their native boundaries. That calculation
matches all 2,773,440 captured outputs. Omitting the exponential, addition, and
reciprocal intermediate roundings produces 738,801 mismatches even when the
sigmoid output is rounded. Omitting only reciprocal rounding produces 1,479.
This establishes arithmetic on this input set, not a GPU performance result.

## Qualification

`QwenHCScopedActivationTests` probes a model-scoped precise-exponential kernel.
It preserves the typed intermediate operations rather than replacing them with
an FP32 sigmoid. Gates and activations are checked against an exhaustive finite
BF16 table generated through the frozen reference's actual MLX-C library, and
against the captured full HC activation. The known boundary is a standalone
regression fixture. Fixture-dependent tests are explicitly opt-in diagnostics;
they do not create runtime environment requirements.

The initial probe and normalization tests passed 9/9: zero mismatches on the
65,280 finite BF16 inputs and the 2,773,440 captured HC outputs. The integrated
helper must repeat those tests, cover irregular extents and legacy dtypes, and
pass the existing compiled-HC and injection tests. It uses exact grid dispatch
without an additional scalar buffer or shape-specific shader specialization.

Then recapture the first downstream mismatch, replay the five original HTTP
requests, and compare full coding outcomes. A shorter generated answer alone
does not establish quality. Neither this test nor the earlier global sigmoid
experiment qualifies changes to other models.

Artifacts and scripts:
`/Volumes/edata/dev/CODEX/codex-local-coding-eval-20261007/`:

- `analyze-hc-bfloat-rounding-20261024.py` and its JSON result;
- `capture-hc-activation-oracle-20261024.py` for exact reference-library tables;
- original captures in `QwenNativeR5Layer0CaptureRetry20261012` and
  `QwenWarmR5HCNormReference20261012` (unchanged).

The original captured payload hashes are recorded in the CPU analysis JSON.
Library, metallib, and derived fixture hashes belong in the generated oracle
manifest. GPU diagnostics must run after the paired coding tasks, never during
their timing measurements.
