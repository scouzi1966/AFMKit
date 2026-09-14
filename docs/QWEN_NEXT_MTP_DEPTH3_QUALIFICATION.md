# Qwen Next depth-3 experimental qualification

## Scope

Same ddalcu checkpoint on M3 Ultra, development binary SHA-256
`b06651e8a38df1d80b00f0960dd61aeda32375c2a6d3bd459b39c68d396734da`.
These are opt-in batched-verifier measurements, not installed/default performance.
Anchor retention and SIMD unpacking were disabled. Native positional reads were
disabled; mapped reads were used. The Swift `AffineRowGather` initializer was
inspected and does not independently disable the native implementation.

Artifacts and complete invocation manifests:
`/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909`.
See `driver-frontier-d3-validation-v1.log` and the corresponding run directories.

## Results

Decode medians in tokens/second, excluding warmup, three measured trials per
context, 128 generated tokens, seed 42. Context labels correspond to actual
493/864/2112/4150 prompt tokens. Sampled temperature is 0.6.

| Mode | 0.5K | 1K | 2K | 4K |
| --- | ---: | ---: | ---: | ---: |
| Greedy depth 3, first mapped run | 98.66 | 95.08 | 98.70 | 77.43 |
| Greedy depth 3, repeat | 98.84 | 95.08 | 97.98 | 77.70 |
| Sampled top-p 0.95, depth 3 | 96.34 | 93.35 | 81.10 | 74.91 |
| Sampled top-p 1, depth 3 | 90.32 | 87.33 | 82.34 | 72.69 |
| Fresh greedy reference | 94.71 | 84.47 | 94.07 | 84.47 |
| Earlier stronger sampled reference, top-p 0.95 | 90.08 | 90.20 | 87.65 | 85.33 |
| Earlier stronger sampled reference, top-p 1 | 87.28 | 88.51 | 88.83 | 80.37 |

Greedy repeated within 10% at all four contexts. Sampled top-p 1 also falls
within that band against the listed stronger reference, narrowly at 4K.
Top-p 0.95 still misses at 4K (approximately 12.2% behind). Do not claim global
parity: reference sampling can produce different text/acceptance workloads,
and these short runs do not establish broad quality or long-generation parity.

Known answers passed 24/24; the prefix-enabled API qualification passed 13
requests with 12 MTP round summaries. This is limited qualification, not proof
of general continuous batching or broad model quality.

## Interpretation and next experiment

The matching depth-3 native-read control measured 96.51/93.08/96.30/76.97;
mapped reads improved approximately 0.6–2.5%, with all 12 paired responses
identical. This is a small effect, not the explanation for all depth-3 gains.
Changing verification depth changes numerical geometry and potentially output
and acceptance, so depth-3 versus depth-4 is not an identical-token comparison.

Depth 4 previously did better on sampled 4K while depth 3 improved greedy 4K.
A useful next experiment is a request-local adaptive draft-depth policy driven
by accepted tokens per measured round cost, with bounded exploration and no
cross-request state. It must preserve rollback/cache invariants, measure its
own overhead, and be evaluated on held-out prompts rather than a context-label
lookup table. This policy is a proposal, not implemented by this checkpoint.

No defaults or installed release were changed.
