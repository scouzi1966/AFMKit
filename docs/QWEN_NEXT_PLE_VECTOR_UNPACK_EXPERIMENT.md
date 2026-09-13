# Qwen Next: CPU-vectorized sidecar unpacking

This AFM implementation experiment targets the scalar loop in
`Qwen4ExpMappedNGramTable.dequantize`, not the trained model or the hash
function. It is not a claim that SIMD unpacking is a new research technique.

The old generic loop re-reads packed words and scale/bias coefficients while
decoding individual columns. The optional q4 path reads one word for eight
values, unpacks its nibbles with Swift SIMD, reuses each group's coefficients,
and converts eight FP32 results to BF16 with the original rounding rule.
It uses unaligned input loads and `memcpy` for the UInt16-aligned output.

`AFM_QWEN_PLE_VECTOR_UNPACK=1` enables the experiment. The gate requires 4-bit
weights, a positive group size divisible by eight, and a row width divisible
by the group size. Other geometries and quantizations keep the generic path.
File geometry validation still precedes reads. Hashes, row offsets, sidecar
format, mmap/page-cache residency, target arithmetic, and sampling are unchanged.
No additional buffer, worker, GPU kernel or synchronization is introduced by
the vector loop. It is independent of committed-anchor repair.

Dispatch matters: the current table first attempts native worker reads for
gathers of at most 64 rows. That C++ path has its own unpacker and is unchanged
by this flag. Depth-4 MTP gathers 5 x 16 = 80 rows and therefore uses the Swift
mapped path; large prefill gathers do as well. Ordinary decode's 16 rows
normally use the native path, so AR decode is a negative control for this
experiment (its prefill can still exercise SIMD). The Swift fallback for
smaller failed/disabled native reads can also use the SIMD loop. Do not claim
that this setting alone vectorizes the existing native worker implementation.

## Qualification before full-model timing

- Byte-exact comparison against an independent nibble-by-nibble scalar oracle
  passed at row/group geometries 160/32, 256/64, 128/128 and 32/8.
- Tests cover deliberately unaligned weight/coefficient/output addresses,
  output sentinels, signed zeros, subnormals, large finite values and infinities.
- Unsupported geometries decline without writing the output.
- Existing mapped PLE history/rollback testing passed with the flag enabled;
  its group-size-4 fixture exercises the unchanged fallback, not SIMD. The
  byte oracle covers production group size 32; full-model checks cover its
  integration with the real checkpoint.
- The broader enabled pipeline/admission suite passed 64 tests; two optional
  timing tests were skipped. The unpack timing test passed separately.
- Release consumer build passed in 91.69 s. Inference binary SHA-256:
  `b06651e8a38df1d80b00f0960dd61aeda32375c2a6d3bd459b39c68d396734da`.

The isolated warm-row probe reported medians of 165.14 ns for its scalar
oracle and 48.15 ns for SIMD, about 3.4x. **This is not a measurement of the
full existing sidecar path**: the oracle is simpler than its generic unpacker,
the input is a tiny hot buffer, and hashing, random mapped reads, buffer
allocation, MLX-array creation and dependency waiting are excluded. Do not
claim that sidecar lookup or generation is 3.4x faster.

Full-model comparisons use the exact ddalcu checkpoint, first four contexts,
one excluded warmup plus three measured 128-token replies, and separate
greedy, top-p 0.95, top-p 1 and ordinary-generation controls. Anchor retention
is disabled in both arms. Only the vector-unpack setting changes within each
same-binary A/B. Known-answer and live API checks precede timing. Saved text
must match across enabled/disabled arms before attributing performance changes
to the unpacker.

Artifacts remain untracked under
`/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909`, with the
`ple-vector-*`, `test-ple-vector-*` and `build-ple-vector-*` prefixes.

## Completed full-model A/B

All 24 known-answer checks and 13 API checks passed. All **48 timed response
pairs** are text-identical across off/on (four modes, four contexts, three
trials). Median decode tok/s, shown as disabled / enabled:

| Context | Temp 0.6, top-p 0.95 MTP | Temp 0.6, top-p 1 MTP | Greedy MTP | Temp 0.6, top-p 1 AR |
|---|---:|---:|---:|---:|
| 0.5K | 88.92 / 88.62 | 105.79 / 105.58 | 100.85 / 100.13 | 67.88 / 68.12 |
| 1K | 85.88 / 85.66 | 77.90 / 78.06 | 94.12 / 93.24 | 66.54 / 67.29 |
| 2K | 74.57 / 74.91 | 84.49 / 85.88 | 90.99 / 90.92 | 61.54 / 61.39 |
| 4K | 91.78 / 91.79 | 87.79 / 88.12 | 70.66 / 69.79 | 60.41 / 60.68 |

Decode differences range from -1.2% to +1.6%; this is not evidence of a
material decode improvement. TTFT-derived prompt throughput increases by
0.1–1.3% across these points, but these small differences need independent
repeats before attributing them to the optimization. Pure device-prefill time
was not measured. The default remains unchanged; the isolated 3.4x row result
must not be promoted into a model-speedup claim.

The row-count dispatch boundary warrants a separate experiment using the
existing `AFM_QWEN_PLE_NATIVE_READS` control. That compares worker/positional
reads with direct mapping; it is not an isolated SIMD comparison and must be
reported separately. The constructor currently creates the native worker pool
when native reads are enabled, independently of the legacy Swift positional
fallback switch. These details are established from the current source, not
inferred from its comments about default read behavior.
