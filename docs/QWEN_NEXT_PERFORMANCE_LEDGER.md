# Qwen Next: peak-performance and degradation ledger

## Tracking contract

Keep historical peaks, current measurements, and quality qualification separate.
Do not replace a peak when a slower candidate is tested. A faster configuration
is not quality-qualified merely because its output is nonempty or coherent.

`Scripts/qwen-next-performance-ledger.py` records evidence paths, sample hashes,
executable hashes, launch options/environment, prompt identity, sampling, output
length, MTP state, cache/concurrency class, individual extrema and run medians.
Excluded samples retain a reason. Two comparisons are distinct:

- **Historical configuration envelope:** same checkpoint path, prompt hash/token
  length, sampling, output cap, MTP on/off and cache/concurrency class. Presets
  and MTP depths may differ. This is an observed gap, not attribution to a bug.
- **Same explicit launch:** additionally matches launch settings. Implicit
  defaults can still change between revisions. A mutable checkpoint path alone
  is not a content hash; never replace weights in this frozen experiment.

Summary peaks are the best run median from **at least three trials**, not the
fastest individual request. Warmups and instrumented runs do not enter warmed
peaks. Initial warmup delays remain available for cold-path investigations.

Client prefill is `prompt tokens / (TTFT - measured endpoint latency)`: a
**prefill proxy**, including first-token/API work, not isolated GPU prefill.
Decode is `(output tokens - 1) / (last text time - first text time)`. Neither
is concurrent aggregate throughput or successful agentic tasks/sec. Never
compare the existing 512-token agentic wall-time rates with this curve.

## September 15 checkpoint

M3 Ultra 512 GiB; unchanged ddalcu checkpoint:

```text
/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit
```

Provider branch `perf/qwen-next-mtp-parity`, HEAD `ceab5db8`; consumer `9acccfc9`.
Release inference executable SHA-256:
`10086d6504aa7dbd6e48ed5c8678cc0b1bdfd3a260a64836667de1a141b0b0c6`.
No inference rebuild, runtime default change, installation, merge or release
occurred in this increment.

Temperature 0.6, top-p 1, seed 42, 128 output tokens, thinking off, prefix cache
off, one request at a time. Actual prompt lengths: 493/864/2112/4150. Each cell
has one excluded warmup and three trials. These retain documented opt-ins;
they are **not unset-environment tests**.

### Quality-study preset (M25), MTP off/on

Each pair is **prefill proxy / decode**, in tokens/sec.

| Context | Non-MTP current | Non-MTP historical envelope | MTP depth 3 current | MTP historical envelope |
|---|---:|---:|---:|---:|
| 0.5K | 893.08 / 68.72 | 902.07 / 68.12 | 959.63 / 89.00 | 972.41 / 105.79 |
| 1K | 1045.58 / 68.28 | 1054.33 / 68.05 | 1119.43 / 86.45 | 1131.63 / 87.33 |
| 2K | 1190.89 / 62.00 | 1206.47 / 62.60 | 1276.50 / 81.85 | 1299.94 / 87.54 |
| 4K | 1293.68 / 60.77 | 1308.27 / 61.92 | 1261.43 / 82.92 | 1326.52 / 88.12 |

Envelope maxima can come from different runs: they are not one achievable
preset. Non-MTP remains within 1.9% of recorded decode peaks and 1.3% of
prefill peaks. MTP's apparent 0.5K deficit requires a depth-4 control.

### Historical-launch controls on today's binary

| Context | Depth 3 decode | Depth 4 decode | Depth 4 with explicit 8192 prefill |
|---|---:|---:|---:|
| 0.5K | 89.05 | 102.99 | 105.44 |
| 1K | 87.13 | 75.77 | not run |
| 2K | 82.33 | 85.67 | not run |
| 4K | 84.44 | 72.31 | 88.63 |

Depth 3 replays `frontier-d3-p1-v1`; depth 4 replays `ple-vector-p1-off-v1`.
The latter's historical 0.5K peak is 105.79 tok/s. Depth 4 recovers the short
context high watermark, but is **not uniformly faster than depth 3**.

At 4K the same explicit depth-4 launch falls 17.64% below its best historical
same-launch decode median. Adding `--prefill-step-size 8192` raises decode
from **72.31 to 88.63 (+22.57%)**, and the prefill proxy from **1268.33 to
1304.12 (+2.82%)**, on the same executable.

All three depth-3 and depth-4 responses match their respective historical
outputs exactly through 2K. At 4K, where the current 4096-token prefill policy
splits the prompt, they change. The 8192 override restores all three historical
depth-4 responses byte-for-byte. This isolates a policy/trajectory effect,
not a blanket loss of kernel speed.

Separate `AFM_DEBUG=1` captures (no phase profiler) expose the work difference:

| 4K, MTP depth 4 | Draft acceptance | Cycles / 128 tokens | Replays |
|---|---:|---:|---:|
| Explicit 4096 prefill | 46.7% (84/180) | 45 | 35 |
| Explicit 8192 prefill | 60.8% (90/148) | 37 | 21 |

Each repeats across four 128-token responses. Diagnostic timing is excluded
from performance peaks. The two paths produce different sampled summaries;
this is **not** proof that the larger chunk generates better answers, or that
all of its throughput gain is faster kernels.

The bounded-prefill repair is **not reverted**. It makes MTP honor the same
request policy as ordinary decoding and bounds initialization memory. Explicit
8192 still uses the repaired implementation. The user rejected automatic
per-request selection: keep explicit CLI control and measured guidance.
Changing a stable default would require quality/memory qualification and a
tradeoff discussion, not a speed-only decision. See the follow-up
[prefill tradeoff gate](QWEN_NEXT_PREFILL_TRADEOFFS.md).

### Cold-path and quality caveats

Initial non-MTP warmup TTFTs were 5.78/5.07/25.59/17.28 seconds; subsequent
measured trials were stable. These stalls remain in `AUDIT.json`. Their
individual JIT/mapping/filesystem causes have not been isolated.

The performance controls completed **54 measured requests + 18 excluded
warmups**. Eight additional diagnostic requests are recorded separately.
Selected unique outputs were inspected as coherent summaries, not AI-judged.
The frozen independent agentic screen remains 37/50 AFM MTP, 38/50 ordinary,
41/50 reference MTP, and 40/50 reference ordinary on unique-key JSON plus file
identity. **Quality parity is still open.** See the
[quality investigation](QWEN_NEXT_MTP_PREFILL_QUALITY.md).

## Evidence and reproduction

### September 18 refreshed reference and retention experiment

The [reference refresh](QWEN_NEXT_REFERENCE_REFRESH.md) pins released v26.9.4
and reruns the unchanged AFM control on the exact checkpoint. It supersedes
older reference comparisons without overwriting the historical AFM peaks.
Depth-3 C1 AFM decode is 88.16/85.26/80.13/84.58 tok/s across the first four
contexts; refreshed reference is 105.70/91.52/81.29/77.85. Short-context MTP
therefore misses the 10% gate. All 24 AFM measured AR/MTP responses match the
prior same-mode text; historical timing alone does not establish a regression.

New C15 no-prefix sampled AR is 47.08/47.52 first/repeat versus reference
54.82/55.66 (about 14% behind); do not hide that gap behind cached results.
With prefix cache, sampled MTP is 136.69/177.22 versus reference 126.53/143.66,
but correctness is 23/30 versus 24/30. Metrics and denominators remain separate.

The M28 opt-in preserves an earlier shared boundary within unchanged cache
limits. Same-binary reversed-order A/B pairs improve greedy second-repeat
162.50→185.10 and 160.15→186.90 aggregate tok/s; sampled repeats vary from
−2.6% to +2.4%, with no stable gain or loss. Despite unchanged 23/30 sampled
totals, pair B improves one case and regresses another. Keep M28 experimental;
do not replace a qualified default or a C1 Context peak with these C15 numbers.
Targeted two-class Release tests pass (95 plus one optional skip), as do 565
lifecycle assertions. A 45-prompt eviction screen completes but changes some
answers. Real model switching and sustained qualification remain open.

New untracked evidence/ledger root:

```text
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/reference-refresh-20260918
```

### September 18 boundary-work regression control

The [earlier-boundary experiment](QWEN_NEXT_PREFIX_BOUNDARY_EXPERIMENT.md)
replays the frozen C1/prefix-off M25 controls on binary
`21ee9bf82b49a361a240abd2fe09a3a416cf5cf9e794e95183a67805983387be`.
Boundary opt-ins are unset. All 24 measured outputs match the saved text.

| Context | Non-MTP prefill / decode | MTP depth 3 prefill / decode |
|---|---:|---:|
| 0.5K | 893.48 / 69.63 | 962.96 / 90.88 |
| 1K | 1042.31 / 69.25 | 1121.94 / 86.60 |
| 2K | 1200.45 / 62.99 | 1293.54 / 83.71 |
| 4K | 1305.33 / 61.94 | 1275.51 / 85.29 |

Same-launch decode is +0.18% to +2.85% and prefill proxy −0.31% to +1.34%
against September 15. No material regression is observed; this does not
attribute small changes to an inactive cache optimization. The historical
depth-4 105.79 short-context peak, 87.54 at 2K and 88.63 at 4K remain recorded.
Current depth-3 short-context decode is 14.09% below that different-preset
peak, not a same-launch regression. Eight initial warmups stay separate.

The separate C15/window-15 sampled cache screen measures first/repeat
MTP 54.37/186.11 → 137.94/182.75 aggregate tok/s; AR 87.60/159.24 →
119.92/158.67. Greedy MTP repeats still fall 14.3%, and AR adds one wrong
answer shape. These costs and correct-task totals are in the linked report;
no peak is overwritten and no default is promoted.

New append-only index and raw controls:

```text
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/prefix-boundary-20260917/peak-ledger
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/prefix-boundary-20260917/context-control-c
```

### Earlier retained follow-up

September 17 aggregate follow-up is recorded separately in
[retained qualification](QWEN_NEXT_RETAINED_QUALIFICATION.md). The same frozen
M25 binary measured 179.69 tok/s on sampled C15 repeats, versus the prior
184.18 (-2.4%, with slightly different generated text/token totals). No Context
curve is replaced by that value. A9 AR reached 154.36 on this new task set;
it must not be compared as a regression against the old 175 tok/s workload.
One-client scheduler replay improved wall-time throughput 47.95→120.70 tok/s
with all 90 paired texts exact, while decode-only rate was 128.66→126.42 and
first-use throughput 48.05→47.09. These costs remain visible alongside the win.

Reports remain external and untracked:

```text
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/performance-ledger-20260915
```

`current-a`, `peak-control-a`, `peak-depth4-a`, `depth4-prefill8192-a` retain
launches, raw streams, responses and timings. `audit-controls.py` is read-only.
The `*-diagnostic-a` folders are never timing baselines. Generated indexes are
append-only: select a new output directory, never overwrite an older index.

CPU-only tests (eight tests, no inference):

```sh
AFM_LEDGER_TEST_DIRECTORY=/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/performance-ledger-20260915 \
  python3 Scripts/tests/test_qwen_next_performance_ledger.py
```

Generate a new index:

```sh
python3 Scripts/qwen-next-performance-ledger.py \
  --evidence-root /Volumes/edata2/afm-benchmarks/qwen-next-rebaseline-20260909 \
  --evidence-root /Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909 \
  --current-root /Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/performance-ledger-20260915/current-a \
  --output /Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/performance-ledger-20260915/index-next
```

Coverage is the frozen September 9–15 Qwen Next context evidence, not every
AFM model or harness. Other checkpoints, batching, 512-token agentic work and
older differently defined plots need explicit metric adapters and provenance.
