# Qwen Next: independent semantic and combined-mode gate

September 16, 2026. Follow-up to the default-off GDN Q/K normalization experiment
on PR #123. This report preserves the earlier five-task quality evidence rather
than replacing it with a different test set. No inference code or defaults
changed during this gate.

## Why another screen

The original sampled screen checks unique-key JSON and file identity, not
whether the proposed repair is correct. Its 38/50 to 43/50 improvement is a
useful local signal, not proof of general answer quality. The fused candidate
preserves that prototype exactly and has no demonstrated material warm speed
advantage. Promotion requires more than repeating those five tasks.

`Scripts/qwen-next-broader-quality.py` adds 15 independent, fixed-answer service
review scenarios. These cover cache identity, LRU, cancellation isolation,
memory admission, causal masks, validation, split stop markers, dependency
ordering, retries, context budgets, prefix matching, quantized storage, lease
expiry, duplicate keys and an authorization repair with an untrusted comment.
Answers, request IDs and evidence-record IDs are checked; valid JSON alone
does not pass. No generated code is executed and no external AI judge is used.
This is a narrow semantic-constraint screen, not open-ended semantic evaluation.

The workload contains all 15 records in a common packet, then a per-request
question. Actual prompt counts are approximately 1K, not the previous 4K
file-selection workload. Exact payloads and prompt-token counts match between
AFM and the reference. A cancellation question is **not** a runtime cancellation
test. Similarly, cache questions do not themselves prove actual cache reuse.

## Controls and provenance

- Exact unchanged ddalcu checkpoint, template and weight-index hashes from the
  previous fixed baseline; mapped n-gram sidecar.
- Default-off executable SHA-256:
  `e7970fd1f444e13f23fc8f589b7dad29723216ea803697d7b02d71c4f85d7f91`.
- Private fused executable SHA-256:
  `f59804bc41db85f57366aa29d5c5a27d2a10a12d71bd820a7f0ad307d2fe8a3c`.
  Both use provider `2c9c8241`; the private build enables only the internal
  normalization diagnostic. Source activation was never committed.
- Preserved reference executable SHA-256:
  `f3ce20fba143e908110d44d1b8304bf305962a4600d23c9c112934112514bbe4`.
  Its frozen launch disables speculation other than MTP, attention quantization,
  prefix/tokenization caches and uses top-k 0. It was run live on these new tasks.
- Fifteen greedy controls plus 30 samples (two fixed distinct seed sets),
  temperature 0/0.6, top-p 1, 512-token cap, thinking off. One unrelated tiny
  warmup per process is excluded. No grammar constraints or forced parser.
- AFM retains the saved M25 opt-ins, MTP depth 3/batched verification. These
  are **not** unset-environment measurements. Same seeds do not promise equal
  RNG consumption between ordinary, speculative or concurrent execution.
- Single GPU owner, no builds during inference, 160 GiB pre-load and 100 GiB
  runtime available-memory floors, one-second process RSS sampling. RSS does
  not account for all Metal allocations; this is not a leak soak.

## C1, prefix off: completed

All six configurations complete 45/45 requests, with 45/45 structural and
request-identity checks. All outputs terminate normally, without token-cap
finishes or unexpected reasoning.

| Semantic passes | Default | Fused candidate | Reference |
|---|---:|---:|---:|
| MTP off, greedy | 13/15 | 12/15 | 12/15 |
| MTP off, sampled | 23/30 | 23/30 | 22/30 |
| MTP on, greedy | 12/15 | 12/15 | 12/15 |
| MTP on, sampled | 22/30 | 23/30 | 23/30 |

The candidate loses the greedy LRU answer in both modes. With MTP off there
are no sampled pass/fail changes. With MTP on it gains a greedy prefix-match
answer, and gains two sampled dependency-order answers while losing one LRU
answer. Thus equal totals do not mean identical failures. These small counts
do not establish statistical quality equivalence or superiority.

The reference reproduces the greedy memory-budget and quantized-storage
mistakes; those observed failures are not AFM-only. This does not attribute
every failure to model behavior or exonerate all runtime arithmetic. Other
failures differ by engine, mode and seed.

Sampled timing, including all cases whether right or wrong:

| Measure | Default AR | Fused AR | Reference AR | Default MTP | Fused MTP | Reference MTP |
|---|---:|---:|---:|---:|---:|---:|
| Median decode tok/s | 68.80 | 68.45 | 69.72 | 128.88 | 127.35 | 130.17 |
| Output / phase-wall tok/s | 34.33 | 34.16 | 34.59 | 47.69 | 47.08 | 45.47 |
| Correct tasks/s | 0.3932 | 0.4014 | 0.3916 | 0.5217 | 0.5522 | 0.5352 |

These are short, structured answers around 1K context, not new Context-curve
peaks. MTP accelerates this workload but does not guarantee better answers.
Phase-wall rates include prefill and harness bookkeeping. The initial default
greedy arm includes a 14.52-second first full-packet request after a tiny
warmup; do not present its difference from later greedy arms as a speedup.
The sampled phases are warmed, but this remains one ordered comparison,
not a counterbalanced repeated performance bound.

**Decision:** keep the fused normalization default-off. The local five-task
gain has not generalized convincingly, and there is an explicit greedy
regression. Retain the implementation and evidence as an experiment; do not
roll back earlier qualified optimizations or overwrite their peak ledger.

## Combined-mode checks

Controlled first/repeat screens are planned for C15/prefix off, C1/prefix on,
and C15/prefix on with both binaries and MTP off/on. They collect actual client
overlap and cache usage, aggregate output tok/s and correct tasks/s. They do
not promote the candidate or waive the quality caveat above.

The same M25 controls are retained to isolate the normalization change. In
particular, this is **not** the separately documented fastest AR request-banked
recipe, which enables additional mixed-position/GDN/banking controls. Do not
interpret this matrix as an AR maximum-throughput search. The candidate variant
only applies to B1 prefill of at least 128 tokens; short cached suffixes and
batched prefill can take the unchanged arithmetic path.

Runtime cancellation, longer-context semantic tasks, broader open-ended quality,
and full release regressions remain separate gates.

## Evidence

External, append-only root:

```text
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/consolidation-20260915
```

`BROADER-QUALITY-PROTOCOL.md` was recorded before inference. `broader-c1-a` and
`broader-reference-c1-a` preserve launches, full fixtures, requests, raw SSE,
scores, timing and guards. `AUDIT-BROADER-C1.json` and
`AUDIT-BROADER-REFERENCE-C1.json` recheck all recorded scores and hashes.
Preserved runners v1/v2 permit audits after the harness evolves. Twelve CPU
tests cover scorer failures, fixture identity, exact launch transformations,
aggregate denominators and actual client-overlap accounting.
