# Qwen Next: consolidation and qualification order

September 15. Continue on PR #123; do not create a new optimization PR or
promote experimental defaults. This page is the entry point for the current
work, not a replacement for historical evidence.

## Decisions retained

| Area | Decision | Evidence |
|---|---|---|
| Context decode improvements | Preserve the exact working configurations and historical peaks | [Performance ledger](QWEN_NEXT_PERFORMANCE_LEDGER.md) |
| Shared/batched verification, cache/replay experiments | Retain opt-ins; combined qualification remains required | [Opt-in matrix](QWEN_NEXT_OPT_IN_MATRIX.md) |
| Bounded MTP initialization | Keep the correctness/memory repair; track trajectory-dependent speed differences | [Quality investigation](QWEN_NEXT_MTP_PREFILL_QUALITY.md) |
| Prefill 8192 as a general default | Rejected: no agentic throughput benefit in the paired screen, fewer strict passes | [Prefill tradeoffs](QWEN_NEXT_PREFILL_TRADEOFFS.md) |
| Precision/normalization changes | Diagnostic only; no proven whole-model quality fix | [Quality investigation](QWEN_NEXT_MTP_PREFILL_QUALITY.md) |
| New automatic tuning | Not authorized; keep explicit CLI choice and guidance | [Prefill tradeoffs](QWEN_NEXT_PREFILL_TRADEOFFS.md) |

## Ordered work

1. Commit the existing diagnostics, scripts, tests and findings. Preserve the
   previously tested executable and adjacent resources outside the repository.
2. Integrate the ten missing main commits into this feature branch, preserving
   Qwen changes. Main adds GLM/DeepSeek bounds, DwarfStar snapshot updates, and
   release-test gating. Its documented DSpARK parity skip is not a test pass.
3. Build through the consumer's reliable wrapper and run focused regressions.
   Recheck incoming memory/sparse-attention tests; report any failures explicitly.
4. Reproduce a fixed baseline before changing inference code: exact ddalcu
   checkpoint, saved prompts, temperature 0.6, top-p 1, fixed seeds, thinking off,
   C1/prefix off, MTP off and depth 3. Keep the existing documented M25 settings;
   do not describe these as unset-environment tests. Context and agentic metrics
   remain separate.
5. Resume the MTP-off numerical investigation, checking an unmodified reference
   diagnostic build against the frozen executable before trusting layer traces.
   Locate meaningful divergence before selecting a production change. A closer
   intermediate tensor alone is not proof of better output quality.
6. Once numerical/quality gates justify it, requalify retained MTP, prefix/radix
   cache, C15 aggregate throughput and cancellation together. Do not claim older
   isolated results certify the newly combined build.

The first-four-context curve uses 493/864/2112/4150 prompt tokens, 128 output
tokens, one excluded warmup and three trials per cell. Agentic replay uses the
saved five task families, five greedy controls and 50 sampled requests per mode,
with a 512-token cap. Report runtime, identity and strict structure separately;
the latter is not semantic judging or statistical proof of quality parity.

Record prefill proxy, decode-only rate, output tokens per wall second and
successful tasks/sec separately. Never substitute a new lower result for a
historical peak. Ask about performance/quality tradeoffs before adopting them.

Local evidence root:
`/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/consolidation-20260915`.
