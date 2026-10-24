# Qwen Next request reasoning budget candidate

This branch addresses one reproducible failure in the serial coding profile. It
does **not** establish coding-quality or elapsed-time parity with the reference,
and is not a release qualification.

## Why this change

The saved fifth request of the October 12 Orbital diagnostic has identical
model-visible inputs across engines. It requests `reasoning.effort: "medium"`.
With a reused prefix, AFM exhausted all 16,384 output tokens in reasoning and
returned no tool call. The reference returned a tool call in 1,824 tokens.

The reference maps medium effort to an 8,192-native-token reasoning limit.
Previously AFM passed the effort word to the chat template but did not enforce
that limit in generation. Its isolated native-token diagnostic recovered a tool
call in 8,747 tokens (142 seconds, versus 266 seconds without the limit). The
reference took 29 seconds and did not reach its limit. Thus the missing limit is
a confirmed missing safeguard, **not an explanation of the underlying longer
reasoning trajectory**.

Those are historical observations, not results of the new branch. New results
must record the binary hash, both source commits, exact checkpoint, request
hashes, launch arguments, sampling, prefix reuse, and returned calls.

## Scope and behavior

| Condition | Candidate behavior |
| --- | --- |
| Explicit minimal / low / medium effort | 1,024 / 2,048 / 8,192 native reasoning tokens on the eligible serial path |
| Effort omitted, high, xhigh, or unknown | Existing behavior; no implicit budget |
| Thinking disabled, closed template thinking header, other architectures | Existing behavior |
| Serial Qwen Next text, including text-only VLM with existing native-ID prefill | Budget supported |
| MTP, batch scheduler, images/video, logprobs, KV quantization, logit processors | Existing behavior; budget integration is not qualified in these paths |
| VLM input without a prepared native-ID prefill callback | Existing behavior; no alteration of model.prepare geometry |

The latter limitations mean this is an intermediate candidate, not universal
effort-budget support. No environment variable, model-path allowlist, or prompt
hash activates the policy. Checkpoint marker IDs and the transition suffix are
resolved through its tokenizer.

At the limit, generation commits the already-sampled pending token, then the
reference transition sentence and `</think>` separator as one forced span. It
resumes sampling from the last suffix row. The total output limit remains
unchanged. The transition is skipped if there is insufficient room for the
entire span and at least one answer token. Tokens already sampled or forced are
counted once; forced tokens receive no fabricated logprobs.

Only independently captured prompt snapshots enter the Qwen radix cache. A
cancelled or partially delivered forced span must never become a reusable
prompt snapshot. Iterator tests cover cancellation, ownership, partial delivery,
RNG draw counts, EOS, insufficient room, and native token accounting.

## Preservation and performance

The parent is `8b0ffa5c35197db9bf85dc389a348610a2e7b2df`, which retains the
tool-history, reasoning-preservation, raw-score ordering, and rejected-call text
fixes. The HC-normalization experiment at `6598268f` and uncommitted global
sigmoid changes are excluded. Existing worktrees and frozen binaries remain
untouched.

The per-token check is present only when a budget is armed. The mask validation
and tokenizer work are bypassed for requests that do not select this policy.
No model kernels, quantization, prompt template, prefill chunk size, or cache
geometry are changed. Validation must nevertheless compare actual outputs and
speed, not assume those facts guarantee non-regression.

## Acceptance sequence

1. Run the native-token iterator and request-policy tests.
2. Build the exact consumer `6839b0566dbd166f11ad4203db9568165ac8ff49` against
   this provider and record a clean source commit and binary checksum.
3. Replay the five saved independent HTTP requests with the same checkpoint,
   parameters and prefix state; execute none of the returned tool calls.
4. Check requests 1–4 against the previous exact outputs, and request 5 against
   the bounded diagnostic's recovered call. Compare fresh reference results.
5. Run a fresh coding task only after that check. Report task completion and
   elapsed time separately from token throughput. Broader repeated-task and
   unsupported-path qualification remain required before release.

Reference source: `mlx-serve` v26.10.1, commit
`02bee553f48cd3bc7d82aba0f8073820bd924738`, `src/responses.zig`
(`effortBudget`), `src/server.zig` (`armThinkBound`), and `src/generate.zig`
(`commitForcedTokens`). This policy uses the previously tested AFM native-token
primitive, rather than copying a text-level truncation into the HTTP layer.

## October 24 measured checkpoint

Provider code: `2235ab88ac94cfcecc4b3abf49d81ca82da6a354`.
Consumer: `6839b0566dbd166f11ad4203db9568165ac8ff49`.
Binary SHA-256: `a4bf9ab36acf0799e255c12736dbc830e4436575a95d1f4a27344fc62c016f96`.
The 12 focused tests and Release build passed. All dependency revisions match
the previous diagnostic build. No tuning environment variables were set.

On the exact five saved requests, requests 1–4 preserve the previous unbounded
outputs and usage. All five match the bounded diagnostic's semantic outputs
(ignoring generated IDs) and usage exactly, including a second process restart.
The fifth request's fresh measurements are:

| Binary | Output tokens | API seconds | Tool calls | Native generation tok/s |
| --- | ---: | ---: | ---: | ---: |
| Preserved best (`136d8e55`) | 14,963 | 244.25 | 1 | 61.51 |
| Isolated budget candidate (`2235ab88`) | 8,747 | 141.86 | 1 | 62.08 |
| Frozen reference (`02bee553`) | 1,824 | 29.13 | 1 | Not independently captured |

The reference's API output rate is 62.62 tokens/s, which includes prefill and
transport and must not be presented as its native decode measurement. These
are independent saved HTTP requests; returned tools were not executed. The
table does not measure completion or quality of an entire coding project.

The candidate's first pass had 53.50 seconds of first-request prefill; after a
restart that became 5.12 seconds (6.11 seconds API wall time). Outputs were
identical. The first pass also had a brief process sample during request 5.
Both passes are retained, and only the unprofiled repeat is used above. The
initial cold-start effect is not attributed to a particular cause from these
observations alone.

Artifacts are under
`/Volumes/edata/dev/CODEX/codex-local-coding-eval-20261007/`, in
`QwenRequestReasoningBudget20261024`,
`QwenRequestReasoningBudgetWarm20261024`,
`QwenRequestPreservedBest20261024`, and
`QwenRequestReasoningReference20261024`. The warm candidate contains
`REPORT.md` and `comparison.json` with all five cases and provenance.

### Retained numerical experiments and remaining gap

This isolated safeguard does **not supersede** the HC numerical work. The
following historical results on the same five-request fixture are retained:

| Experiment | Fifth-request tokens | API seconds | Qualification limit |
| --- | ---: | ---: | --- |
| HC normalization (`6598268f`, `QwenHCNormFive20261012`) | 4,030 | 67.78 | Numerical and limited coding tests; not a release gate |
| HC plus global precise sigmoid exponential (`QwenHCPreciseFive20261012`) | 2,064 | 37.06 | Uncommitted shared-kernel change; global model effects unqualified |

Those timings are historical, not the October 24 repeat. The native captured
layer-0 HC normalized and down-projection tensors can already match exactly.
The next recorded mismatch is the sigmoid/SiLU activation, before GDN inputs.
The saved activation test found 6,767 differing BF16 words with RMS error
approximately `3.01e-6`; separating sigmoid from multiply did not remove it.

The next numerical experiment should isolate the reference activation's
arithmetic in Qwen-owned code, compare against the actual captured tensors,
and then advance to the first remaining GDN mismatch. Do not promote the dirty
global exponential replacement or infer whole-task parity from its single
shorter response. Earlier GDN variants were tested with different upstream
activations; they are not proof that reference recurrence arithmetic is
irrelevant once upstream activations agree.

## Fresh full-task comparison

The October 24 Orbital pair used fresh directories, the identical initial
model-visible request, identical Codex launcher/catalog and harness files, and
the same checkpoint. The existing 180k-output-token/60-minute budget was used
for both engines. Session/message IDs differ; their contents do not. MTP was
off, thinking and prefix cache on, temperature 0, top-p 1, seed 123, and explicit
medium reasoning effort. No tuning environment variables were set.

| Measurement | Budget candidate | Reference |
| --- | ---: | ---: |
| Acceptance checks | 10/11 | 11/11 |
| Elapsed seconds | 1,010.16 | 481.57 |
| Output tokens | 57,587 | 27,897 |
| Requests | 38 | 30 |
| API seconds, summed | 993.84 | 466.48 |
| API output tokens/s, including prefill | 57.94 | 59.80 |
| Forced reasoning transitions | 2 | 0 |

The safeguard is **not sufficient for coding parity**. AFM took 2.098 times as
long and emitted 2.064 times as many tokens. Its native decode rate was 60.75
tokens/s, with 947.88s generation and 41.06s prefill. Reference native timing was
not logged by this frozen task profile; its API rate is not a raw-decode number.
AFM reused 869,374 of 904,345 accumulated prompt tokens (96.1%), so absent
prefix reuse is not the explanation for this run's elapsed gap.

AFM's failed check returned `boreal` instead of `Boreal` from selection
diagnostics. Its client also recorded two tool-format failures and one command
policy rejection. The reference passed all 11 checks and had no client tool
errors. Both HTTP runs completed without API errors. These are one independent
trajectory per engine, not a variance study or a causal quality comparison.

Reports and raw interactions are retained in `OrbitalRequestBudgetAFM20261024`
and `OrbitalRequestBudgetReference20261024`. The curated report, exact source
and binary identities, commands, and initial-request identity checks are in
`QwenRequestReasoningBudgetWarm20261024/ORBITAL-COMPARISON.md` and
`orbital-comparison.json`. The preserved-best October 13 11/11 result remains
intact; its 60k/20-minute profile has different prompt wording and is not pooled
with this pair. No release or main-branch promotion is justified by this result.
