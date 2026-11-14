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
