# Qwen Next vision-wrapper growing text-prefix cache

## Failure and scope

The October 8, 2026 coding-agent investigation reproduced a lost reuse
opportunity with a text-only request loaded through `Qwen4ExpVL`. The first
rendered prompt had 6,772 tokens, the next turn 7,640, and their common token
prefix was 6,771. Appending the next turn changed the final newline token
from 198 to 271. A saved recurrent state at 6,772 cannot safely be trimmed
back to 6,771. The radix lookup correctly rejected that later state.

The September 21 serial growing-conversation fix (`e13bcbb434e7332a7f6c25407cd62db3b187391e`)
captured an earlier real checkpoint for `Qwen4ExpModel`, but its type guard
excluded `Qwen4ExpVL`. This is a missing wrapper-path qualification, not
evidence that arbitrary recurrent-cache trimming is safe. Only the first
captured transition has been attributed to this exact token divergence;
do not extrapolate the cause to every historical miss without replaying it.

## Fix

Extend the existing serial text backoff policy to the Qwen vision wrapper.
The existing eligibility guard still excludes media, quantized KV, empty
prompts and disabled prefix caching. Both streaming and non-streaming serial
paths use the same policy. No environment setting is required.

For this example, save the actual state at 6,741 in addition to the final
prompt boundary. The next turn restores 6,741 and processes only its 899-token
suffix. Identical repeats retain the existing full-boundary/logits reuse.
No batch admission policy, MTP verifier or media preparation is changed.

## Validation so far

- Release build passes against consumer `138ca5d` and provider base `68cb807b`.
- Two policy/radix tests pass, including the captured token-boundary pattern.
- Existing tiny hybrid snapshot test passes.
- New tiny VLM numerical test passes: restored suffix logits and primary
  cache state equal an independent same-chunk control exactly. Donor and
  repeated recipient advancement preserve the saved state.
- Broader cache/replay selection: 64 tests pass with zero failures, covering
  prefix policy, scheduler cache selection, exact-prompt retention and the
  new VLM regression cases. The separate tiny hybrid test also passed.
- Live replay: growing turn now reports 6,741 cached tokens (88.2% reuse),
  compared with zero in the released binary.
- Preliminary 32-output-token diagnostic: first request 5.90 seconds, exact
  repeat 0.58 seconds, growing turn 1.65 seconds. These are not release gates.
- An initial cold candidate run took 69 seconds for first prefill; it did
  not recur on subsequent runs. A process sample locates execution inside
  `Qwen4ExpMappedNGramTable.gatherFromMapping/dequantize`, not scheduling.
  Cold mapped-page access is a possibility, not a proven complete cause.
  Do not hide this observation or claim startup performance is qualified.

### Five frozen coding requests, explicit concurrency 1, MTP off

Both arms use the same saved inputs and checkpoint, temperature 0, top-p 1,
top-k 0, reasoning enabled, 8192 prefill step, prefix caching enabled and a
128-output-token cap. Generated tools are inspected, not executed.

| Measure | Released binary | Candidate |
|---|---:|---:|
| Growing-turn hits | 0/4 | 4/4 |
| Total reported prefill seconds | 42.616 | 9.742 |
| Output tokens / total generation seconds | 59.781 | 59.587 |

Reused boundaries are 6741, 7609, 9318 and 9475. First-use prefill is
5.785 versus 5.275 seconds. Responses are coherent but not token-identical:
changing chunk geometry changes greedy wording. The output cap truncates
some responses; this is not a full coding-project acceptance test.

### Uncached comparison and cold-start limitation

The candidate's first four Context sizes showed slower prefill than the
saved release run. Raw trials have identical input token counts; do not
compare the CSV's independently selected representative rows. A freshly
rebuilt control with wrapper backoff disabled also showed slower/variable
first-use results. The subsequent back-to-back released/fixed comparison,
at their preserved runtime paths, produced these all-three-trial means:

| Context | Released prefill / decode tok/s | Fixed prefill / decode tok/s | Change |
|---|---:|---:|---:|
| 0.5K | 915.39 / 67.19 | 922.85 / 68.15 | +0.81% / +1.44% |
| 1K | 1047.33 / 66.51 | 1057.42 / 66.95 | +0.96% / +0.66% |
| 2K | 1186.98 / 60.43 | 1199.38 / 61.47 | +1.04% / +1.72% |
| 4K | 1220.58 / 61.20 | 1232.21 / 61.52 | +0.95% / +0.53% |

This supports steady-state non-regression, not a claim that the initial
slow runs never occurred or that cold startup is fully qualified. All
initial raw trials remain saved in `context-vlm-fix` and
`context-backoff-control`; the matched rechecks are `context-release-recheck`
and `context-fix-recheck`. MTP was not qualified by this focused test.
Do not publish broader release claims from it.

### Complete coding-agent rerun

One unchanged dashboard fixture was implemented by the coding agent using
the preserved candidate, with no manual edits or interventions. The same
acceptance harness was used for all three runs:

| Run | Wall seconds | Automated checks | Requests | Output tokens |
|---|---:|---:|---:|---:|
| Previous released AFM | 877.6 | 13/15 | 37 | 24929 |
| Previous reference | 720.5 | 11/15 | 55 | 38964 |
| Fixed AFM | 554.9 | 15/15 | 41 | 27838 |

Fixed AFM reused 641,547 of 711,468 input tokens (90.2%), with 39/41
requests reporting reuse. The two misses were the initial request and
request 24, which removed all four tool definitions from the prompt.
The earlier conversation input remained unchanged, but the tools did not;
reusing state across that different prompt prefix would be unsafe.

This is one run per arm, not statistical quality superiority. Source output
and tool trajectories differ, so the full wall-time change must not be
attributed entirely to caching; the frozen-request replay isolates prefill.
Four generated tool commands failed during the fixed run; the agent
recovered, and the final application builds successfully.

Visual review adds an important limitation: the fixed application's table
headings and cell positions do not match. The previous reference also has
table-alignment defects, whereas the previous AFM desktop table is aligned.
The 15-check harness misses this visual/semantic issue. Do not describe
15/15 automated checks as perfect application quality or silently modify
the generated project. Desktop/mobile screenshots and final source remain
unaltered under the sibling `cacheFixVerification/` evidence directory.

The local workspace generator initially floated `swift-metrics` from 2.11
to 2.12. The candidate was rebuilt with the release lock as the seed; all
common remote dependency pins then matched. The focused patch passes the
measured growing-prefix and steady-state throughput checks; it has not been
merged, installed or qualified as a full release.

Evidence root (untracked, external disk):
`/Volumes/edata/dev/CODEX/codex-local-coding-eval-20261007/dashboardDiagnostic/`.
Preserved candidate runtime: `vlm-fix-runtime/afm`, SHA-256
`fd10ce7de5c6b67fff4067e6dffca98242033ab1046b36be2a005dc966b55de3`.
