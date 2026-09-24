# Qwen Next native MLX checkpoint compatibility

Investigation uses `mlx-community/Qwen3.8-Flash-Next-4bit-mtp`, revision
`a53d7aa384247a095068485ac75f4383cd25e3fd`, without changing its files.

## Confirmed observations

- Installed `v0.9.20-next.20260922.a1d7461` repeats `_coordinates` with MTP off,
  both with thinking enabled and explicitly disabled.
- An independent native-layout loader generates `4` for the same arithmetic
  prompt from the same checkpoint. All 23 safetensor SHA-256 hashes match the
  pinned Hub revision. Download corruption is ruled out for the weight files.
- The native checkpoint retains zero-centered normalization weights. AFM's
  converter folds `1 + weight`, and AFM's existing sanitizer unconditionally
  undoes that fold. Applying the subtraction to native weights is incorrect.
- Native PLE embeddings use `.ngram_embedding.shards.N`; legacy converted
  embeddings use `.ngram_embedding.shard_N` before sanitization. Mapped AFM
  checkpoints explicitly declare `ngram_table`.
- The checkpoint includes 333 `vision_tower.*` tensors. AFM already has a
  Qwen3-VL tower in the Qwen4ExpVL wrapper. Presence is not image qualification.
- Native MTP uses `mtp.*`, mixes BF16 and affine quantization, and quantizes
  block-injection projections. Existing AFM embedded-MTP checks expect the
  converter's particular quantization pattern, while the tensor loader selects
  only `language_model.mtp.*`. Accepting the name alone is insufficient.

## Scope and validation gates

Preserve legacy/mapped checkpoint normalization and performance. Detect formats
from serialized layout, not model IDs or a heuristic over weight values. Keep
strict MTP tensor shape and completeness checks when supporting mixed precision.
Do not requantize the user's already-quantized MTP head silently.

Live release-build checks now pass for text generation, native MTP loading and
generation, image color identification, and a native weather tool call. The
vision wrapper's text-only preparation now delegates to the text model: it no
longer creates multimodal position state that blocks radix snapshot insertion.
An exact 978-token repeated text prompt reports all 978 tokens cached, with
0.05-second prompt time. This does not establish arbitrary partial-prefix reuse
or MTP radix-cache reuse. Image requests retain the existing position-state path.

Bounded 192-token responses were coherent, with roughly 30–35 generated tokens/s
across these probes. These are compatibility checks, not a controlled comparison
with another checkpoint or a complete quality/performance qualification.

Validation completed: 50 targeted regression tests passed, including legacy and
native normalization, missing/malformed MTP tensors, and text-only vision-wrapper
preparation. Final-build MTP recheck passed for text, tool calls, reasoning, and
a contrasting blue image. Image requests retain the non-speculative vision path.
This is not a full release suite, a vision benchmark, or proof of MTP prefix reuse.

Development binary (not installed over Homebrew):
`/Volumes/edata2/dev/CODEX/maclocal-api-error-status/.build/release/afm`.
SHA-256: `2646a0b824d15ff0a4950ae7ced8ca669618b583bc42d57d10d0bb2e10c468bd`.

Raw local evidence is retained at
`/Volumes/edata/afm-benchmarks/mlx-community-qwen-next-20260923`.
