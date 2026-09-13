# Qwen Next MTP memory comparison — September 10, 2026

Same ddalcu checkpoint and experimental AFM depth-3 configuration described in
QWEN_NEXT_MTP_DEPTH3_QUALIFICATION.md. Reference uses its adaptive MTP policy.
Greedy decoding, prefix reuse disabled, 128 generated tokens, one warmup and
three measured requests per context. Servers ran sequentially.

| Context | Reference peak RSS GiB | AFM peak RSS GiB | Reference decode tok/s | AFM decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 0.5K | 70.37 | 70.55 | 94.26 | 95.65 |
| 1K | 70.38 | 70.56 | 85.35 | 93.76 |
| 2K | 70.38 | 70.57 | 96.02 | 93.22 |
| 4K | 70.39 | 70.57 | 77.45 | 75.81 |

Memory is maximum sampled process RSS across measured requests, sampled every
50 ms. Throughput is the median of three trials. RSS includes resident mapped
pages; it is not Metal allocation accounting or macOS physical footprint.
Short warmed requests do not establish long-context leak/OOM safety. AFM used
about 0.2 GiB more resident memory, approximately 0.3%, with decode within 10%
at all four contexts. No installed version or defaults changed.

Raw manifests, binary hashes, responses, and memory samples remain untracked at
`/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/`, in
`memory-20260910-reference-v1-reference-mtp-1` and
`memory-20260910-afm-v1-afm-mtp-1`.
