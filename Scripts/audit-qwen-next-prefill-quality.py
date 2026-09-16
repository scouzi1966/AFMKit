#!/usr/bin/env python3
"""Read-only audit of a completed prefill quality screen; write a new report folder."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

spec = importlib.util.spec_from_file_location("prefill_quality", Path(__file__).with_name("qwen-next-prefill-quality.py"))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def percent(new, old):
    return 100 * (new / old - 1) if old else None


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def paired(old, new):
    assert len(old) == len(new)
    assert all(a["case"] == b["case"] and a["payload"] == b["payload"] for a, b in zip(old, new))
    pairs = [(a, b) for a, b in zip(old, new) if a["case"]["kind"] == "sampled"]
    return {
        "sampled_pairs": len(pairs),
        "unchanged_text": sum(a["text"] == b["text"] for a, b in pairs),
        "strict_gained": [b["case"] for a, b in pairs if not a["strict_ok"] and b["strict_ok"]],
        "strict_lost": [b["case"] for a, b in pairs if a["strict_ok"] and not b["strict_ok"]],
        "both_strict": sum(a["strict_ok"] and b["strict_ok"] for a, b in pairs),
        "neither_strict": sum(not a["strict_ok"] and not b["strict_ok"] for a, b in pairs),
    }


def audit(root):
    plan = q.read(root / "plan.json")
    assert sha(Path(q.__file__)) == plan["runner_sha256"], "Runner changed since launch"
    assert sha(Path(plan["lifecycle_helper"])) == plan["lifecycle_sha256"]
    assert q.read(root / "all-complete.json")["status"] == "completed"
    assert len(plan["cases"]) == 55
    arms, records, inventory = {}, {}, []
    for step, mtp in plan["arms"]:
        name = f"prefill-{step}-afm-mtp-{int(mtp)}"
        folder = root / name
        launch = q.read(folder / "launch.json")
        assert launch["binary_sha256"] == plan["afm_sha256"]
        assert launch["explicit_prefill_step"] == step
        assert q.read(folder / "exit.json")["exit_code"] == 0
        assert q.read(folder / "complete.json")["status"] == "completed"
        guard = q.read(folder / "resource-isolation.json")
        assert not guard["unsafe"] and not guard["collisions"]
        assert guard["samples"]
        rows = []
        for i, case in enumerate(plan["cases"]):
            path = folder / f"case-{i:02d}.json"
            row = q.read(path)
            assert row["case"] == case and row["runtime_ok"] and not row.get("error")
            assert not row["reasoning"]
            assert row["payload"]["max_tokens"] == 512
            assert row["payload"]["top_p"] == 1.0
            assert row["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
            assert (row["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0) in (0, None)
            assert row["usage"]["prompt_tokens"] > 4096
            assert all(math.isfinite(row[key]) and row[key] > 0 for key in ("seconds", "ttft", "decode_tps"))
            fixture = plan["fixtures"][case["task"]]
            score = q.score(row["text"], fixture)
            assert all(row[key] == value for key, value in score.items())
            if step == 4096:
                assert row["text_matches_frozen_4096"]
            rows.append(row)
            inventory.append({"path": str(path.relative_to(root)),
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        assert q.read(folder / "quality-summary.json") == q.summarize(rows)
        arms[name] = {**q.summarize(rows),
            "peak_process_rss_gib": max(s["rss"] for s in guard["samples"]) / q.GIB,
            "minimum_available_gib": min(s["available"] for s in guard["samples"]) / q.GIB,
            "sampled_length_finishes": sum("length" in r["finish_reasons"] for r in rows if r["case"]["kind"] == "sampled"),
            "failures": [{"case": r["case"], "expected": r["expected"], "actual_file": r["actual_file"],
                          "identity_ok": r["identity_ok"]} for r in rows if not r["strict_ok"]]}
        records[name] = rows
    comparisons, historical = {}, {}
    metrics = ("output_tokens_per_wall_second_including_prefill", "strict_tasks_per_second",
               "median_ttft_seconds", "median_decode_tps")
    for mtp in (0, 1):
        old, new = f"prefill-4096-afm-mtp-{mtp}", f"prefill-8192-afm-mtp-{mtp}"
        comparisons[str(mtp)] = {**paired(records[old], records[new]),
            "change_percent": {metric: percent(arms[new]["sampled"][metric], arms[old]["sampled"][metric])
                               for metric in metrics},
            "rss_change_gib": arms[new]["peak_process_rss_gib"] - arms[old]["peak_process_rss_gib"]}
        previous_folder = Path(plan["baseline"]) / ("afm-batched-afm-mtp-1" if mtp else "afm-ar-afm-mtp-0")
        previous = [q.read(previous_folder / f"case-{i:02d}.json") for i in range(55)]
        assert all(a["text"] == b["text"] and a["payload"] == b["payload"]
                   for a, b in zip(previous, records[old]))
        sampled = [r for r in previous if r["case"]["kind"] == "sampled"]
        previous_rate = sum(r["usage"]["completion_tokens"] for r in sampled) / sum(r["seconds"] for r in sampled)
        current_rate = arms[old]["sampled"]["output_tokens_per_wall_second_including_prefill"]
        historical[str(mtp)] = {"source": str(previous_folder), "identical_answers": 55,
            "previous_output_tokens_per_wall_second_including_prefill": previous_rate,
            "current_output_tokens_per_wall_second_including_prefill": current_rate,
            "change_percent": percent(current_rate, previous_rate),
            "note": "Same binary/payloads/answers, different run time; not a cause attribution or peak replacement."}
    return {"status": "completed", "measured_responses": len(inventory), "warmups": len(arms),
            "arms": arms, "comparisons_8192_vs_4096": comparisons, "response_inventory": inventory,
            "frozen_4096_timing_comparison": historical,
            "note": "C1, prefix off, saved five task families and seeds. Strict means unique-key JSON plus identity, "
                    "not semantic correctness. Median client decode is not fixed-token kernel timing. "
                    "RSS samples start after model readiness, not isolated GPU memory or a load-time peak. "
                    "Same 4096 answers must match frozen baseline."}


def markdown(result):
    lines = ["# Paired prefill quality/performance screen", "", result["note"], "",
             "| Chunk | MTP | Runtime | Greedy strict | Sampled identity | Sampled strict | Output tok/s incl. prefill | Strict tasks/s | Median TTFT (s) | Median decode tok/s | Peak RSS GiB |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mtp in (0, 1):
        for step in (4096, 8192):
            arm = result["arms"][f"prefill-{step}-afm-mtp-{mtp}"]
            s, g = arm["sampled"], arm["greedy"]
            lines.append(f"| {step} | {'on, d3' if mtp else 'off'} | {s['runtime']+g['runtime']}/55 | "
                         f"{g['strict']}/5 | {s['identity']}/50 | {s['strict']}/50 | "
                         f"{s['output_tokens_per_wall_second_including_prefill']:.2f} | "
                         f"{s['strict_tasks_per_second']:.4f} | {s['median_ttft_seconds']:.3f} | "
                         f"{s['median_decode_tps']:.2f} | {arm['peak_process_rss_gib']:.2f} |")
    lines += ["", "## Paired changes", ""]
    for mtp, change in result["comparisons_8192_vs_4096"].items():
        lines.append(f"- MTP {'on' if mtp == '1' else 'off'}: {len(change['strict_gained'])} strict passes gained, "
                     f"{len(change['strict_lost'])} lost; {change['unchanged_text']}/50 sampled texts unchanged.")
    lines += ["", "## Historical timing regression tracking", ""]
    for mtp, comparison in result["frozen_4096_timing_comparison"].items():
        lines.append(f"- MTP {'on' if mtp == '1' else 'off'}, same 4096 answers: "
                     f"{comparison['previous_output_tokens_per_wall_second_including_prefill']:.2f} → "
                     f"{comparison['current_output_tokens_per_wall_second_including_prefill']:.2f} output tok/s "
                     f"including prefill ({comparison['change_percent']:+.2f}%). "
                     "Different run time; no cause attribution or replacement of the saved peak.")
    lines += ["", "These are exploratory paired results, not a noninferiority certificate or semantic-judge scores.",
              "No new default, automatic selection, cache policy or precision change was adopted.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_root)
    args.output.mkdir(parents=True, exist_ok=False)
    q.save(args.output / "audit.json", result)
    with (args.output / "README.md").open("x") as handle:
        handle.write(markdown(result))
    with (args.output / "INPUT-SHA256SUMS.txt").open("x") as handle:
        for path in sorted(args.run_root.rglob("*")):
            if path.is_file():
                handle.write(f"{sha(path)}  {os.path.relpath(path, args.output)}\n")
    print(markdown(result))


if __name__ == "__main__":
    main()
