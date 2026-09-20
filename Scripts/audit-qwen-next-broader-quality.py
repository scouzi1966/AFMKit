#!/usr/bin/env python3
"""Read-only evidence audit; writes a new report, never changes inference data."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("broader", Path(__file__).with_name("qwen-next-broader-quality.py"))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit(root, completed_arms_only=False):
    plan = q.read(root / "plan.json")
    all_complete = (root / "all-complete.json").exists()
    if all_complete:
        assert q.read(root / "all-complete.json")["status"] == "completed"
    else:
        assert completed_arms_only, "Incomplete run: use explicit --completed-arms-only for a partial audit"
    expected_cases = plan["cases"]
    assert expected_cases == q.cases(), "Fixture changed after inference"
    assert sha(Path(q.__file__)) == plan["runner_sha256"], "Runner changed after inference"
    variants = {v["name"]: v for v in plan["variants"]}
    rows_by_arm, details, payloads, skipped = {}, [], {}, []
    phases = ["first", "repeat"] if plan["repeat"] else ["first"]
    for name, mtp in plan["arms"]:
        arm = f"{name}-{plan.get('engine', 'afm')}-mtp-{int(mtp)}"
        folder = root / arm
        if not (folder / "complete.json").exists() and completed_arms_only:
            skipped.append(arm)
            continue
        variant = variants[name]
        assert sha(Path(variant["binary"])) == variant["sha256"]
        launch = q.read(folder / "launch.json")
        assert launch["binary_sha256"] == variant["sha256"]
        assert q.read(folder / "exit.json")["exit_code"] == 0
        assert q.read(folder / "complete.json")["status"] == "completed"
        guard = q.read(folder / "guard.json")
        assert not guard["unsafe"] and guard["samples"]
        assert all(not s["competing"] and s["available"] >= q.MIN_AVAILABLE_GIB * q.GIB for s in guard["samples"])
        rows = {}
        for phase in phases:
            for case in expected_cases:
                key = f"{phase}-{case['request_id']}"
                row = q.read(folder / f"{key}.json")
                assert row["case"] == case
                assert row["payload"]["messages"] == case["messages"]
                assert row["payload"]["temperature"] == case["temperature"]
                assert row["payload"]["seed"] == case["seed"]
                assert row["payload"]["top_p"] == plan["top_p"]
                assert row["payload"]["max_tokens"] == plan["max_tokens"]
                assert row["runtime_ok"] and not row.get("error") and not row["reasoning"]
                rescored = q.score(row["text"], case)
                assert all(row[k] == v for k, v in rescored.items())
                if key in payloads:
                    assert payloads[key] == row["payload"], "Paired payload mismatch"
                payloads[key] = row["payload"]
                rows[key] = row
            for kind in ("greedy", "sampled"):
                group = [r for key, r in rows.items() if key.startswith(phase + "-") and r["case"]["kind"] == kind]
                summary = q.read(folder / f"{phase}-{kind}-summary.json")
                assert summary == q.summarize(group, summary["wall_seconds"])
                if phase == "repeat" and plan.get("require_cache_hits"):
                    assert summary["cached_tokens"] > 0, "Required cache coverage missing"
                details.append({"arm": arm, "phase": phase, "kind": kind, **summary,
                                "failures": [{"request_id": r["case"]["request_id"], "task": r["case"]["task"],
                                              "expected": r["case"]["answer"], "actual_text": r["text"]}
                                             for r in group if not r["semantic_ok"]]})
        if (folder / "window-01.json").exists():
            groups = q.request_groups(expected_cases, plan["repeat"], plan.get("replay_window", 0))
            window_paths = sorted(folder.glob("window-*.json"))
            assert len(window_paths) == len(groups)
            walls = {}
            for path, (phase, kind, cases) in zip(window_paths, groups):
                window = q.read(path)
                assert (window["phase"], window["kind"]) == (phase, kind)
                assert window["request_ids"] == [c["request_id"] for c in cases]
                group = [rows[f"{phase}-{c['request_id']}"] for c in cases]
                elapsed = window["summary"]["wall_seconds"]
                assert window["summary"] == q.summarize(group, elapsed)
                walls[(phase, kind)] = walls.get((phase, kind), 0) + elapsed
                if phase == "repeat" and plan.get("require_cache_hits"):
                    assert window["summary"]["cached_tokens"] > 0
            for (phase, kind), elapsed in walls.items():
                assert q.read(folder / f"{phase}-{kind}-summary.json")["wall_seconds"] == elapsed
        rows_by_arm[arm] = rows
        details.append({"arm": arm, "guard": {"peak_rss_gib": max(s["rss"] or 0 for s in guard["samples"]) / q.GIB,
                                               "min_available_gib": min(s["available"] for s in guard["samples"]) / q.GIB}})
    comparisons = []
    if len(variants) == 2:
        control, candidate = [v["name"] for v in plan["variants"]]
        for mtp in (False, True):
            left, right = f"{control}-afm-mtp-{int(mtp)}", f"{candidate}-afm-mtp-{int(mtp)}"
            if left not in rows_by_arm or right not in rows_by_arm:
                continue
            for phase in phases:
                for kind in ("greedy", "sampled"):
                    keys = [k for k, r in rows_by_arm[left].items() if k.startswith(phase + "-") and r["case"]["kind"] == kind]
                    pairs = [(k, rows_by_arm[left][k], rows_by_arm[right][k]) for k in keys]
                    comparisons.append({"mtp": mtp, "phase": phase, "kind": kind,
                        "identical_texts": sum(a["text"] == b["text"] for k, a, b in pairs),
                        "improved": [k for k, a, b in pairs if not a["semantic_ok"] and b["semantic_ok"]],
                        "regressed": [k for k, a, b in pairs if a["semantic_ok"] and not b["semantic_ok"]]})
    assert rows_by_arm, "No completed arms to audit"
    return {"root": str(root), "status": "audited" if all_complete else "partial-completed-arms-only",
            "skipped_incomplete_arms": skipped, "details": details, "comparisons": comparisons,
            "hashes": {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*.json"))}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path, help="Preserved exact runner revision used for this evidence")
    parser.add_argument("--completed-arms-only", action="store_true",
                        help="Explicit partial audit; never labels an incomplete whole run as passed")
    args = parser.parse_args()
    if args.runner:
        spec = importlib.util.spec_from_file_location("frozen_broader", args.runner)
        q = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(q)
    result = audit(args.root, args.completed_arms_only)
    q.save(args.output, result)
    print(json.dumps({"status": result["status"], "skipped_incomplete_arms": result["skipped_incomplete_arms"],
                      "comparisons": result["comparisons"]}, indent=2))
