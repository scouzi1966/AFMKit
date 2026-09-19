#!/usr/bin/env python3
"""Replay a frozen agentic screen with explicit prefill sizes, without runtime edits.

This measures structure/identity and wall throughput, not semantic quality. It
imports the frozen server-lifecycle helper selected by the operator. Evidence
is append-only, and an unchanged Release executable is required throughout.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import threading
import time

MIN_AVAILABLE_GIB = 100
REQUEST_TIMEOUT_SECONDS = 600
GIB = 1024**3
REQUIRED_KEYS = {"request_id", "file", "diagnosis", "fix", "test"}
BUSY_NAMES = {"afm", "mlx-serve", "xctest", "swift-build", "swift-frontend",
              "metal", "metallib", "clang", "ld"}


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key: {key}")
        result[key] = value
    return result


def score(text, fixture):
    # Keep the old identity metric separately for exact baseline comparability.
    file_match = re.search(r'"file"\s*:\s*"([^"]*)"', text)
    id_match = re.search(r'"request_id"\s*:\s*"([^"]*)"', text)
    identity = bool(file_match and id_match and file_match[1] == fixture["file"]
                    and id_match[1] == fixture["request_id"])
    try:
        obj = json.loads(text, object_pairs_hook=unique_object)
        strict = (isinstance(obj, dict) and set(obj) == REQUIRED_KEYS
                  and all(isinstance(v, str) and v for v in obj.values())
                  and obj["file"] == fixture["file"]
                  and obj["request_id"] == fixture["request_id"])
    except (ValueError, TypeError):
        strict = False
    return {"identity_ok": identity, "strict_ok": bool(strict),
            "actual_file": file_match[1] if file_match else None}


def summarize(rows):
    result = {}
    for kind in ("greedy", "sampled"):
        group = [r for r in rows if r["case"]["kind"] == kind]
        seconds = sum(r["seconds"] for r in group)
        result[kind] = {
            "total": len(group), "runtime": sum(r["runtime_ok"] for r in group),
            "identity": sum(r["identity_ok"] for r in group),
            "strict": sum(r["strict_ok"] for r in group),
            "output_tokens_per_wall_second_including_prefill":
                sum(r["usage"]["completion_tokens"] for r in group) / seconds,
            "strict_tasks_per_second": sum(r["strict_ok"] for r in group) / seconds,
            "median_ttft_seconds": statistics.median(r["ttft"] for r in group),
            "median_decode_tps": statistics.median(r["decode_tps"] for r in group),
            "wall_seconds": seconds,
            "per_task": {str(task): sum(r["strict_ok"] for r in group
                                       if r["case"]["task"] == task)
                         for task in sorted({r["case"]["task"] for r in group})},
        }
    return result


def launch_argv(frozen, binary, step):
    argv = list(frozen)
    binary_indices = [i for i, value in enumerate(argv) if Path(value).name == "afm"]
    if len(binary_indices) != 1 or "--prefill-step-size" in argv:
        raise ValueError("Require one AFM binary and an unmodified frozen chunk policy")
    argv[binary_indices[0]] = str(binary)
    if "--enable-prefix-caching" in argv or argv[argv.index("--concurrent") + 1] != "1":
        raise ValueError("This screen is C1 with prefix caching off")
    return argv + ["--prefill-step-size", str(step)]


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Frozen independent-full directory with plan and both AFM arms")
    parser.add_argument("--lifecycle-helper", type=Path, required=True)
    parser.add_argument("--afm", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("frozen_lifecycle", args.lifecycle_helper)
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    b.AFM, b.ROOT = args.afm.resolve(), args.output.resolve()
    assert b.sha(b.AFM) == args.expected_sha256, "Release binary changed"
    frozen = read(args.baseline / "plan.json")
    assert frozen["afm_binary_sha256"] == args.expected_sha256
    assert frozen["concurrency"] == 1 and not frozen["prefix_cache"]
    assert frozen["max_tokens"] == 512 and frozen["mtp_depth"] == 3
    fixtures, cases = frozen["fixtures"], frozen["cases"]
    assert len(cases) == 55
    b.MODEL = Path(frozen["model"])
    arms = [(4096, False), (8192, False), (8192, True), (4096, True)]
    sources = {False: args.baseline / "afm-ar-afm-mtp-0",
               True: args.baseline / "afm-batched-afm-mtp-1"}
    launches = {mtp: read(folder / "launch.json") for mtp, folder in sources.items()}
    for launch in launches.values():
        assert not any(arg.startswith(("AFM_DEBUG=", "AFM_PERF=")) for arg in launch["argv"])
    # The frozen lifecycle helper removes AFM/MLX environment controls itself.
    # Remove the two additional research namespaces it predates, recording names only.
    extra_removed = [key for key in os.environ if key.startswith(("QWEN4_", "VMLX_"))]
    for key in extra_removed:
        del os.environ[key]

    def competing(owner=None):
        return [p.info for p in psutil.process_iter(["pid", "name", "status"])
                if p.info["pid"] != owner and p.info["name"] in BUSY_NAMES
                and p.info["status"] not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)]

    assert not competing(), "Competing inference/build"
    b.ROOT.mkdir(parents=True, exist_ok=False)
    save(b.ROOT / "plan.json", {
        "baseline": str(args.baseline.resolve()), "baseline_plan_sha256": b.sha(args.baseline / "plan.json"),
        "runner_sha256": b.sha(Path(__file__)), "lifecycle_helper": str(args.lifecycle_helper),
        "lifecycle_sha256": b.sha(args.lifecycle_helper), "afm_sha256": args.expected_sha256,
        "model": str(b.MODEL), "cases": cases, "fixtures": fixtures, "arms": arms,
        "extra_removed_environment_names": extra_removed,
        "metadata_hashes": {name: b.sha(b.MODEL / name) for name in
            ("config.json", "tokenizer_config.json", "model.safetensors.index.json")
            if (b.MODEL / name).is_file()},
        "note": "Same saved 50 seeds plus five greedy controls per arm; one excluded warmup. "
                "M25 opt-ins retained. Not new independent tasks, semantic judging, or a concurrency test. "
                "No default, precision, cache or runtime source changes."})

    def tagged_save(path, value):
        if path.name == "launch.json":
            value["launcher_sha256"] = value["binary_sha256"]
            value["binary_sha256"] = args.expected_sha256
            value["request"] = launches[active[1]]["request"]
            value["explicit_prefill_step"] = active[0]
        save(path, value)

    b.save = tagged_save
    b.command = lambda engine, mtp: launch_argv(launches[mtp]["argv"], b.AFM, active[0])

    def workload(client, out, *unused):
        owner = read(out / "process.json")["pid"]
        proc = psutil.Process(owner)
        stop, unsafe = threading.Event(), threading.Event()
        samples, collisions = [], []

        def watch():
            while not stop.is_set():
                try:
                    available = psutil.virtual_memory().available
                    samples.append({"time": time.time(), "rss": proc.memory_info().rss,
                                    "available": available})
                    busy = competing(owner)
                    if busy:
                        collisions.append({"time": time.time(), "processes": busy})
                    if busy or available < MIN_AVAILABLE_GIB * GIB:
                        unsafe.set()
                except psutil.NoSuchProcess:
                    unsafe.set()
                    break
                stop.wait(1)

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()

        def one(case, label):
            fixture = fixtures[case["task"]]
            payload = {"model": str(b.MODEL), "messages": fixture["messages"],
                       "temperature": case["temperature"], "top_p": 1.0,
                       "seed": case["seed"], "max_tokens": 512, "stream": True,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}
            row = {"case": case, "payload": payload, "expected": fixture["file"]}
            start, first, last = time.monotonic(), None, None
            chunks, text, reasoning, usage = [], "", "", {}
            try:
                assert not unsafe.is_set(), "Resource guard"
                params = dict(payload)
                extra = params.pop("chat_template_kwargs")
                with client.with_options(timeout=REQUEST_TIMEOUT_SECONDS).chat.completions.create(
                        **params, extra_body={"chat_template_kwargs": extra}) as response:
                    for chunk in response:
                        assert not unsafe.is_set(), "Resource guard"
                        data = chunk.model_dump(mode="json")
                        now = time.monotonic()
                        chunks.append({"elapsed_s": now - start, "chunk": data})
                        if chunk.usage:
                            usage = chunk.usage.model_dump()
                        if chunk.choices:
                            delta = data["choices"][0]["delta"]
                            part = delta.get("content") or ""
                            reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
                            if part:
                                first = now if first is None else first
                                last = now
                            text += part
                finishes = [c["finish_reason"] for item in chunks
                            for c in item["chunk"].get("choices", []) if c.get("finish_reason")]
                tokens = usage.get("completion_tokens", 0)
                row.update(text=text, reasoning=reasoning, usage=usage, chunks=chunks,
                           seconds=time.monotonic() - start, ttft=first - start if first else None,
                           decode_tps=(tokens - 1) / (last - first) if first and last > first else 0,
                           finish_reasons=finishes,
                           runtime_ok=bool(text and tokens and finishes and not reasoning))
                row.update(score(text, fixture))
                assert row["runtime_ok"], "Runtime response invalid"
                assert (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) in (0, None)
                if label != "warmup":
                    previous = read(sources[active[1]] / f"{label}.json")
                    assert payload == previous["payload"], "Request payload changed"
                    row["text_matches_frozen_4096"] = text == previous["text"]
                    row["previous_score"] = score(previous["text"], fixture)
                    if active[0] == 4096:
                        assert row["text_matches_frozen_4096"], "Control no longer reproduces baseline"
            except Exception as error:
                row.update(error=repr(error), text=text, usage=usage, chunks=chunks,
                           runtime_ok=False, seconds=time.monotonic() - start)
                raise
            finally:
                save(out / f"{label}.json", row)
            print(out.name, label, "strict=", row["strict_ok"], "file=", row["actual_file"],
                  "wall=", round(row["seconds"], 2), flush=True)
            return row

        try:
            one(cases[0], "warmup")
            rows = [one(case, f"case-{i:02d}") for i, case in enumerate(cases)]
            summary = summarize(rows)
            save(out / "quality-summary.json", summary)
            print("ARM SUMMARY", out.name, json.dumps(summary), flush=True)
        finally:
            stop.set()
            watcher.join()
            save(out / "resource-isolation.json", {"samples": samples, "collisions": collisions,
                 "unsafe": unsafe.is_set(), "note": "1 Hz RSS/available guard; not isolated GPU memory accounting"})
        assert not unsafe.is_set()

    b.request = workload
    for active in arms:
        assert not competing()
        assert b.sha(b.AFM) == args.expected_sha256
        b.run_arm("afm", active[1], f"prefill-{active[0]}", True)
        folder = b.ROOT / f"prefill-{active[0]}-afm-mtp-{int(active[1])}"
        assert read(folder / "exit.json")["exit_code"] == 0
    save(b.ROOT / "all-complete.json", {"status": "completed", "arms": arms})


if __name__ == "__main__":
    main()
