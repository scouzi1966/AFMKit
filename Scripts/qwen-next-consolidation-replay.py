#!/usr/bin/env python3
"""Replay frozen C1 Qwen evidence on a rebuilt executable, without tuning changes.

Runs context timing and/or strict agentic replay. Imports the explicitly chosen
frozen lifecycle helper; records all inputs and preserves old results. Structure
plus identity is NOT semantic quality qualification.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import threading
import time

MIN_AVAILABLE_GIB = 100
GIB = 1024**3
TIMEOUT_SECONDS = 600
SAMPLING = {"temperature": 0.6, "top_p": 1.0, "seed": 42}
CONTEXTS = ["0.5", "1", "2", "4"]
BUSY_NAMES = {"afm", "mlx-serve", "xctest", "swift-build", "swift-frontend", "metal", "metallib", "clang", "ld"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def replacement_command(argv, binary):
    indices = [i for i, value in enumerate(argv) if Path(value).name == "afm"]
    if len(indices) != 1:
        raise ValueError("Require exactly one inference executable")
    result = list(argv)
    result[indices[0]] = str(binary)
    if "--enable-prefix-caching" in result or result[result.index("--concurrent") + 1] != "1":
        raise ValueError("This replay is C1 without prefix reuse")
    if any(v.startswith(("AFM_DEBUG=", "AFM_PERF=")) for v in result):
        raise ValueError("Instrumented launches are not performance baselines")
    return result


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifecycle-helper", type=Path, required=True)
    parser.add_argument("--agentic-baseline", type=Path, required=True)
    parser.add_argument("--context-baseline", type=Path, required=True)
    parser.add_argument("--afm", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["context", "agentic", "all"], default="all")
    args = parser.parse_args()
    q = load_module("quality_scoring", Path(__file__).with_name("qwen-next-prefill-quality.py"))
    b = load_module("frozen_lifecycle", args.lifecycle_helper)
    b.AFM, b.ROOT = args.afm.resolve(), args.output.resolve()
    assert b.sha(b.AFM) == args.expected_sha256
    baseline = read(args.agentic_baseline / "plan.json")
    assert baseline["concurrency"] == 1 and not baseline["prefix_cache"]
    assert baseline["max_tokens"] == 512 and baseline["mtp_depth"] == 3
    assert len(baseline["cases"]) == 55
    b.MODEL = Path(baseline["model"])
    context_plan = read(args.context_baseline / "plan.json")
    assert context_plan["model"] == str(b.MODEL)
    for name, key in (("config.json", "config_sha256"),
                      ("chat_template.jinja", "template_sha256"),
                      ("model.safetensors.index.json", "weight_index_sha256")):
        assert b.sha(b.MODEL / name) == context_plan[key], "Checkpoint metadata changed"
    paths = {False: args.agentic_baseline / "afm-ar-afm-mtp-0",
             True: args.agentic_baseline / "afm-batched-afm-mtp-1"}
    launches = {mtp: read(path / "launch.json") for mtp, path in paths.items()}
    commands = {mtp: replacement_command(launch["argv"], b.AFM) for mtp, launch in launches.items()}
    extra_removed = [k for k in os.environ if k.startswith(("QWEN4_", "VMLX_"))]
    for key in extra_removed:
        del os.environ[key]

    def competing(owner=None):
        return [p.info for p in psutil.process_iter(["pid", "name", "status"])
                if p.info["pid"] != owner and p.info["name"] in BUSY_NAMES
                and p.info["status"] not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)]

    assert not competing(), "Competing build/inference; will not stop it"
    b.ROOT.mkdir(parents=True, exist_ok=False)
    save(b.ROOT / "plan.json", {"phase": args.phase, "binary_sha256": args.expected_sha256,
         "runner_sha256": b.sha(Path(__file__)), "scorer_sha256": b.sha(Path(q.__file__)),
         "lifecycle_helper": str(args.lifecycle_helper), "lifecycle_sha256": b.sha(args.lifecycle_helper),
         "agentic_baseline": str(args.agentic_baseline), "context_baseline": str(args.context_baseline),
         "baseline_plan_sha256": b.sha(args.agentic_baseline / "plan.json"),
         "model": str(b.MODEL), "metadata_hashes": {name: b.sha(b.MODEL / name) for name in
             ("config.json", "chat_template.jinja", "model.safetensors.index.json")},
         "extra_removed_environment_names": extra_removed,
         "note": "Same saved M25 controls, no default/precision change. Historical timing is not a concurrent A/B. "
                 "All measured texts must match their saved same-mode controls; mismatch stops qualification."})

    def tagged_save(path, data):
        if path.name == "launch.json":
            data["launcher_sha256"] = data["binary_sha256"]
            data["binary_sha256"] = args.expected_sha256
            data["request"].update(SAMPLING)
        save(path, data)

    b.save = tagged_save
    b.command = lambda engine, mtp: commands[mtp]
    stream_chat = b.common.stream_chat

    def sampled_stream(*positional, **kwargs):
        kwargs.update(temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"])
        kwargs["extra_body"] = {**kwargs.get("extra_body", {}), "seed": SAMPLING["seed"]}
        return stream_chat(*positional, **kwargs)

    b.common.stream_chat = sampled_stream
    context_request = b.request
    b.CONTEXTS = CONTEXTS
    rows = []
    monitor_stop = threading.Event()
    unsafe = threading.Event()
    samples = []

    def checked():
        assert not unsafe.is_set(), "Resource guard"
        assert b.sha(b.AFM) == args.expected_sha256, "Binary changed during qualification"

    def context(client, out, ctx, trial, warmup, latency):
        checked()
        result = context_request(client, out, ctx, trial, warmup, latency)
        assert result["usage"]["completion_tokens"] == 128
        if not warmup:
            previous = read(args.context_baseline / f"m25-afm-mtp-{int(active_mtp)}" / f"trial-{trial}-{ctx}k.json")["result"]
            assert result["prompt_sha256"] == previous["prompt_sha256"]
            assert result["generated_text"] == previous["generated_text"], "Context text differs after consolidation"
            rows.append({"context": ctx, "trial": trial, "mtp": active_mtp,
                         "prefill_proxy": result["client_prefill_tps"], "decode": result["client_decode_tps"],
                         "previous_prefill_proxy": previous["client_prefill_tps"],
                         "previous_decode": previous["client_decode_tps"]})

    def agentic(client, out, *unused):
        records = []
        for index in [-1, *range(len(baseline["cases"]))]:
            checked()
            case = baseline["cases"][max(index, 0)]
            fixture = baseline["fixtures"][case["task"]]
            payload = {"model": str(b.MODEL), "messages": fixture["messages"],
                       "temperature": case["temperature"], "top_p": 1.0, "seed": case["seed"],
                       "max_tokens": 512, "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}
            label = "warmup" if index < 0 else f"case-{index:02d}"
            text, reasoning, chunks, usage = "", "", [], {}
            start, first, last = time.monotonic(), None, None
            record = {"case": case, "payload": payload}
            try:
                params = dict(payload)
                extra = params.pop("chat_template_kwargs")
                with client.with_options(timeout=TIMEOUT_SECONDS).chat.completions.create(
                        **params, extra_body={"chat_template_kwargs": extra}) as response:
                    for chunk in response:
                        assert not unsafe.is_set(), "Resource guard"
                        now = time.monotonic()
                        data = chunk.model_dump(mode="json")
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
                finishes = [c["finish_reason"] for item in chunks for c in item["chunk"].get("choices", []) if c.get("finish_reason")]
                tokens = usage.get("completion_tokens", 0)
                record.update(text=text, reasoning=reasoning, usage=usage, chunks=chunks,
                              seconds=time.monotonic() - start, ttft=first - start if first else None,
                              decode_tps=(tokens - 1) / (last - first) if first and last > first else 0,
                              finish_reasons=finishes, runtime_ok=bool(text and tokens and finishes and not reasoning))
                record.update(q.score(text, fixture))
                assert record["runtime_ok"] and not reasoning
                assert (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) in (0, None)
                if index >= 0:
                    previous = read(paths[active_mtp] / f"{label}.json")
                    record["matches_previous_text"] = text == previous["text"]
                    assert payload == previous["payload"]
                    assert record["matches_previous_text"], "Agentic text differs after consolidation"
                    records.append(record)
            except Exception as error:
                record.update(error=repr(error), text=text, chunks=chunks, usage=usage)
                raise
            finally:
                save(out / f"{label}.json", record)
            print(out.name, label, "strict=", record["strict_ok"], "wall=", round(record["seconds"], 2), flush=True)
        save(out / "quality-summary.json", q.summarize(records))
        print("SUMMARY", out.name, json.dumps(q.summarize(records)), flush=True)

    def monitor():
        while not monitor_stop.wait(1):
            process_file = b.ROOT / f"{active_phase}-afm-mtp-{int(active_mtp)}" / "process.json"
            # The lifecycle helper publishes the PID just after spawn. Do not
            # mistake our own startup process for competing work in that gap.
            try:
                owner = read(process_file)["pid"]
            except (FileNotFoundError, json.JSONDecodeError):
                owner = None
            available = psutil.virtual_memory().available
            busy = competing(owner) if owner is not None else []
            samples.append({"time": time.time(), "available": available, "competing": busy})
            if available < MIN_AVAILABLE_GIB * GIB or busy:
                unsafe.set()

    for active_phase in (["context", "agentic"] if args.phase == "all" else [args.phase]):
        for active_mtp in (False, True):
            assert not competing()
            checked()
            monitor_stop.clear()
            watcher = threading.Thread(target=monitor, daemon=True)
            watcher.start()
            try:
                b.request = context if active_phase == "context" else agentic
                b.run_arm("afm", active_mtp, active_phase, active_phase == "agentic")
            finally:
                monitor_stop.set()
                watcher.join()
                save(b.ROOT / f"{active_phase}-{int(active_mtp)}-guard.json", {"unsafe": unsafe.is_set(), "samples": samples})
                samples = []
            assert read(b.ROOT / f"{active_phase}-afm-mtp-{int(active_mtp)}" / "exit.json")["exit_code"] == 0
            assert not unsafe.is_set()
    save(b.ROOT / "context-comparison.json", rows)
    save(b.ROOT / "all-complete.json", {"status": "completed", "phase": args.phase})


if __name__ == "__main__":
    main()
