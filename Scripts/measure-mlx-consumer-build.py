#!/usr/bin/env python3
"""Summarize one clean AFMKitMLX downstream SwiftPM build."""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys


def kibibytes(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    result = subprocess.run(
        ["/usr/bin/du", "-sk", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.split()[0])


def main() -> int:
    if len(sys.argv) != 6:
        print(
            f"Usage: {sys.argv[0]} scratch-path build-log elapsed-seconds source-sha jobs",
            file=sys.stderr,
        )
        return 64

    scratch = pathlib.Path(sys.argv[1])
    log_path = pathlib.Path(sys.argv[2])
    elapsed = float(sys.argv[3])
    source_sha = sys.argv[4]
    jobs = int(sys.argv[5])
    descriptions = list(scratch.glob("**/release/description.json"))
    selected = None
    for path in descriptions:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        if "AFMKitMLXConsumer" in candidate.get("targetDependencyMap", {}):
            selected = candidate
            break
    if selected is None:
        raise SystemExit("The consumer build produced no usable release description.json.")

    closure = selected["targetDependencyMap"]["AFMKitMLXConsumer"]
    if "AFMKitMLXReleaseGraph" in closure:
        raise SystemExit("The downstream consumer still reaches AFMKitMLXReleaseGraph.")

    action_totals = [
        int(match.group(1))
        for match in re.finditer(r"\[\d+/(\d+)\]", log_path.read_text(encoding="utf-8"))
    ]
    report = {
        "sourceSHA": source_sha,
        "buildJobs": jobs,
        "elapsedSeconds": round(elapsed, 2),
        "maximumReportedBuildActions": max(action_totals, default=0),
        "consumerDependencyTargetCount": len(closure),
        "describedTargetCount": len(selected["targetDependencyMap"]),
        "swiftCommandCount": len(selected.get("swiftCommands", {})),
        "writeCommandCount": len(selected.get("writeCommands", {})),
        "copyCommandCount": len(selected.get("copyCommands", {})),
        "scratchSizeKiB": kibibytes(scratch),
        "checkoutSizeKiB": kibibytes(scratch / "checkouts"),
        "releaseSizeKiB": kibibytes(scratch / "arm64-apple-macosx" / "release"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
