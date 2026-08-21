#!/usr/bin/env python3
import json
import sys


def main() -> int:
    package = json.load(sys.stdin)
    modules = {
        target
        for product in package.get("products", [])
        if "library" in product.get("type", {})
        for target in product.get("targets", [])
    }
    if not modules:
        print("Package manifest exposes no public library modules.", file=sys.stderr)
        return 1
    for module in sorted(modules):
        print(module)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
