#!/usr/bin/env python3
import json
import sys


VOLATILE_KEYS = {"generator", "location", "uri", "range"}


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def symbol_key(symbol: object) -> tuple[str, str]:
    if not isinstance(symbol, dict):
        return ("", stable_json(symbol))
    identifier = symbol.get("identifier", {})
    precise = identifier.get("precise", "") if isinstance(identifier, dict) else ""
    return (precise, stable_json(symbol))


def relationship_key(relationship: object) -> tuple[str, str, str, str]:
    if not isinstance(relationship, dict):
        return ("", "", "", stable_json(relationship))
    return (
        str(relationship.get("source", "")),
        str(relationship.get("kind", "")),
        str(relationship.get("target", "")),
        stable_json(relationship),
    )


def normalize(value: object, path: tuple[str, ...] = ()) -> object:
    if isinstance(value, dict):
        return {
            key: normalize(value[key], path + (key,))
            for key in sorted(value)
            if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        normalized = [normalize(item, path) for item in value]
        if path == ("symbols",):
            return sorted(normalized, key=symbol_key)
        if path == ("relationships",):
            return sorted(normalized, key=relationship_key)
        return normalized
    return value


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} raw-symbols.json normalized-symbols.json", file=sys.stderr)
        return 64

    raw_path, normalized_path = sys.argv[1:3]
    with open(raw_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    with open(normalized_path, "w", encoding="utf-8") as handle:
        json.dump(normalize(raw), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
