#!/usr/bin/env python3
import json
import pathlib
import re
import sys
from urllib.parse import urlparse


def swift_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def main() -> int:
    if len(sys.argv) != 5:
        print(
            f"Usage: {sys.argv[0]} output-directory dependency-url version package-identity",
            file=sys.stderr,
        )
        return 64

    output_directory = pathlib.Path(sys.argv[1])
    dependency_url = sys.argv[2]
    version = sys.argv[3]
    package_identity = sys.argv[4]
    if urlparse(dependency_url).scheme not in {"file", "https"}:
        print("Downstream qualification requires a file or HTTPS Git remote.", file=sys.stderr)
        return 1

    package = json.load(sys.stdin)
    products = []
    for product in package.get("products", []):
        if "library" not in product.get("type", {}):
            continue
        name = product.get("name", "")
        targets = product.get("targets", [])
        if not name or not targets:
            print("A public library product has no name or targets.", file=sys.stderr)
            return 1
        products.append((name, targets))

    if not products:
        print("Package manifest exposes no public library products.", file=sys.stderr)
        return 1

    products.sort(key=lambda item: item[0])
    target_entries = []
    product_entries = []
    validator_products = []

    for index, (product_name, module_names) in enumerate(products, start=1):
        suffix = re.sub(r"[^A-Za-z0-9_]", "_", product_name)
        target_name = f"Validate{index}_{suffix}"
        validator_product = f"{product_name}Consumer"
        validator_products.append(validator_product)
        product_entries.append(
            "        .executable(name: "
            f"{swift_string(validator_product)}, targets: [{swift_string(target_name)}])"
        )
        target_entries.append(
            "        .executableTarget(\n"
            f"            name: {swift_string(target_name)},\n"
            "            dependencies: [\n"
            "                .product(name: "
            f"{swift_string(product_name)}, package: {swift_string(package_identity)})\n"
            "            ]\n"
            "        )"
        )

        source_directory = output_directory / "Sources" / target_name
        source_directory.mkdir(parents=True, exist_ok=True)
        imports = "\n".join(f"import {module_name}" for module_name in module_names)
        (source_directory / "main.swift").write_text(
            f"{imports}\n\nprint({swift_string(product_name + ' imported')})\n",
            encoding="utf-8",
        )

    rendered_products = ",\n".join(product_entries)
    rendered_targets = ",\n".join(target_entries)
    manifest = f"""// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "AFMKitDownstreamQualification",
    platforms: [
        .macOS("26.0")
    ],
    products: [
{rendered_products}
    ],
    dependencies: [
        .package(url: {swift_string(dependency_url)}, exact: {swift_string(version)})
    ],
    targets: [
{rendered_targets}
    ]
)
"""
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "Package.swift").write_text(manifest, encoding="utf-8")
    (output_directory / "validator-products.txt").write_text(
        "\n".join(sorted(validator_products)) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
