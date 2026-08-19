#!/bin/zsh
set -euo pipefail

script_directory="${0:A:h}"
app_root="${script_directory:h}"

swift build --package-path "$app_root" -c release
bin_path="$(swift build --package-path "$app_root" -c release --show-bin-path)"
bundle_path="$app_root/.build/DecisionBrief.app"
contents_path="$bundle_path/Contents"

rm -rf "$bundle_path"
mkdir -p "$contents_path/MacOS" "$contents_path/Resources"
cp "$app_root/Resources/Info.plist" "$contents_path/Info.plist"
cp "$bin_path/DecisionBrief" "$contents_path/MacOS/DecisionBrief"
codesign --force --sign - "$bundle_path"

printf '%s\n' "$bundle_path"
