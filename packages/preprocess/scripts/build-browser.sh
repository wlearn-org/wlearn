#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="${PROJECT_DIR}/dist"
ENTRY="${PROJECT_DIR}/scripts/browser-entry.mjs"
TRANFI_ENTRY="${TRANFI_WASM_ENTRY:-}"

if [ -z "$TRANFI_ENTRY" ] && [ -n "${TRANFI_WASM_PATH:-}" ]; then
  TRANFI_ENTRY="${TRANFI_WASM_PATH%/}/index.js"
fi

ALIAS_ARGS=()
if [ -n "$TRANFI_ENTRY" ]; then
  if [ ! -f "$TRANFI_ENTRY" ]; then
    echo "Tranfi WASM entry not found: $TRANFI_ENTRY" >&2
    exit 1
  fi
  ALIAS_ARGS+=("--alias:tranfi/wasm=$TRANFI_ENTRY")
fi

mkdir -p "$DIST_DIR"

npx esbuild "$ENTRY" \
  --bundle \
  --platform=browser \
  --external:node:fs \
  --external:node:crypto \
  --format=iife \
  --global-name=wlearnPreprocess \
  --minify \
  "${ALIAS_ARGS[@]}" \
  --outfile="${DIST_DIR}/preprocess.js"

npx esbuild "$ENTRY" \
  --bundle \
  --platform=browser \
  --external:node:fs \
  --external:node:crypto \
  --format=esm \
  --minify \
  "${ALIAS_ARGS[@]}" \
  --outfile="${DIST_DIR}/preprocess.mjs"

ls -lh "$DIST_DIR/preprocess.js" "$DIST_DIR/preprocess.mjs"
