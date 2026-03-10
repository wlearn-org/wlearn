#!/bin/bash
set -euo pipefail

# Build browser-ready IIFE + ESM bundles using esbuild
# Pure JS package (no WASM) -- bundles @wlearn/core and @wlearn/ensemble inline

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="${PROJECT_DIR}/dist"

# Read package name from package.json
# Browser global + bundle base name: @wlearn/automl -> automl
NAME=$(node -e "
  const p = require('${PROJECT_DIR}/package.json')
  console.log(p.name.split('/').pop())
")

# Auto-infer export names from module.exports in src/index.js
EXPORTS=$(node -e "
  const m = require('${PROJECT_DIR}/src/index.js')
  console.log(Object.keys(m).join(','))
")

echo "=== Building browser bundles ==="
echo "  Package: ${NAME}"
echo "  Files: ${NAME}.js, ${NAME}.mjs"
echo "  Exports: ${EXPORTS}"

mkdir -p "$DIST_DIR"

# Common esbuild flags
COMMON_FLAGS=(
  --bundle
  --platform=browser
  --minify
)

# IIFE bundle (browser global, for <script> tags)
npx esbuild "${PROJECT_DIR}/src/index.js" \
  "${COMMON_FLAGS[@]}" \
  --format=iife \
  --global-name="${NAME}" \
  --outfile="${DIST_DIR}/${NAME}.js"

# ESM bundle (IIFE with private global + appended named exports)
INTERNAL="__${NAME}"
npx esbuild "${PROJECT_DIR}/src/index.js" \
  "${COMMON_FLAGS[@]}" \
  --format=iife \
  --global-name="${INTERNAL}" \
  --outfile="${DIST_DIR}/${NAME}.mjs"

# Append named ESM exports
IFS=',' read -ra KEYS <<< "$EXPORTS"
DESTRUCTURE=$(IFS=','; echo "${KEYS[*]}")
EXPORT_LINE=$(IFS=','; echo "${KEYS[*]}")
echo "var {${DESTRUCTURE}}=${INTERNAL};export{${EXPORT_LINE}};" >> "${DIST_DIR}/${NAME}.mjs"

echo "=== Browser bundles built ==="
ls -lh "${DIST_DIR}/${NAME}.js" "${DIST_DIR}/${NAME}.mjs"
