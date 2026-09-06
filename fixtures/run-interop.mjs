#!/usr/bin/env node
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, delimiter, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const mode = process.argv[2] || '--full'
if (!['--full', '--minimal'].includes(mode) || process.argv.length > 3) {
  throw new Error('Usage: node fixtures/run-interop.mjs [--full|--minimal]')
}
const output = process.env.WLEARN_INTEROP_OUTPUT_DIR || mkdtempSync(join(tmpdir(), 'wlearn-interop-'))
const env = {
  ...process.env,
  WLEARN_INTEROP_OUTPUT_DIR: resolve(output),
  WLEARN_INTEROP_REQUIRE_ALL: mode === '--full' ? '1' : '0',
  PYTHONPATH: [join(root, 'py'), process.env.PYTHONPATH].filter(Boolean).join(delimiter),
}
console.log(`Interop output: ${env.WLEARN_INTEROP_OUTPUT_DIR}`)
// Every stage receives the same directory; outputs remain available for review.
for (const [command, args] of [
  [process.execPath, ['fixtures/test-verify-utils.mjs']],
  [process.execPath, ['fixtures/verify.mjs']],
  [process.env.WLEARN_PYTHON || 'python3', ['-m', 'pytest', 'py/tests/test_compat.py', '-q']],
  [process.execPath, ['fixtures/verify-py-bundles.mjs']],
]) {
  const result = spawnSync(command, args, { cwd: root, env, stdio: 'inherit' })
  if (result.error) throw result.error
  if (result.status !== 0) process.exit(result.status || 1)
}
