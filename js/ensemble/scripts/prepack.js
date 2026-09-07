'use strict'

const fs = require('fs')
const path = require('path')
const { spawnSync } = require('child_process')

const pkgDir = path.resolve(__dirname, '..')
const pkg = JSON.parse(fs.readFileSync(path.join(pkgDir, 'package.json'), 'utf8'))
const baseName = pkg.name.split('/').pop()
const scripts = pkg.scripts || {}
const files = pkg.files || []
const skipBuild = /^(1|true|yes)$/i.test(process.env.WLEARN_SKIP_BUILD || '')

function run(cmd, args) {
  const result = spawnSync(cmd, args, {
    cwd: pkgDir,
    stdio: 'inherit',
    env: process.env
  })
  if (result.status !== 0) {
    process.exit(result.status || 1)
  }
}

const wantsDist = files.includes('dist/')

if (wantsDist) {
  if (!scripts['build:browser']) {
    console.error('prepack: package declares dist/ in files but has no build:browser script')
    process.exit(1)
  }

  const distDir = path.join(pkgDir, 'dist')
  const wantJs = path.join(distDir, baseName + '.js')
  const wantMjs = path.join(distDir, baseName + '.mjs')

  if (skipBuild) {
    if (!fs.existsSync(wantJs) || !fs.existsSync(wantMjs)) {
      console.error('prepack: WLEARN_SKIP_BUILD=1 but browser dist files are missing')
      process.exit(1)
    }
    console.log('prepack: WLEARN_SKIP_BUILD=1, skipping browser dist rebuild')
  } else {
    run('npm', ['run', 'build:browser'])
  }
}

for (const entry of files) {
  const full = path.join(pkgDir, entry.replace(/\/$/, ''))
  if (!fs.existsSync(full)) {
    console.error('prepack: missing published path ' + entry)
    process.exit(1)
  }
}
