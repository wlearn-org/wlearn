#!/usr/bin/env node
'use strict'

const fs = require('fs')
const path = require('path')

const ROOT = path.resolve(__dirname, '..')
const REPOS_ROOT = path.resolve(process.env.WLEARN_REPOS_ROOT || path.join(ROOT, '..'))
const SCOPE_DIR = path.join(ROOT, 'node_modules', '@wlearn')
const args = new Set(process.argv.slice(2))
const unlink = args.has('--unlink')
const force = args.has('--force')

const SKIP_DIRS = new Set(['wlearn', 'node_modules', '.git'])

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'))
  } catch (_) {
    return null
  }
}

function packageAt(dir) {
  const pkg = readJson(path.join(dir, 'package.json'))
  if (!pkg || typeof pkg.name !== 'string') return null
  if (!pkg.name.startsWith('@wlearn/')) return null
  return { dir, name: pkg.name }
}

function discoverPackages() {
  const packages = new Map()
  for (const entry of fs.readdirSync(REPOS_ROOT, { withFileTypes: true })) {
    if (!entry.isDirectory() || SKIP_DIRS.has(entry.name) || entry.name.startsWith('test-')) continue
    const repoDir = path.join(REPOS_ROOT, entry.name)
    for (const candidate of [path.join(repoDir, 'js'), repoDir]) {
      const found = packageAt(candidate)
      if (found && !packages.has(found.name)) packages.set(found.name, found.dir)
    }
  }
  return [...packages.entries()].sort((a, b) => a[0].localeCompare(b[0]))
}

function sameTarget(linkPath, targetDir) {
  try {
    return fs.realpathSync(linkPath) === fs.realpathSync(targetDir)
  } catch (_) {
    return false
  }
}

function removeIfSafe(linkPath, targetDir) {
  if (!fs.existsSync(linkPath)) return false
  const stat = fs.lstatSync(linkPath)
  if (stat.isSymbolicLink()) {
    fs.rmSync(linkPath)
    return true
  }
  if (force) {
    fs.rmSync(linkPath, { recursive: true, force: true })
    return true
  }
  console.log(`skip ${path.basename(linkPath)}: existing non-symlink; rerun with --force to replace`)
  return false
}

function linkPackage(name, targetDir) {
  const shortName = name.split('/')[1]
  const linkPath = path.join(SCOPE_DIR, shortName)

  if (unlink) {
    if (!fs.existsSync(linkPath)) return 'missing'
    if (fs.lstatSync(linkPath).isSymbolicLink() && sameTarget(linkPath, targetDir)) {
      fs.rmSync(linkPath)
      return 'unlinked'
    }
    return 'skipped'
  }

  fs.mkdirSync(SCOPE_DIR, { recursive: true })
  if (sameTarget(linkPath, targetDir)) return 'already'
  if (!removeIfSafe(linkPath, targetDir)) return 'skipped'
  fs.symlinkSync(targetDir, linkPath, process.platform === 'win32' ? 'junction' : 'dir')
  return 'linked'
}

function main() {
  const packages = discoverPackages()
  if (packages.length === 0) {
    console.log(`no @wlearn sibling packages found under ${REPOS_ROOT}`)
    return
  }

  for (const [name, dir] of packages) {
    const status = linkPackage(name, dir)
    console.log(`${status.padEnd(8)} ${name} -> ${path.relative(ROOT, dir)}`)
  }
}

main()
