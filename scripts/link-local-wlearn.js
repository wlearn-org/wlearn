#!/usr/bin/env node
'use strict'

const fs = require('fs')
const path = require('path')
const { createRequire } = require('node:module')

const ROOT = path.resolve(__dirname, '..')
const REPOS_ROOT = path.resolve(process.env.WLEARN_REPOS_ROOT || path.join(ROOT, '..'))
const SCOPE_DIR = path.join(ROOT, 'node_modules', '@wlearn')
const args = new Set(process.argv.slice(2))
const unlink = args.has('--unlink')
const force = args.has('--force')
const check = args.has('--check')

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

function removeIfSafe(linkPath) {
  // lstat distinguishes a dangling owned link from an absent destination.
  const stat = fs.lstatSync(linkPath, { throwIfNoEntry: false })
  if (!stat) return true
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
  if (!removeIfSafe(linkPath)) return 'skipped'
  fs.symlinkSync(targetDir, linkPath, process.platform === 'win32' ? 'junction' : 'dir')
  return 'linked'
}

function main() {
  const packages = discoverPackages()
  if (check) {
    checkDependencies(packages)
    return
  }
  if (packages.length === 0) {
    console.log(`no @wlearn sibling packages found under ${REPOS_ROOT}`)
    return
  }

  for (const [name, dir] of packages) {
    const status = linkPackage(name, dir)
    console.log(`${status.padEnd(8)} ${name} -> ${path.relative(ROOT, dir)}`)
  }
}

function parseVersion(value) {
  if (typeof value !== 'string' || !/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(value)) {
    throw new Error(`unsupported version ${JSON.stringify(value)}; expected major.minor.patch`)
  }
  const version = value.split('.').map(Number)
  if (!version.every(Number.isSafeInteger)) throw new Error(`version exceeds safe integer range: ${value}`)
  return version
}

function compare(a, b) {
  for (let i = 0; i < 3; i++) if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1
  return 0
}

function satisfies(version, range) {
  const actual = parseVersion(version)
  // Deliberately bounded to this workspace's grammar; unknown syntax fails
  // explicitly instead of approximating npm's general range semantics.
  const alternatives = range.split('||').map(part => {
    const match = /^(\^)?((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))$/.exec(part.trim())
    if (!match) throw new Error(`unsupported dependency range ${JSON.stringify(range)}; use exact/caret versions or ||`)
    const lower = parseVersion(match[2])
    if (!match[1]) return compare(actual, lower) === 0
    const upper = lower.slice()
    const digit = lower[0] ? 0 : lower[1] ? 1 : 2
    upper[digit]++
    for (let i = digit + 1; i < 3; i++) upper[i] = 0
    return compare(actual, lower) >= 0 && compare(actual, upper) < 0
  })
  return alternatives.some(Boolean)
}

function checkDependencies(siblings) {
  const packages = new Map(siblings)
  for (const workspace of readJson(path.join(ROOT, 'package.json'))?.workspaces || []) {
    const wildcard = workspace.endsWith('/*')
    const base = path.join(ROOT, wildcard ? workspace.slice(0, -2) : workspace)
    const dirs = wildcard
      ? fs.readdirSync(base, { withFileTypes: true }).filter(entry => entry.isDirectory()).map(entry => path.join(base, entry.name))
      : [base]
    for (const dir of dirs) {
      const pkg = packageAt(dir)
      if (pkg) packages.set(pkg.name, dir)
    }
  }
  let checked = 0
  let failures = 0
  let coreEntry
  try { coreEntry = require.resolve(packages.get('@wlearn/core')) } catch (_) { /* Source may not be built yet. */ }
  for (const [name, dir] of [...packages].sort()) {
    const pkg = readJson(path.join(dir, 'package.json'))
    for (const field of ['dependencies', 'peerDependencies', 'optionalDependencies', 'devDependencies']) {
      for (const [dependency, range] of Object.entries(pkg[field] || {})) {
        if (!dependency.startsWith('@wlearn/')) continue
        const target = packages.get(dependency)
        if (!target) {
          console.error(`${name}: missing local source for ${dependency} (${field})`)
          failures++
          continue
        }
        const version = readJson(path.join(target, 'package.json')).version
        checked++
        try {
          if (!satisfies(version, range)) throw new Error(`requires ${range}; local version is ${version}`)
        } catch (error) {
          console.error(`${name}: ${dependency} (${field}) ${error.message}`)
          failures++
        }
      }
    }
    // Resolution alone is read-only: never execute backend modules during a check.
    if (coreEntry) {
      try {
        const installed = createRequire(path.join(dir, 'package.json')).resolve('@wlearn/core')
        if (installed !== coreEntry) {
          console.error(`${name}: different installed core at ${installed}; expected ${coreEntry}`)
          failures++
        }
      } catch (error) {
        if (error.code !== 'MODULE_NOT_FOUND') throw error
      }
    }
  }
  console.log(`checked ${checked} local dependency ranges; ${failures} failures`)
  if (failures) process.exitCode = 1
}

main()
