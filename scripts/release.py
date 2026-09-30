#!/usr/bin/env python3
"""Publish an audited set of immutable npm/PyPI archives and GitHub releases.

Preparation and testing happen before this command. A qualified manifest binds
archives to commits and test evidence; publishing never builds from a worktree.
Python 3.10+, npm, git, gh and Twine are required on the publishing host.
"""
import argparse
import base64
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from email.parser import BytesParser


class ReleaseError(RuntimeError):
    pass


def run(args, cwd=None, stream=False):
    if stream:
        result = subprocess.run(args, cwd=cwd)
        if result.returncode:
            raise ReleaseError(f'{shlex.join(map(str, args))}: exit {result.returncode}')
        return ''
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise ReleaseError(f'{shlex.join(map(str, args))}: {result.stderr.strip() or result.stdout.strip()}')
    return result.stdout.strip()


@functools.lru_cache(maxsize=1)
def github_token():
    token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if token:
        return token
    try:
        result = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def get_json(url):
    headers = {'User-Agent': 'wlearn-release'}
    if url.startswith('https://api.github.com/') and github_token():
        headers['Authorization'] = 'Bearer ' + github_token()
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise ReleaseError(f'{url}: HTTP {error.code}') from error
    except (OSError, ValueError) as error:
        raise ReleaseError(f'{url}: {error}') from error


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def name_key(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def archive_identity(path, kind):
    with tarfile.open(path, 'r:gz') as archive:
        if kind == 'npm':
            data = json.load(archive.extractfile('package/package.json'))
            if data.get('private'):
                raise ReleaseError(f'{path}: private package')
            return data['name'], data['version'], {**data.get('dependencies', {}), **data.get('peerDependencies', {})}
        members = [m for m in archive.getmembers() if m.name.count('/') == 1 and m.name.endswith('/PKG-INFO')]
        if len(members) != 1:
            raise ReleaseError(f'{path}: expected one root PKG-INFO')
        metadata = BytesParser().parsebytes(archive.extractfile(members[0]).read())
        return metadata['Name'], metadata['Version'], metadata.get_all('Requires-Dist', [])


def ordered(packages):
    """Dependencies are manifest package IDs, including peer requirements.

    Optional Python extras do not create edges: base wlearn must be published
    before its optional estimator distributions, which themselves require it.
    """
    remaining = {p['id']: p for p in packages}
    if len(remaining) != len(packages):
        raise ReleaseError('Duplicate package ID')
    result = []
    known = set(remaining)
    for package in packages:
        missing = set(package.get('needs', [])) - known
        if missing:
            raise ReleaseError(f"{package['id']}: unknown dependencies {sorted(missing)}")
    while remaining:
        ready = [p for p in remaining.values() if not set(p.get('needs', [])) & remaining.keys()]
        if not ready:
            raise ReleaseError(f'Dependency cycle: {sorted(remaining)}')
        for package in ready:
            result.append(package)
            del remaining[package['id']]
    return result


def validate(manifest, directory, workspace):
    if manifest.get('schema') != 1 or manifest.get('status') != 'qualified':
        raise ReleaseError('Release manifest is not qualified')
    if not manifest.get('checks') or any(c.get('exit') != 0 for c in manifest['checks']):
        raise ReleaseError('Missing or failed qualification checks')
    for check in manifest['checks']:
        path = directory / check['log']
        if digest(path) != check['sha256']:
            raise ReleaseError(f'Qualification evidence changed: {path}')
    repos = {r['id']: r for r in manifest['repos']}
    if len(repos) != len(manifest['repos']):
        raise ReleaseError('Duplicate repository ID')
    for repo in repos.values():
        if not re.fullmatch(r'wlearn-org/[\w.-]+', repo['github']):
            raise ReleaseError('Unexpected GitHub owner/repository')
        if not re.fullmatch(r'[0-9a-f]{40}', repo['commit']):
            raise ReleaseError(f"Invalid commit: {repo['id']}")
        local = workspace / repo['path']
        # Archives were tested before publishing. Unrelated current worktree
        # edits/HEAD movement cannot alter them; require the recorded Git object.
        run(['git', 'cat-file', '-e', repo['commit'] + '^{commit}'], local)
    for package in ordered(manifest['packages']):
        if package['repo'] not in repos or package['kind'] not in ('npm', 'pypi'):
            raise ReleaseError(f"Invalid package: {package['id']}")
        if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', package['version']):
            raise ReleaseError('Only stable three-component package versions are supported')
        path = directory / package['artifact']
        if digest(path) != package['sha256']:
            raise ReleaseError(f'Archive changed: {path}')
        name, version, dependencies = archive_identity(path, package['kind'])
        if name_key(name) != name_key(package['name']) or version != package['version']:
            raise ReleaseError(f'{path}: archive name/version does not match manifest')
        repo = repos[package['repo']]
        metadata = run(['git', 'show', repo['commit'] + ':' + package['source']], workspace / repo['path'])
        if hashlib.sha256(metadata.encode()).hexdigest() != package['source_sha256']:
            raise ReleaseError(f'{path}: committed package metadata differs from qualification')
        names = dependencies if isinstance(dependencies, dict) else [
            re.match(r'[A-Za-z0-9_.-]+', d).group() for d in dependencies if 'extra ==' not in d
        ]
        expected = {p['id'] for p in manifest['packages']
                    if p['kind'] == package['kind'] and p['name'] in names}
        if set(package.get('needs', [])) != expected:
            raise ReleaseError(f"{package['id']}: dependency order differs from archive requirements")
    return repos


def reject_older_release(package, url):
    data = get_json(url)
    if data is None:
        return
    latest = data.get('info', data).get('version', '')
    # This driver qualifies stable three-component versions. Prerelease channels
    # need an explicit dist-tag policy rather than silently replacing latest.
    if re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', latest):
        if tuple(map(int, package['version'].split('.'))) < tuple(map(int, latest.split('.'))):
            raise ReleaseError(f"{package['name']}: candidate {package['version']} is older than published {latest}")


def package_state(package, directory):
    name, version = package['name'], package['version']
    if package['kind'] == 'npm':
        data = get_json('https://registry.npmjs.org/' + urllib.parse.quote(name, safe='') + '/' + version)
        if data is None:
            reject_older_release(package, 'https://registry.npmjs.org/' + urllib.parse.quote(name, safe='') + '/latest')
            return 'missing'
        integrity = data.get('dist', {}).get('integrity', '')
        tokens = integrity.split()
        actual = 'sha512-' + base64.b64encode(bytes.fromhex(digest(directory / package['artifact'], 'sha512'))).decode()
        if actual not in tokens:
            raise ReleaseError(f'{name}@{version}: published archive differs (or has no sha512 integrity)')
    else:
        data = get_json(f'https://pypi.org/pypi/{name}/{version}/json')
        if data is None:
            reject_older_release(package, f'https://pypi.org/pypi/{name}/json')
            return 'missing'
        files = {f['filename']: f for f in data['urls']}
        filename = Path(package['artifact']).name
        if filename not in files or files[filename]['digests']['sha256'] != package['sha256']:
            raise ReleaseError(f'{name}=={version}: existing release does not match qualified sdist')
    return 'published'


def remote_commit(repo, workspace):
    url = f"https://github.com/{repo['github']}.git"
    refs = run(['git', 'ls-remote', url, f"refs/tags/{repo['tag']}", f"refs/tags/{repo['tag']}^{{}}"], workspace)
    values = dict(line.split()[::-1] for line in refs.splitlines())
    return values.get(f"refs/tags/{repo['tag']}^{{}}", values.get(f"refs/tags/{repo['tag']}"))


def github_state(repo, workspace):
    data = get_json(f"https://api.github.com/repos/{repo['github']}")
    if data is None:
        return {'repository': False, 'tag': None, 'release': None}
    target = remote_commit(repo, workspace)
    if target is not None and target != repo['commit']:
        raise ReleaseError(f"{repo['github']} {repo['tag']}: tag points at {target}, expected {repo['commit']}")
    release = get_json(f"https://api.github.com/repos/{repo['github']}/releases/tags/{repo['tag']}")
    if release is not None and target is None:
        raise ReleaseError(f"{repo['github']}: release has no verified tag")
    if release is not None and (release.get('draft') or release.get('prerelease')):
        raise ReleaseError(f"{repo['github']}: existing release is not a final public release")
    return {'repository': True, 'tag': target, 'release': release}


def branch_needs_push(repo, workspace):
    """Check fast-forward safety before uploading anything to a registry."""
    local = workspace / repo['path']
    url = f"https://github.com/{repo['github']}.git"
    refs = run(['git', 'ls-remote', url, f"refs/heads/{repo['branch']}"], local)
    if not refs:
        return True
    current = refs.split()[0]
    if current == repo['commit']:
        return False
    try:
        run(['git', 'cat-file', '-e', current + '^{commit}'], local)
    except ReleaseError:
        # Fetching an immutable object changes no branch or worktree.
        run(['git', 'fetch', '--no-tags', url, current], local)
    def ancestor(a, b):
        result = subprocess.run(['git', 'merge-base', '--is-ancestor', a, b], cwd=local, capture_output=True)
        if result.returncode not in (0, 1):
            raise ReleaseError(f"{repo['github']}: cannot verify branch ancestry")
        return result.returncode == 0
    if ancestor(current, repo['commit']):
        return True
    if ancestor(repo['commit'], current):
        return False  # The branch already includes this release's commit.
    raise ReleaseError(f"{repo['github']}: release commit diverges from remote {repo['branch']}")


def asset_matches(asset, path):
    if asset['size'] != path.stat().st_size:
        return False
    expected = 'sha256:' + digest(path)
    if asset.get('digest'):
        return asset['digest'] == expected
    # Older GitHub uploads have no digest field; verify their actual bytes.
    h = hashlib.sha256()
    with urllib.request.urlopen(asset['browser_download_url'], timeout=60) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest() == expected.split(':', 1)[1]


def release_assets(repo, manifest, directory):
    paths = [directory / p['artifact'] for p in manifest['packages'] if p['repo'] == repo['id']]
    for item in repo.get('assets', []):
        path = directory / item['path']
        if digest(path) != item['sha256']:
            raise ReleaseError(f'GitHub asset changed: {path}')
        paths.append(path)
    if len({p.name for p in paths}) != len(paths):
        raise ReleaseError(f"{repo['id']}: duplicate asset filename")
    return paths


def preflight(manifest, directory, workspace):
    repos = validate(manifest, directory, workspace)
    states = {}
    for package in ordered(manifest['packages']):
        states[package['id']] = package_state(package, directory)
        print(f"{states[package['id']]:10} {package['id']} {package['version']}", flush=True)
    github = {}
    for key, repo in repos.items():
        state = github_state(repo, workspace)
        state['push_branch'] = branch_needs_push(repo, workspace) if state['repository'] else True
        existing = {a['name']: a for a in (state['release'] or {}).get('assets', [])}
        for path in release_assets(repo, manifest, directory):
            if path.name in existing and not asset_matches(existing[path.name], path):
                raise ReleaseError(f"{repo['github']}: existing asset differs: {path.name}")
        github[key] = state
        print(f"GitHub {repo['github']} {repo['tag']}: " + ('released' if state['release'] else 'pending'), flush=True)
    return states, github


def publish(manifest, directory, workspace, create_repos=False, twine=None):
    # Check every immutable conflict before the first write. A resumed release
    # derives progress from registries/tags, never from a stale local done flag.
    states, github = preflight(manifest, directory, workspace)
    run(['npm', 'whoami', '--registry=https://registry.npmjs.org'])
    run(['gh', 'auth', 'status'])
    twine = twine or [sys.executable, '-m', 'twine']
    run([*twine, '--version'])
    missing = [r['github'] for r in manifest['repos'] if not github[r['id']]['repository']]
    if missing and not create_repos:
        raise ReleaseError(f'Missing GitHub repositories: {missing}; pass --create-repos to create public repositories')
    for repo in manifest['repos']:
        if not github[repo['id']]['repository']:
            run(['gh', 'repo', 'create', repo['github'], '--public'], stream=True)
    for package in ordered(manifest['packages']):
        if states[package['id']] == 'published':
            continue
        path = (directory / package['artifact']).resolve()
        print(f"Publishing {package['id']} {package['version']}", flush=True)
        if package['kind'] == 'npm':
            run(['npm', 'publish', str(path), '--ignore-scripts', '--access', 'public', '--registry=https://registry.npmjs.org'], stream=True)
        else:
            # Explicit destination avoids ambient .pypirc selecting TestPyPI.
            run([*twine, 'upload', '--non-interactive', '--repository-url', 'https://upload.pypi.org/legacy/', str(path)], stream=True)
        if package_state(package, directory) != 'published':
            raise ReleaseError(f"{package['id']}: upload returned success but registry has not confirmed it; rerun to resume")
    # Publish dependencies before push-triggered CI attempts their installation.
    for repo in manifest['repos']:
        state = github[repo['id']]
        local = workspace / repo['path']
        # Push the precise tested commit, with normal fast-forward protection.
        # SSH transport works with host git credentials; never force a branch/tag.
        url = f"git@github.com:{repo['github']}.git"
        if state.get('push_branch', True):
            run(['git', 'push', url, f"{repo['commit']}:refs/heads/{repo['branch']}"], local, stream=True)
        if state['tag'] is None:
            run(['git', 'push', url, f"{repo['commit']}:refs/tags/{repo['tag']}"], local, stream=True)
    for repo in manifest['repos']:
        state = github_state(repo, workspace)
        paths = release_assets(repo, manifest, directory)
        if state['release'] is None:
            run(['gh', 'release', 'create', repo['tag'], '--repo', repo['github'], '--verify-tag',
                 '--title', f"{repo['id']} {repo['tag']}", '--generate-notes',
                 '--latest=false' if repo.get('latest') is False else '--latest', *map(str, paths)], stream=True)
        else:
            existing = {a['name']: a for a in state['release'].get('assets', [])}
            missing = [p for p in paths if p.name not in existing]
            if missing:
                run(['gh', 'release', 'upload', repo['tag'], '--repo', repo['github'], *map(str, missing)], stream=True)
    final, releases = preflight(manifest, directory, workspace)
    if any(state != 'published' for state in final.values()):
        raise ReleaseError('Final registry verification is incomplete; rerun to resume')
    for repo in manifest['repos']:
        state = releases[repo['id']]
        names = {a['name'] for a in (state['release'] or {}).get('assets', [])}
        if state['release'] is None or any(p.name not in names for p in release_assets(repo, manifest, directory)):
            raise ReleaseError(f"{repo['github']}: GitHub release is incomplete; rerun to resume")
    print('All registry archives and GitHub releases verified.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'publish'])
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--create-repos', action='store_true')
    parser.add_argument('--twine', help='Twine command, e.g. /path/to/python -m twine')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    directory = args.manifest.resolve().parent
    if args.action == 'publish':
        publish(manifest, directory, args.workspace, args.create_repos, shlex.split(args.twine) if args.twine else None)
    else:
        preflight(manifest, directory, args.workspace)


if __name__ == '__main__':
    try:
        main()
    except (ReleaseError, OSError, ValueError, KeyError, tarfile.TarError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
