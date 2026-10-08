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


def run(args, cwd=None, stream=False, capture_stream=False):
    if stream and capture_stream:
        # Keep stdin/TTY available to Twine's secure credential prompt. Tee only
        # output, retaining a bounded tail to distinguish HTTP throttling.
        tail = ''
        with subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                tail = (tail + line)[-65536:]
            code = process.wait()
        if code:
            error = ReleaseError(f'{shlex.join(map(str, args))}: exit {code}')
            plain = re.sub(r'\x1b\[[0-9;]*m', '', tail)
            match = re.search(r'HTTPError:\s*(\d{3})\b', plain)
            error.http_status = int(match.group(1)) if match else None
            raise error
        return ''
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
    headers = {'User-Agent': 'wlearn-release', 'Cache-Control': 'no-cache'}
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
        # Inspect the shipped copies, including Python package READMEs. Whitespace
        # normalization catches prose wrapped differently from the root README.
        for member in archive.getmembers():
            if member.isfile() and Path(member.name).name == 'README.md':
                text = archive.extractfile(member).read().decode('utf-8')
                prose = ' '.join(text.lower().split())
                if any(notice in prose for notice in (
                    'unreleased main', 'unreleased estimator contract',
                    'registry publication is pending', 'build this branch from source',
                )):
                    raise ReleaseError(f'{path}: stale release notice in {member.name}')
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


def preflight(manifest, directory, workspace, github_only=False):
    repos = validate(manifest, directory, workspace)
    states = {}
    for package in ([] if github_only else ordered(manifest['packages'])):
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


def publication_receipt(package, directory, record=False):
    # A receipt means upload was accepted, never that public bytes were verified.
    # Retain it across restarts so delayed metadata cannot cause a second upload.
    path = directory / 'publication-receipts.json'
    receipts = json.loads(path.read_text()) if path.exists() else {}
    key = package['id'] + '@' + package['version']
    saved = receipts.get(key)
    if saved is not None and saved != package['sha256']:
        raise ReleaseError(f"{key}: accepted upload receipt belongs to a different archive")
    if record:
        receipts[key] = package['sha256']
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(receipts, indent=2) + '\n')
        temporary.replace(path)
    return saved is not None


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
    deferred = []
    pypi_creation_limited = False
    for package in ordered(manifest['packages']):
        if states[package['id']] == 'published':
            continue
        if publication_receipt(package, directory):
            print(f"Skipping {package['id']}; upload already accepted", flush=True)
            continue
        # Preflight can be minutes old after authentication or earlier uploads.
        if package_state(package, directory) == 'published':
            continue
        if package['kind'] == 'pypi' and pypi_creation_limited:
            project = get_json(f"https://pypi.org/pypi/{package['name']}/json")
            if project is None:
                deferred.append(package['id'])
                print(f"Deferring {package['id']}; PyPI new-project quota was reached", flush=True)
                continue
        path = (directory / package['artifact']).resolve()
        print(f"Publishing {package['id']} {package['version']}", flush=True)
        if package['kind'] == 'npm':
            command = ['npm', 'publish', str(path), '--ignore-scripts', '--access', 'public', '--registry=https://registry.npmjs.org']
        else:
            # Keep the real PyPI destination and inherit the terminal for credential prompts.
            command = [*twine, 'upload', '--repository-url', 'https://upload.pypi.org/legacy/', str(path)]
        try:
            run(command, stream=True, capture_stream=package['kind'] == 'pypi')
        except ReleaseError as upload_error:
            if package['kind'] == 'pypi' and getattr(upload_error, 'http_status', None) == 429:
                if package_state(package, directory) == 'published':
                    publication_receipt(package, directory, record=True)
                    continue
                # PyPI's new-project limit can span 24 hours. Do not hammer it
                # or prevent independent npm/GitHub releases from completing.
                project = get_json(f"https://pypi.org/pypi/{package['name']}/json")
                pypi_creation_limited = pypi_creation_limited or project is None
                deferred.append(package['id'])
                print(f"Deferring {package['id']}: PyPI HTTP429; no automatic retry", flush=True)
                continue
            # A concurrent/prior upload or lost response can report failure after
            # acceptance. Only matching registry integrity permits continuing.
            print(f"Upload command failed for {package['id']}; checking registry before stopping", flush=True)
            try:
                if package_state(package, directory) != 'published':
                    raise ReleaseError('registry has not confirmed this upload; no receipt recorded')
            except ReleaseError as confirmation_error:
                raise ReleaseError(f'{upload_error}; {confirmation_error}') from upload_error
            publication_receipt(package, directory, record=True)
        else:
            publication_receipt(package, directory, record=True)
    write_github_releases(manifest, directory, workspace, github)
    final, releases = preflight(manifest, directory, workspace)
    pending = [key for key, state in final.items() if state != 'published']
    verify_github_releases(manifest, directory, releases)
    if deferred:
        raise ReleaseError('Other uploads and GitHub releases completed; PyPI packages deferred: '
                           + ', '.join(deferred)
                           + '. Resume after the quota permits; accepted uploads will be skipped.')
    if pending:
        print('Uploads accepted and GitHub releases verified. Registry visibility pending: ' + ', '.join(pending))
        print('Run the check command later to verify availability; accepted uploads will not be repeated.')
    else:
        print('All registry archives and GitHub releases verified.')


def write_github_releases(manifest, directory, workspace, github):
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


def verify_github_releases(manifest, directory, releases):
    for repo in manifest['repos']:
        state = releases[repo['id']]
        names = {a['name'] for a in (state['release'] or {}).get('assets', [])}
        if state['release'] is None or any(p.name not in names for p in release_assets(repo, manifest, directory)):
            raise ReleaseError(f"{repo['github']}: GitHub release is incomplete; rerun to resume")


def publish_github(manifest, directory, workspace, create_repos=False):
    _, github = preflight(manifest, directory, workspace, github_only=True)
    run(['gh', 'auth', 'status'])
    missing = [r['github'] for r in manifest['repos'] if not github[r['id']]['repository']]
    if missing and not create_repos:
        raise ReleaseError(f'Missing GitHub repositories: {missing}; pass --create-repos')
    for name in missing:
        run(['gh', 'repo', 'create', name, '--public'], stream=True)
    write_github_releases(manifest, directory, workspace, github)
    _, releases = preflight(manifest, directory, workspace, github_only=True)
    verify_github_releases(manifest, directory, releases)
    print('All GitHub releases verified. npm and PyPI were not changed.')


def publish_npm(manifest, directory, workspace):
    validate(manifest, directory, workspace)
    packages = [p for p in ordered(manifest['packages']) if p['kind'] == 'npm']
    states = {p['id']: package_state(p, directory) for p in packages}
    run(['npm', 'whoami', '--registry=https://registry.npmjs.org'])
    for package in packages:
        if states[package['id']] == 'published' or publication_receipt(package, directory):
            print(f"Skipping {package['id']}; already published or accepted", flush=True)
            continue
        if package_state(package, directory) == 'published':
            continue
        path = (directory / package['artifact']).resolve()
        try:
            run(['npm', 'publish', str(path), '--ignore-scripts', '--access', 'public',
                 '--registry=https://registry.npmjs.org'], stream=True)
        except ReleaseError:
            if package_state(package, directory) != 'published':
                raise
        publication_receipt(package, directory, record=True)
    print('All npm uploads accepted. PyPI and GitHub were not changed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'publish'])
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--create-repos', action='store_true')
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--github-only', action='store_true', help='Push commits/tags and create GitHub releases only')
    scope.add_argument('--npm-only', action='store_true', help='Publish npm packages only; no PyPI or GitHub operations')
    parser.add_argument('--twine', help='Twine command, e.g. /path/to/python -m twine')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    directory = args.manifest.resolve().parent
    if args.github_only:
        if args.action == 'publish':
            publish_github(manifest, directory, args.workspace, args.create_repos)
        else:
            preflight(manifest, directory, args.workspace, github_only=True)
    elif args.npm_only:
        if args.action != 'publish':
            parser.error('--npm-only requires publish')
        publish_npm(manifest, directory, args.workspace)
    elif args.action == 'publish':
        publish(manifest, directory, args.workspace, args.create_repos, shlex.split(args.twine) if args.twine else None)
    else:
        preflight(manifest, directory, args.workspace)


if __name__ == '__main__':
    try:
        main()
    except (ReleaseError, OSError, ValueError, KeyError, tarfile.TarError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
