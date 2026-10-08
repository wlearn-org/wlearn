import base64
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import sys
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

spec = importlib.util.spec_from_file_location('release', Path(__file__).with_name('release.py'))
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / 'core.tgz'
        with tarfile.open(self.archive, 'w:gz') as tf:
            data = json.dumps({'name': '@wlearn/core', 'version': '0.3.0'}).encode()
            info = tarfile.TarInfo('package/package.json')
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        self.package = {'id': 'npm:@wlearn/core', 'name': '@wlearn/core', 'kind': 'npm',
                        'version': '0.3.0', 'artifact': 'core.tgz',
                        'sha256': release.digest(self.archive)}

    def test_packaged_readmes_reject_stale_release_notices(self):
        for kind in ['npm', 'pypi']:
            for notice in ['This corrects the unreleased estimator contract',
                           'This corrects the\nunreleased estimator contract',
                           'Unreleased main: install from source',
                           'Registry publication is pending']:
                with self.subTest(kind=kind, notice=notice):
                    with tarfile.open(self.archive, 'w:gz') as tf:
                        prefix = 'package' if kind == 'npm' else 'wlearn-0.3.0'
                        metadata = ('package.json', b'{"name":"@wlearn/core","version":"0.3.0"}') if kind == 'npm' else ('PKG-INFO', b'Name: wlearn\nVersion: 0.3.0\n')
                        for name, data in [metadata, ('README.md', notice.encode())]:
                            info = tarfile.TarInfo(prefix + '/' + name)
                            info.size = len(data)
                            tf.addfile(info, io.BytesIO(data))
                    with self.assertRaisesRegex(release.ReleaseError, 'stale release notice'):
                        release.archive_identity(self.archive, kind)

    def test_topological_order_and_independent_versions(self):
        packages = [{'id': 'sdk', 'version': '0.3.0', 'needs': ['rf', 'core']},
                    {'id': 'rf', 'version': '0.5.0', 'needs': ['core']},
                    {'id': 'core', 'version': '0.3.0'}]
        self.assertEqual([p['id'] for p in release.ordered(packages)], ['core', 'rf', 'sdk'])
        for invalid in [[{'id': 'x'}, {'id': 'x'}], [{'id': 'x', 'needs': ['absent']}],
                        [{'id': 'x', 'needs': ['y']}, {'id': 'y', 'needs': ['x']}]]:
            with self.assertRaises(release.ReleaseError):
                release.ordered(invalid)

    def test_only_404_is_absence(self):
        for status in [401, 403, 429, 500]:
            with patch.object(release.urllib.request, 'urlopen', side_effect=urllib.error.HTTPError('url', status, '', {}, None)):
                with self.assertRaises(release.ReleaseError):
                    release.get_json('https://example.org')
        with patch.object(release.urllib.request, 'urlopen', side_effect=urllib.error.HTTPError('url', 404, '', {}, None)):
            self.assertIsNone(release.get_json('https://example.org'))

    def test_npm_resume_requires_identical_bytes(self):
        integrity = 'sha512-' + base64.b64encode(hashlib.sha512(self.archive.read_bytes()).digest()).decode()
        with patch.object(release, 'get_json', return_value={'dist': {'integrity': integrity}}):
            self.assertEqual(release.package_state(self.package, self.root), 'published')
        with patch.object(release, 'get_json', return_value={'dist': {'integrity': 'sha512-wrong'}}):
            with self.assertRaisesRegex(release.ReleaseError, 'differs'):
                release.package_state(self.package, self.root)
        with patch.object(release, 'get_json', return_value=None):
            self.assertEqual(release.package_state(self.package, self.root), 'missing')

    def test_missing_old_version_does_not_replace_latest(self):
        for kind in ['npm', 'pypi']:
            package = dict(self.package, kind=kind)
            latest = {'version': '0.4.0'} if kind == 'npm' else {'info': {'version': '0.4.0'}}
            with patch.object(release, 'get_json', side_effect=[None, latest]):
                with self.assertRaisesRegex(release.ReleaseError, 'older than published'):
                    release.package_state(package, self.root)

    def test_pypi_existing_version_without_exact_file_is_conflict(self):
        p = {**self.package, 'kind': 'pypi', 'name': 'wlearn'}
        data = {'urls': [{'filename': 'core.tgz', 'digests': {'sha256': p['sha256']}}]}
        with patch.object(release, 'get_json', return_value=data):
            self.assertEqual(release.package_state(p, self.root), 'published')
        for data in [{'urls': []}, {'urls': [{'filename': 'core.tgz', 'digests': {'sha256': 'wrong'}}]}]:
            with patch.object(release, 'get_json', return_value=data):
                with self.assertRaises(release.ReleaseError):
                    release.package_state(p, self.root)

    def test_tag_conflict_aborts_before_upload(self):
        repo = {'github': 'wlearn-org/core', 'tag': 'v0.3.0', 'commit': 'a' * 40}
        with patch.object(release, 'get_json', return_value={}), patch.object(release, 'remote_commit', return_value='b' * 40):
            with self.assertRaisesRegex(release.ReleaseError, 'tag points'):
                release.github_state(repo, self.root)
        with patch.object(release, 'preflight', side_effect=release.ReleaseError('conflict')), patch.object(release, 'run') as execute:
            with self.assertRaises(release.ReleaseError):
                release.publish({}, self.root, self.root)
            execute.assert_not_called()

    def test_annotated_tag_resolves_peeled_commit(self):
        repo = {'github': 'wlearn-org/core', 'tag': 'v0.3.0'}
        refs = 'tagobject\trefs/tags/v0.3.0\ncommit\trefs/tags/v0.3.0^{}'
        with patch.object(release, 'run', return_value=refs):
            self.assertEqual(release.remote_commit(repo, self.root), 'commit')

    def test_not_qualified_refuses_even_matching_artifact(self):
        with self.assertRaisesRegex(release.ReleaseError, 'not qualified'):
            release.validate({'schema': 1, 'status': 'prepared'}, self.root, self.root)

    def test_qualification_binds_evidence_archive_and_committed_metadata(self):
        log = self.root / 'test.log'
        log.write_text('tests passed\n')
        metadata = '{"name":"@wlearn/core","version":"0.3.0"}'
        package = dict(self.package, repo='core', source='package.json', needs=[],
                       source_sha256=hashlib.sha256(metadata.encode()).hexdigest())
        manifest = dict(schema=1, status='qualified',
                        checks=[dict(log='test.log', exit=0, sha256=release.digest(log))],
                        repos=[dict(id='core', github='wlearn-org/core', path='core', commit='a'*40)],
                        packages=[package])
        with patch.object(release, 'run', return_value=metadata):
            release.validate(manifest, self.root, self.root)
            package['source_sha256'] = 'wrong'
            with self.assertRaisesRegex(release.ReleaseError, 'metadata differs'):
                release.validate(manifest, self.root, self.root)
            package['source_sha256'] = hashlib.sha256(metadata.encode()).hexdigest()
            package['needs'] = ['npm:missing']
            with self.assertRaises(release.ReleaseError):
                release.validate(manifest, self.root, self.root)
            package['needs'] = []
            self.archive.write_bytes(b'tampered')
            with self.assertRaisesRegex(release.ReleaseError, 'Archive changed'):
                release.validate(manifest, self.root, self.root)
            log.write_text('replaced evidence')
            with self.assertRaisesRegex(release.ReleaseError, 'evidence changed'):
                release.validate(manifest, self.root, self.root)

    def test_remote_branch_ancestry_and_divergence(self):
        from types import SimpleNamespace
        repo = dict(github='wlearn-org/core', path='.', branch='main', commit='a'*40)
        for codes, expected in [([0], True), ([1, 0], False), ([1, 1], None)]:
            with patch.object(release, 'run', side_effect=['b'*40 + '\trefs/heads/main', '']), patch.object(release.subprocess, 'run', side_effect=[SimpleNamespace(returncode=c) for c in codes]):
                if expected is None:
                    with self.assertRaisesRegex(release.ReleaseError, 'diverges'):
                        release.branch_needs_push(repo, self.root)
                else:
                    self.assertEqual(release.branch_needs_push(repo, self.root), expected)

    def test_existing_github_asset_requires_matching_content(self):
        asset = dict(size=self.archive.stat().st_size, digest='sha256:' + release.digest(self.archive))
        self.assertTrue(release.asset_matches(asset, self.archive))
        asset['digest'] = 'sha256:wrong'
        self.assertFalse(release.asset_matches(asset, self.archive))

    def test_refresh_registry_state_before_upload(self):
        manifest = dict(repos=[], packages=[self.package])
        states = [({self.package['id']: 'missing'}, {}),
                  ({self.package['id']: 'published'}, {})]
        with patch.object(release, 'preflight', side_effect=states), patch.object(release, 'run') as execute, patch.object(release, 'package_state', return_value='published'):
            release.publish(manifest, self.root, self.root)
            self.assertFalse(any('publish' in c.args[0] for c in execute.call_args_list))

    def test_upload_error_requires_matching_registry_archive(self):
        manifest = dict(repos=[], packages=[self.package])
        states = [({self.package['id']: 'missing'}, {}),
                  ({self.package['id']: 'published'}, {})]
        def execute(args, **kwargs):
            if 'publish' in args:
                raise release.ReleaseError('npm exit 1: already published')
        with patch.object(release, 'preflight', side_effect=states), patch.object(release, 'run', side_effect=execute), patch.object(release, 'package_state', side_effect=['missing', 'published']):
            release.publish(manifest, self.root, self.root)

    def test_receipt_rejects_changed_archive(self):
        release.publication_receipt(self.package, self.root, record=True)
        changed = dict(self.package, sha256='0' * 64)
        with self.assertRaisesRegex(release.ReleaseError, 'different archive'):
            release.publication_receipt(changed, self.root)

    def test_failed_upload_does_not_hide_registry_conflict(self):
        manifest = dict(repos=[], packages=[self.package])
        def execute(args, **kwargs):
            if 'publish' in args:
                raise release.ReleaseError('npm exit 1')
        with patch.object(release, 'preflight', return_value=({self.package['id']: 'missing'}, {})), patch.object(release, 'run', side_effect=execute), patch.object(release, 'package_state', side_effect=['missing', release.ReleaseError('published archive differs')]):
            with self.assertRaisesRegex(release.ReleaseError, 'npm exit 1; published archive differs'):
                release.publish(manifest, self.root, self.root)
            self.assertFalse(release.publication_receipt(self.package, self.root))

    def test_upload_all_packages_without_waiting_for_visibility(self):
        second = dict(self.package, id='npm:@wlearn/second', name='@wlearn/second',
                      needs=[self.package['id']])
        packages = [self.package, second]
        manifest = dict(repos=[], packages=packages)
        states = {p['id']: 'missing' for p in packages}
        with patch.object(release, 'preflight', return_value=(states, {})), patch.object(release, 'run') as execute, patch.object(release, 'package_state', return_value='missing'), patch('time.sleep', side_effect=AssertionError('Must not wait')):
            release.publish(manifest, self.root, self.root)
            self.assertEqual(sum('publish' in c.args[0] for c in execute.call_args_list), 2)
            for package in packages:
                self.assertTrue(release.publication_receipt(package, self.root))
            execute.reset_mock()
            release.publish(manifest, self.root, self.root)
            self.assertFalse(any('publish' in c.args[0] for c in execute.call_args_list))

    def test_failed_upload_without_confirmation_stops(self):
        manifest = dict(repos=[], packages=[self.package])
        def execute(args, **kwargs):
            if 'publish' in args:
                raise release.ReleaseError('npm authentication failed')
        with patch.object(release, 'preflight', return_value=({self.package['id']: 'missing'}, {})), patch.object(release, 'run', side_effect=execute), patch.object(release, 'package_state', return_value='missing'):
            with self.assertRaisesRegex(release.ReleaseError, 'npm authentication failed'):
                release.publish(manifest, self.root, self.root)
            self.assertFalse(release.publication_receipt(self.package, self.root))

    def test_pypi_upload_allows_twine_credential_prompt(self):
        package = dict(self.package, id='pypi:wlearn-bo', name='wlearn-bo', kind='pypi')
        manifest = dict(repos=[], packages=[package])
        with patch.object(release, 'preflight', return_value=({package['id']: 'missing'}, {})), patch.object(release, 'run') as execute, patch.object(release, 'package_state', return_value='missing'):
            release.publish(manifest, self.root, self.root, twine=['python', '-m', 'twine'])
            upload = next(c for c in execute.call_args_list if 'upload' in c.args[0])
            self.assertNotIn('--non-interactive', upload.args[0])
            self.assertIn('https://upload.pypi.org/legacy/', upload.args[0])
            self.assertTrue(upload.kwargs['stream'])

    def test_streamed_twine_http_status_is_detected(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(release.ReleaseError) as caught:
            release.run([sys.executable, '-c',
                         "print('ERROR HTTPError: 429 Too Many Requests'); raise SystemExit(1)"],
                        stream=True, capture_stream=True)
        self.assertEqual(caught.exception.http_status, 429)
        self.assertIn('Too Many Requests', output.getvalue())

    def test_pypi_creation_limit_defers_new_projects_but_finishes_other_uploads(self):
        rf = dict(self.package, id='pypi:rf', kind='pypi', name='rf', artifact='rf.tar.gz')
        sym = dict(rf, id='pypi:sym', name='sym', artifact='sym.tar.gz')
        existing = dict(rf, id='pypi:existing', name='existing', artifact='existing.tar.gz')
        packages = [rf, sym, existing, self.package]
        repo = dict(id='core', github='wlearn-org/core', path='.', commit='a'*40, branch='main', tag='v0.3.0')
        manifest = dict(repos=[repo], packages=packages)
        pending = dict(repository=True, tag='a'*40, release=None, push_branch=False)
        done = dict(pending, release={'assets': []})
        states = {p['id']: 'missing' for p in packages}
        calls = []
        def execute(args, *a, **kwargs):
            calls.append(args)
            if 'upload' in args and args[-1].endswith('/rf.tar.gz'):
                error = release.ReleaseError('HTTPError: 429 Too Many Requests')
                error.http_status = 429
                raise error
        def project(url):
            return {'info': {}} if '/existing/' in url else None
        with patch.object(release, 'preflight', side_effect=[(states, {'core': pending}), (states, {'core': done})]), patch.object(release, 'run', side_effect=execute), patch.object(release, 'package_state', return_value='missing'), patch.object(release, 'get_json', side_effect=project), patch.object(release, 'github_state', return_value=pending), patch.object(release, 'release_assets', return_value=[]), patch('time.sleep', side_effect=AssertionError('Must not retry daily quota')):
            with self.assertRaisesRegex(release.ReleaseError, 'deferred'):
                release.publish(manifest, self.root, self.root)
        uploads = [c for c in calls if 'upload' in c or 'publish' in c]
        self.assertEqual(len(uploads), 3)
        self.assertTrue(any(c[:3] == ['gh', 'release', 'create'] for c in calls))
        self.assertFalse(release.publication_receipt(rf, self.root))
        self.assertFalse(release.publication_receipt(sym, self.root))
        self.assertTrue(release.publication_receipt(existing, self.root))
        self.assertTrue(release.publication_receipt(self.package, self.root))

    def test_npm_only_has_no_twine_github_or_wait(self):
        python = dict(self.package, id='pypi:wlearn', kind='pypi')
        manifest = dict(repos=[], packages=[self.package, python])
        with patch.object(release, 'validate'), patch.object(release, 'run') as execute, patch.object(release, 'package_state', return_value='missing') as state, patch('time.sleep', side_effect=AssertionError('No waits')):
            release.publish_npm(manifest, self.root, self.root)
            release.publish_npm(manifest, self.root, self.root)
            self.assertTrue(all(c.args[0][0] == 'npm' for c in execute.call_args_list))
            self.assertEqual(sum('publish' in c.args[0] for c in execute.call_args_list), 1)
            self.assertTrue(all(c.args[0]['kind'] == 'npm' for c in state.call_args_list))

    def test_github_only_pushes_exact_commit_and_releases_without_registries(self):
        repo = dict(id='core', github='wlearn-org/core', path='.', commit='a'*40, branch='main', tag='v0.3.0')
        manifest = dict(repos=[repo], packages=[])
        pending = dict(repository=True, tag=None, release=None, push_branch=True)
        done = dict(repository=True, tag='a'*40, release={'assets': []}, push_branch=False)
        with patch.object(release, 'preflight', side_effect=[({}, {'core': pending}), ({}, {'core': done})]) as check, patch.object(release, 'run') as execute, patch.object(release, 'github_state', return_value=pending), patch.object(release, 'release_assets', return_value=[]), patch.object(release, 'package_state', side_effect=AssertionError('No registry requests')):
            release.publish_github(manifest, self.root, self.root)
            calls = [c.args[0] for c in execute.call_args_list]
            self.assertTrue(all(c[0] in ['gh', 'git'] for c in calls))
            self.assertTrue(any(c[-1] == 'a'*40 + ':refs/heads/main' for c in calls))
            self.assertTrue(any(c[:3] == ['gh', 'release', 'create'] for c in calls))
            self.assertTrue(all(c.kwargs['github_only'] for c in check.call_args_list))

    def test_already_published_packages_are_not_uploaded(self):
        manifest = {'repos': [], 'packages': [self.package]}
        with patch.object(release, 'preflight', return_value=({self.package['id']: 'published'}, {})), patch.object(release, 'run') as execute:
            release.publish(manifest, self.root, self.root)
            calls = [call.args[0] for call in execute.call_args_list]
            self.assertFalse(any('publish' in args or 'upload' in args for args in calls))


if __name__ == '__main__':
    unittest.main()
