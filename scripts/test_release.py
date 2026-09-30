import base64
import hashlib
import importlib.util
import io
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

    def test_successful_upload_requires_registry_confirmation(self):
        manifest = dict(repos=[], packages=[self.package])
        with patch.object(release, 'preflight', return_value=({self.package['id']: 'missing'}, {})), patch.object(release, 'run') as execute, patch.object(release, 'package_state', return_value='missing'):
            with self.assertRaisesRegex(release.ReleaseError, 'registry has not confirmed'):
                release.publish(manifest, self.root, self.root)
            command = next(c.args[0] for c in execute.call_args_list if 'publish' in c.args[0])
            self.assertIn('--ignore-scripts', command)
            self.assertIn(str(self.archive), command)

    def test_already_published_packages_are_not_uploaded(self):
        manifest = {'repos': [], 'packages': [self.package]}
        with patch.object(release, 'preflight', return_value=({self.package['id']: 'published'}, {})), patch.object(release, 'run') as execute:
            release.publish(manifest, self.root, self.root)
            calls = [call.args[0] for call in execute.call_args_list]
            self.assertFalse(any('publish' in args or 'upload' in args for args in calls))


if __name__ == '__main__':
    unittest.main()
