"""Coverage regressions for the README gate."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('readmes', Path(__file__).with_name('check-readmes.py'))
readmes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readmes)


class ReadmeCoverageTest(unittest.TestCase):
    def test_every_executable_block_is_selected_not_only_the_quick_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            doc = root / 'README.md'
            doc.write_text('```js\nfirst()\n```\n```js\nsecond()\n```\n'
                           '```python\nthird()\n```\n```text\nmethod(arg?)\n```\n')
            blocks = readmes.extract_blocks([doc], root)
            self.assertEqual([b['code'].strip() for b in blocks.values()],
                             ['first()', 'second()', 'third()'])
            self.assertEqual([b['sources'][0] for b in blocks.values()],
                             ['README.md:1', 'README.md:4', 'README.md:7'])

    def test_deduplication_retains_each_source_and_hash_changes_with_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / 'README.md', root / 'other.md']
            for p in paths:
                p.write_text('```js\ncall()\n```')
            original = readmes.extract_blocks(paths, root)
            self.assertEqual(len(original), 1)
            self.assertEqual(len(next(iter(original.values()))['sources']), 2)
            paths[1].write_text('```js\nchanged()\n```')
            self.assertEqual(len(readmes.extract_blocks(paths, root)), 2)


if __name__ == '__main__':
    unittest.main()
