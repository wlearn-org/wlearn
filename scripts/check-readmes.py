#!/usr/bin/env python3
"""Execute every JS/Python README block against an installed ecosystem.

The reviewed catalog supplies data and explicit prior-example context, never
mock model APIs. Its content hashes make new/changed examples fail closed until
reviewed. Reference signatures use text fences and are not executable examples.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def extract_blocks(paths, root):
    blocks = {}
    for path in dict.fromkeys(paths):
        if not path.exists():
            continue
        text = path.read_text()
        for match in re.finditer(r'```(\w+)\n(.*?)```', text, re.S):
            lang = 'js' if match[1] == 'javascript' else match[1]
            if lang not in ('js', 'python'):
                continue
            key = hashlib.sha256((lang + match[2]).encode()).hexdigest()
            block = blocks.setdefault(key, dict(lang=lang, code=match[2], sources=[]))
            block['sources'].append(f'{path.relative_to(root)}:{text[:match.start()].count(chr(10)) + 1}')
    return blocks


def example_paths(root, inventory):
    paths = []
    for repo in inventory:
        paths.append(root / repo['repo'] / 'README.md')
        paths.extend(root / Path(p['path']).parent / 'README.md' for p in repo['npm'])
        if 'python' in repo:
            paths.append(root / repo['repo'] / 'py/README.md')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--consumer', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--catalog', type=Path, default=Path(__file__).with_name('readme-cases.json'))
    parser.add_argument('--readme', action='append')
    parser.add_argument('--language', choices=['js', 'python'])
    args = parser.parse_args()
    root, out = args.workspace.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = example_paths(root, json.loads(args.inventory.read_text()))
    if args.readme:
        paths = [p for p in paths if str(p.relative_to(root)) in args.readme]
    blocks = extract_blocks(paths, root)
    if args.language:
        blocks = {k: b for k, b in blocks.items() if b['lang'] == args.language}
    if not blocks:
        raise SystemExit('No README examples selected')
    catalog = json.loads(args.catalog.read_text())
    unknown = [b['sources'] for key, b in blocks.items() if key not in catalog]
    if unknown:
        raise SystemExit('Unreviewed README examples: ' + json.dumps(unknown))
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'POLY_LIB', 'WLEARN_SYM_POLYGRAD_JS', 'TRANFI_LIB_PATH'):
        env.pop(key, None)
    env.update(POLY_DEV='CPU', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    require = "require('node:module').createRequire(" + json.dumps(str(args.consumer.resolve() / 'package.json')) + ')'
    results = []
    for key, block in blocks.items():
        case = catalog[key]
        work = out / key[:12]
        work.mkdir(exist_ok=True)
        for filename in case.get('files', []):
            target = work / Path(filename).name
            if not target.exists():
                target.symlink_to(root / filename)
        print('RUN', block['sources'][0], flush=True)
        # Cross-language loading examples require real bundles from the installed
        # JS backend, not arbitrary bytes or a patched loader.
        if case.get('bundle'):
            backend, cls = case['bundle']
            bootstrap = f"const packageRequire={require}; (async(require)=>{{const m=await require('@wlearn/{backend}').{cls}.create({{task:'classification',numRound:3,maxRounds:3}});m.fit([[-2,-2],[-1,-1],[1,1],[2,2]],[0,0,1,1]);require('node:fs').writeFileSync('model.wlrn',m.save());m.dispose()}})(packageRequire).catch(e=>{{console.error(e);process.exitCode=1}})"
            subprocess.run(['node', '-e', bootstrap], cwd=work, env=env, check=True, capture_output=True)
        data = case.get('data', {})
        fixture = Path(__file__).with_name('readme-fixtures.' + ('js' if block['lang'] == 'js' else 'py')).read_text()
        if block['lang'] == 'js':
            code = 'const packageRequire = ' + require + '\n'
            code += ';(async (require) => {\n' + fixture + '\nObject.assign(globalThis,' + json.dumps(data) + ')\n'
            code += case.get('context', '') + '\n{\n' + block['code'] + '\n' + case.get('after', '') + '\n}'
            code += '\n})(packageRequire).catch(e=>{console.error(e);process.exitCode=1})'
            command = ['node', '-e', code]
        else:
            code = fixture + '\n'
            code += 'globals().update({k: _np.asarray(v) if isinstance(v, list) else v for k,v in _json.loads(' + repr(json.dumps(data)) + ').items()})\n'
            code += case.get('context', '') + '\n' + block['code'] + '\n' + case.get('after', '')
            command = [args.python, '-c', code]
        (work / ('example.js' if block['lang'] == 'js' else 'example.py')).write_text(code)
        try:
            result = subprocess.run(command, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=case.get('timeout', 240))
            status, log = result.returncode, result.stdout
        except subprocess.TimeoutExpired as error:
            status, log = 124, str(error)
        (work / 'output.log').write_text(log)
        results.append(dict(sources=block['sources'], sha256=key, lang=block['lang'], context=case.get('description', 'data fixture only'), exit=status, log=str(work / 'output.log')))
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print('DONE', status, block['sources'][0], flush=True)
    print(f"{len(results)} distinct examples: {sum(r['exit'] == 0 for r in results)} passed")
    return int(any(r['exit'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
