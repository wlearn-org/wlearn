#!/usr/bin/env python3
"""Run the first JS/Python example in each product README against installed packages.

Examples that name training data, an objective, or a previously saved model get
explicit deterministic fixtures. Later reference/browser/shell fragments are not
executed by this quick-start gate; package suites cover those APIs separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--consumer', type=Path, required=True, help='isolated npm install directory')
    parser.add_argument('--python', required=True, help='isolated Python interpreter')
    parser.add_argument('--readme', action='append', help='limit to these workspace-relative README paths')
    parser.add_argument('--language', choices=['js', 'python'])
    args = parser.parse_args()
    root, out = args.workspace.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for repo in json.loads(args.inventory.read_text()):
        paths.append(root / repo['repo'] / 'README.md')
        paths.extend(root / Path(p['path']).parent / 'README.md' for p in repo['npm'])
        if 'python' in repo:
            paths.append(root / repo['repo'] / 'py/README.md')
    blocks = {}
    for path in dict.fromkeys(paths):
        if args.readme and str(path.relative_to(root)) not in args.readme:
            continue
        if not path.exists():
            continue
        text, seen = path.read_text(), set()
        for match in re.finditer(r'```(\w+)\n(.*?)```', text, re.S):
            lang = 'js' if match[1] == 'javascript' else match[1]
            if lang not in ('js', 'python') or lang in seen or args.language and lang != args.language:
                continue
            seen.add(lang)
            key = hashlib.sha256((lang + match[2]).encode()).hexdigest()
            block = blocks.setdefault(key, dict(lang=lang, code=match[2], sources=[]))
            block['sources'].append(f'{path.relative_to(root)}:{text[:match.start()].count(chr(10)) + 1}')
    if not blocks:
        raise SystemExit('No README examples selected')
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'POLY_LIB', 'WLEARN_SYM_POLYGRAD_JS', 'TRANFI_LIB_PATH'):
        env.pop(key, None)
    env.update(POLY_DEV='CPU', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    matrix = [[i / 20 - 2, (i % 7) / 7, (i % 11) / 11, (i % 5) / 5] for i in range(80)]
    labels = [int(row[0] > 0) for row in matrix]
    fixtures = dict(X=matrix, X_train=matrix, X_test=matrix[:8], Xtest=matrix[:8], X_new=matrix[:8], y=labels, y_train=labels, y_test=labels[:8],
                    calibrationProbabilities=[[.1, .9], [.7, .3]] * 20, calibrationLabels=[[0, 1], [1, 0]] * 20,
                    testProbabilities=[[.3, .7], [.8, .2]], calibration_probabilities=[[.1, .9], [.7, .3]] * 20,
                    calibration_labels=[[0, 1], [1, 0]] * 20, test_probabilities=[[.3, .7], [.8, .2]])
    require = "require('node:module').createRequire(" + json.dumps(str(args.consumer.resolve() / 'package.json')) + ')'
    results = []
    for key, block in blocks.items():
        work = out / key[:10]
        work.mkdir(exist_ok=True)
        print('RUN', block['sources'][0], flush=True)
        fixture_code = ''
        # These examples explicitly require a separately distributed ONNX file
        # or a prior cross-language bundle. Supply real files, never stub APIs.
        if block['sources'][0].startswith('mitra-onnx/'):
            model = root / 'mitra-onnx/mitra-classifier.onnx'
            target = work / model.name
            if not target.exists():
                target.symlink_to(model)
        if block['lang'] == 'python' and "load('model.wlrn')" in block['code']:
            backend, cls = ('ebm', 'EBMModel') if 'wlearn.ebm' in block['code'] else ('xgboost', 'XGBModel')
            setup = f"const req={require}; (async()=>{{const m=await req('@wlearn/{backend}').{cls}.create({{task:'classification',numRound:3,maxRounds:3}}); m.fit({json.dumps(matrix)},{json.dumps(labels)}); require('node:fs').writeFileSync('model.wlrn',m.save());m.dispose()}})().catch(e=>{{console.error(e);process.exitCode=1}})"
            subprocess.run(['node', '-e', setup], cwd=work, env=env, check=True, capture_output=True)
        if block['lang'] == 'js':
            fixture_code = 'globalThis.evaluate = p => -Object.values(p).reduce((s,v)=>s+(typeof v === "number" ? v*v : 0),0)\n'
            code = f'const packageRequire = {require}\nObject.assign(globalThis,{json.dumps(fixtures)})\n' + fixture_code
            code += ';(async (require) => {\n' + block['code'] + '\n})(packageRequire).catch(e=>{console.error(e);process.exitCode=1})'
            command = ['node', '-e', code]
        else:
            code = 'import json\ntry:\n    import numpy as np\nexcept ImportError:\n    np = None\nglobals().update({k: np.asarray(v) if np is not None else v for k,v in json.loads(' + repr(json.dumps(fixtures)) + ').items()})\n'
            code += 'def evaluate(p): return -sum(v*v for v in p.values() if isinstance(v, (int, float)))\n' + block['code']
            command = [args.python, '-c', code]
        try:
            result = subprocess.run(command, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
            status, log = result.returncode, result.stdout
        except subprocess.TimeoutExpired as error:
            status, log = 124, str(error)
        (work / 'output.log').write_text(log)
        results.append(dict(sources=block['sources'], lang=block['lang'], exit=status, log=str(work / 'output.log')))
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print('DONE', status, block['sources'][0], flush=True)
    print(f"{len(results)} distinct examples: {sum(r['exit'] == 0 for r in results)} passed")
    return int(any(r['exit'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
