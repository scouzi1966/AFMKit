#!/usr/bin/env python3
"""Qualify a frozen Qwen launch without rebuilding or promoting defaults.

Mixed-lane checks extend the preserved run_qwen_scheduler_lifecycle.py fixture.
This is lifecycle coverage, not a performance or general model-quality claim.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import threading
import time


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def checks(row, cap, logprobs=False, full_replay=False, identity=None, snapshot_backoff_tokens=0,
           allow_endpoint_promotion=False):
    probs = [p for c in row['chunks'] for choice in c.get('choices', [])
             for p in (choice.get('logprobs') or {}).get('content') or []]
    usage = row['usage']
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0
    result = dict(nonempty=bool(row['text'].strip()),
                  token_cap=0 < usage.get('completion_tokens', 0) <= cap,
                  logprobs_visible=bool(probs) == logprobs,
                  logprobs_finite=all(math.isfinite(p['logprob']) and p['logprob'] <= .001 for p in probs),
                  stop_not_leaked='END_MARKER' not in row['text'],
                  finished=any(c.get('finish_reason') for chunk in row['chunks'] for c in chunk.get('choices', [])))
    if full_replay:
        prompt = usage.get('prompt_tokens', -1)
        boundary = prompt - snapshot_backoff_tokens if prompt > snapshot_backoff_tokens else prompt
        key = 'prompt_boundary_replay' if snapshot_backoff_tokens else 'full_prompt_replay'
        allowed = {boundary, prompt} if allow_endpoint_promotion else {boundary}
        result[key] = cached in allowed and cached > 0
    if identity is not None:
        result['identity'] = set(re.findall(r'OWNER_\d{2}_ISOLATED', row['text'])) == {identity}
    return result


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--lifecycle-helper', type=Path, required=True)
    parser.add_argument('--long-fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot-backoff-tokens', type=int, default=0,
                        help='Explicit earlier-boundary qualification; must match frozen launch (default 0).')
    parser.add_argument('--allow-endpoint-promotion', action='store_true',
                        help='Require explicit on-miss policy; accept earlier or promoted full boundary.')
    args = parser.parse_args()
    frozen = json.loads(args.launch.read_text())
    argv = frozen['argv']
    assert 0 <= args.snapshot_backoff_tokens <= 256
    backoff_settings = [s.split('=', 1)[1] for s in argv if s.startswith('AFM_QWEN_MTP_REPLAY_BACKOFF=')]
    assert len(backoff_settings) <= 1
    assert (int(backoff_settings[0]) if backoff_settings else 0) == args.snapshot_backoff_tokens
    promotion = [s.split('=', 1)[1] for s in argv if s.startswith('AFM_QWEN_MTP_REPLAY_BACKOFF_ON_MISS=')]
    assert len(promotion) <= 1
    assert (promotion == ['1']) == args.allow_endpoint_promotion
    assert not args.allow_endpoint_promotion or args.snapshot_backoff_tokens > 0
    binary, = [Path(s) for s in argv if Path(s).name == 'afm']
    assert '--mtp' in argv and '--enable-prefix-caching' in argv
    assert argv[argv.index('--concurrent') + 1] == '15'
    b = load('lifecycle_helper', args.lifecycle_helper)
    assert b.sha(binary) == frozen['binary_sha256'], 'Frozen binary changed'
    b.MODEL = Path(argv[argv.index('-m') + 1])
    b.PORT = int(argv[argv.index('--port') + 1])
    b.ROOT = args.output.resolve()
    b.ROOT.mkdir(parents=True, exist_ok=False)
    system = json.loads(args.long_fixture.read_text())['fixtures'][0]['messages'][0]['content']
    removed = [k for k in os.environ if k.startswith(('QWEN4_', 'VMLX_'))]
    for key in removed:
        del os.environ[key]
    busy_names = {'afm', 'mlx-serve', 'xctest', 'swift-build', 'swift-frontend', 'metal', 'metallib', 'clang', 'ld'}

    def competing(owner=None):
        return [p.info for p in psutil.process_iter(['pid', 'name', 'status'])
                if p.info['pid'] != owner and p.info['name'] in busy_names
                and p.info['status'] not in (psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE)]

    assert not competing(), 'Competing inference/build; leaving it alone'
    save(b.ROOT / 'plan.json', dict(argv=argv, binary_sha256=b.sha(binary), model=str(b.MODEL),
         runner_sha256=b.sha(Path(__file__)), helper_sha256=b.sha(args.lifecycle_helper),
         fixture_sha256=b.sha(args.long_fixture), removed_environment_names=removed,
         snapshot_backoff_tokens=args.snapshot_backoff_tokens,
         allow_endpoint_promotion=args.allow_endpoint_promotion,
         checkpoint_metadata={name: b.sha(b.MODEL / name) for name in
                              ('config.json', 'chat_template.jinja', 'model.safetensors.index.json')},
         scope='Mixed MTP/AR cancellation, long complete-state replay, C15 identity and slot reuse. '
               'No real model switch or reference parity claimed.'))

    def tagged_save(path, value):
        if path.name == 'launch.json':
            value['launcher_sha256'] = value['binary_sha256']
            value['binary_sha256'] = frozen['binary_sha256']
            value['request'] = 'Per-request JSON artifacts are authoritative'
        save(path, value)

    b.save = tagged_save
    b.command = lambda *_: argv

    def workload(client, out, *unused):
        owner = json.loads((out / 'process.json').read_text())['pid']
        stop, unsafe = threading.Event(), threading.Event()
        samples = []

        def watch():
            while not stop.wait(1):
                available = psutil.virtual_memory().available
                busy = competing(owner)
                try:
                    rss = psutil.Process(owner).memory_info().rss
                except psutil.NoSuchProcess:
                    rss = None
                samples.append(dict(time=time.time(), available=available, rss=rss, competing=busy))
                if available < 100 * 1024**3 or busy or rss is None:
                    unsafe.set()
                    try:
                        os.kill(owner, signal.SIGINT)
                    except ProcessLookupError:
                        pass
                    return

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()

        def one(i, phase, isolation=False):
            assert not unsafe.is_set(), 'Resource guard'
            expected_mtp = isolation or i in (0, 1, 4)
            cap = 96 if isolation else [8, 24, 32, 16, 64, 16][i]
            cancel = phase == 'cancel' and (i in (0, 5, 10) if isolation else i == 4)
            identity = f'OWNER_{i:02d}_ISOLATED' if isolation else None
            if isolation:
                messages = [dict(role='system', content='Follow the user literally. No reasoning or markdown.'),
                            dict(role='user', content=f'Print {identity} on the first line. Then list integers 1 through 250, '
                                 'each on its own line. Do not print any other OWNER identifier.')]
            else:
                messages = [dict(role='system', content=system), dict(role='user', content=
                    f'Isolated request SAFE-{i:02d}. In cache.swift the cache key omits tenant. '
                    'Propose the smallest fix and an isolation regression test.')]
            extra = {} if isolation else ({'presence_penalty': .1} if i == 3 else
                                         {'stop': ['END_MARKER']} if i == 5 else {})
            params = dict(model=str(b.MODEL), messages=messages, max_tokens=cap,
                          temperature=.6 if not isolation and i == 1 else 0,
                          top_p=.95 if not isolation and i == 1 else 1, seed=42+i,
                          logprobs=not isolation and i == 2, stream=True,
                          stream_options={'include_usage': True},
                          extra_body={'chat_template_kwargs': {'enable_thinking': False}}, **extra)
            row = dict(index=i, phase=phase, request=params, expected_mtp=expected_mtp,
                       chunks=[], text='', usage={}, cancelled=False, started_monotonic=time.monotonic())
            try:
                visible = 0
                with client.with_options(timeout=300).chat.completions.create(**params) as response:
                    for chunk in response:
                        assert not unsafe.is_set(), 'Resource guard'
                        row['chunks'].append(chunk.model_dump(mode='json'))
                        if chunk.usage:
                            row['usage'] = chunk.usage.model_dump()
                        if chunk.choices:
                            part = chunk.choices[0].delta.content or ''
                            row['text'] += part
                            visible += bool(part)
                            if cancel and visible >= 2:
                                row['cancelled'] = True
                                break
                row['checks'] = {'closed_early': row['cancelled']} if cancel else checks(
                    row, cap, params['logprobs'], expected_mtp and phase in ('after-cancel', 'repeat'),
                    identity, args.snapshot_backoff_tokens, args.allow_endpoint_promotion)
                if not isolation and not cancel:
                    row['checks']['long_prompt'] = 4096 < row['usage'].get('prompt_tokens', 0) <= 8192
            except Exception as error:
                row.update(error=repr(error), checks={'transport': False})
            row['seconds'] = time.monotonic() - row['started_monotonic']
            save(out / f'{"identity" if isolation else "mixed"}-{phase}-{i:02d}.json', row)
            return row

        totals = []

        def record(rows, label):
            result = dict(phase=label, requests=len(rows), assertions=sum(len(r['checks']) for r in rows),
                          passed=sum(sum(r['checks'].values()) for r in rows),
                          failures=[{'index': r['index'], 'checks': r['checks']} for r in rows
                                    if not all(r['checks'].values())])
            totals.append(result)
            save(out / f'{label}-summary.json', result)
            print(json.dumps(result), flush=True)
            if result['failures']:
                raise RuntimeError('Lifecycle failure; saved evidence, no promotion')

        try:
            record([one(0, 'warmup')], 'mixed-warmup')
            for phase in ('cancel', 'after-cancel', 'repeat'):
                with ThreadPoolExecutor(max_workers=6) as pool:
                    record(list(pool.map(lambda i: one(i, phase), range(6))), f'mixed-{phase}')
            # Establish isolated answers before cancellation/concurrency attribution.
            record([one(i, 'isolated', True) for i in range(15)], 'identity-isolated')
            for phase in ('cancel', 'after-cancel', 'repeat'):
                with ThreadPoolExecutor(max_workers=15) as pool:
                    record(list(pool.map(lambda i: one(i, phase, True), range(15))), f'identity-{phase}')
        finally:
            stop.set()
            watcher.join(timeout=3)
            save(out / 'resources.json', samples)
            save(out / 'safety-summary.json', totals)
        assert not unsafe.is_set(), 'Resource guard fired'

    b.request = workload
    b.run_arm('afm', True, 'retained', True)
    out = b.ROOT / 'retained-afm-mtp-1'
    log = (out / 'server.log').read_text()
    groups = re.findall(r'Qwen MTP shared verification: batches=(\d+) \| rows=(\d+)', log)
    shared = sum(int(pair[0]) for pair in groups)
    exit_code = json.loads((out / 'exit.json').read_text())['exit_code']
    save(b.ROOT / 'coverage.json', dict(shared_batches=shared, clean_exit=exit_code == 0,
                                     server_shutdown_exit_code=exit_code))
    assert shared > 0 and exit_code == 0, 'Shared verification or clean shutdown coverage missing'


if __name__ == '__main__':
    main()
