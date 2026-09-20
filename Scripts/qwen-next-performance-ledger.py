#!/usr/bin/env python3
"""Index immutable context evidence; keep speed peaks separate from quality claims.

Reads immediate run directories only, never starts inference. Outputs to a new
directory. Supply the frozen evidence roots and the current-a root separately.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

METRICS = ('client_prefill_tps', 'client_decode_tps')
ACTUAL_HASHES = ('actual_inference_binary_sha256', 'measured_binary_sha256', 'binary_sha256')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def configuration(launch):
    argv = launch['argv']
    env = sorted(a for a in argv if '=' in a and a.startswith(('AFM_', 'MLX_', 'MACAFM_')))
    def flag(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default
    model = flag('-m', flag('--model'))
    # Execution ports/binary locations do not alter the model experiment.
    ignored = {'--port', '-m', '--model'}
    flags, skip = [], False
    executable = next(i for i, a in enumerate(argv) if Path(a).name == 'afm')
    for a in argv[executable + 1:]:
        if skip:
            skip = False
        elif a in ignored:
            skip = True
        elif a == 'mlx':
            continue
        else:
            flags.append(a)
    request = launch.get('request', {})
    sampling = {k: request.get(k, 'unspecified') for k in ('temperature', 'top_p', 'top_k', 'seed')}
    base = {'model': model, 'mtp': '--mtp' in argv, 'sampling': sampling,
            'concurrency': flag('--concurrent', 'default'),
            'prefix_cache': '--enable-prefix-caching' in argv,
            'thinking_disabled': '--no-think' in argv}
    trace = any(x.startswith(('AFM_DEBUG=', 'AFM_PERF=', 'AFM_QWEN_PROFILE_'))
                and x.rsplit('=', 1)[-1] not in ('0', '') for x in env)
    return base, {'flags': flags, 'environment': env}, trace


def index_run(run):
    launch_path = run / 'launch.json'
    if not launch_path.exists():
        return [], []
    launch = json.loads(launch_path.read_text())
    if not any(Path(x).name == 'afm' for x in launch.get('argv', [])):
        return [], []
    base, settings, trace = configuration(launch)
    binary_sha = next((launch.get(k) for k in ACTUAL_HASHES if launch.get(k)), None)
    # Older env launchers accidentally recorded /usr/bin/env, not afm.
    if binary_sha == '9e3f338ac5a436322acb4821d6023faa48df0a54f12e23f806c59316226dec45':
        binary_sha = None
    groups, exclusions = {}, []
    for path in sorted(run.glob('trial-*.json')):
        data = json.loads(path.read_text())
        r = data.get('result', {})
        usage = r.get('usage', {})
        reasons = []
        if trace: reasons.append('instrumented launch')
        if r.get('warmup', False): reasons.append('warmup')
        if usage.get('completion_tokens') != 128: reasons.append('not the 128-token context workload')
        if r.get('cached_tokens') not in (0, None): reasons.append('prefix reuse')
        if not r.get('generated_text', '').strip() or r.get('reasoning_text'): reasons.append('empty or thinking output')
        if r.get('finish_reasons') != ['length']: reasons.append('unexpected finish reason')
        if not r.get('prompt_sha256') or not usage.get('prompt_tokens'): reasons.append('missing prompt identity')
        if not all(isinstance(r.get(m), (float, int)) and math.isfinite(r[m]) and r[m] > 0 for m in METRICS):
            reasons.append('invalid/missing timing')
        if not (run / 'complete.json').exists(): reasons.append('incomplete run')
        if not binary_sha: reasons.append('missing actual inference binary hash')
        if reasons:
            exclusions.append({'file': str(path), 'reasons': reasons})
            continue
        identity = {**base, 'prompt_sha256': r['prompt_sha256'],
                    'prompt_tokens': usage['prompt_tokens'], 'output_tokens': usage['completion_tokens']}
        key = digest(identity)
        entry = groups.setdefault(key, {'identity': identity, 'samples': [], 'context': r['context']})
        entry['samples'].append({'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'text_sha256': hashlib.sha256(r['generated_text'].encode()).hexdigest(),
                                 'trial': r['trial'], **{m: r[m] for m in METRICS}})
    cells = []
    for entry in groups.values():
        samples = entry['samples']
        # Default concurrency was historically C1, but preserve that omission
        # in the strict key. Family key compares uncached single-request runs;
        # no batch/aggregate metric is ever admitted here.
        family = dict(entry['identity'])
        if family['concurrency'] == 'default': family['concurrency'] = '1'
        cells.append({**entry, 'run': str(run), 'time': launch.get('time'), 'binary_sha256': binary_sha,
                      'settings': settings, 'family_key': digest(family),
                      'strict_launch_key': digest([entry['identity'], settings]),
                      'n': len(samples), 'quality': 'Saved nonempty responses; broad quality parity NOT established',
                      'metrics': {m: {'median': statistics.median(s[m] for s in samples),
                                      'min': min(s[m] for s in samples), 'max': max(s[m] for s in samples)}
                                  for m in METRICS}})
    return cells, exclusions


def compare(cells, current_root):
    rows = []
    current_root = str(current_root.resolve())
    history = [c for c in cells if not Path(c['run']).is_relative_to(current_root)]
    current = [c for c in cells if Path(c['run']).is_relative_to(current_root)]
    for c in current:
        for metric in METRICS:
            family = [h for h in history if h['family_key'] == c['family_key'] and h['n'] >= 3]
            strict = [h for h in family if h['strict_launch_key'] == c['strict_launch_key']]
            row = {'current_run': c['run'], 'context': c['context'], 'mtp': c['identity']['mtp'],
                   'metric': metric, 'current_median': c['metrics'][metric]['median'], 'n': c['n']}
            for name, candidates in [('historical_configuration_envelope', family), ('same_explicit_launch', strict)]:
                if candidates:
                    best = max(candidates, key=lambda h: h['metrics'][metric]['median'])
                    peak = best['metrics'][metric]['median']
                    row[name] = {'best_median': peak, 'run': best['run'], 'binary_sha256': best['binary_sha256'],
                                 'settings': best['settings'], 'n': best['n'],
                                 'change_percent': 100 * (row['current_median'] / peak - 1),
                                 'fastest_individual': max(h['metrics'][metric]['max'] for h in candidates)}
                else:
                    row[name] = None
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, action='append', required=True)
    parser.add_argument('--current-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cells, excluded, seen = [], [], set()
    for root in [*args.evidence_root, args.current_root]:
        for run in sorted(root.iterdir()):
            if not run.is_dir() or run.resolve() in seen: continue
            seen.add(run.resolve())
            entries, rejects = index_run(run)
            cells.extend(entries); excluded.extend(rejects)
    comparisons = compare(cells, args.current_root)
    if not comparisons: raise RuntimeError('No completed current context runs; preserve the evidence and investigate')
    args.output.mkdir(parents=True, exist_ok=False)
    payload = {'schema_version': 1, 'roots': [str(p) for p in args.evidence_root],
               'current_root': str(args.current_root), 'cells': cells, 'excluded': excluded, 'comparisons': comparisons}
    (args.output / 'ledger.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = ['# Qwen Next performance ledger', '',
             'No peak is a quality certificate. Saved outputs and the separate numerical/agentic audit remain authoritative.', '',
             'Client prefill is prompt tokens / (TTFT − measured endpoint latency): a proxy including first-token/API work, not isolated device prefill. Decode is (output tokens − 1) / the first-to-last text window. Neither is concurrent aggregate throughput.', '',
             'Historical envelope: same checkpoint path, prompt hash/token length, 128 outputs, sampling, MTP on/off, and cache/concurrency class. Opt-in settings and MTP depth may differ. It is an observed performance gap, NOT attribution to a code regression. Same explicit launch is reported independently; changed implicit defaults may still require investigation.', '',
             'Peaks below are the best three-or-more-trial **run median**. Individual maxima are retained in ledger.json, never substituted for repeated-run results. Warmups, traced runs, incomplete runs, missing binary identities and incompatible output lengths are excluded with reasons.', '',
             '| MTP | Context | Metric | Current median | Historical best median | Change | Same-launch change |',
             '|---|---:|---|---:|---:|---:|---:|']
    for r in sorted(comparisons, key=lambda r: (r['mtp'], float(r['context']), r['metric'])):
        envelope, strict = r['historical_configuration_envelope'], r['same_explicit_launch']
        lines.append(f"| {'on' if r['mtp'] else 'off'} | {r['context']}K | {'prefill proxy' if r['metric'] == METRICS[0] else 'decode'} | {r['current_median']:.2f} | "
                     + (f"{envelope['best_median']:.2f} | {envelope['change_percent']:+.2f}%" if envelope else '— | —')
                     + ' | ' + (f"{strict['change_percent']:+.2f}%" if strict else 'no matched history') + ' |')
    lines += ['', f'Indexed {len(cells)} eligible run/context cells; {len(excluded)} excluded sample records. Full source paths, sample hashes, binary identities, settings, ranges, and exclusions are in ledger.json.', '', '## Peak sources', '']
    for run in sorted({r['historical_configuration_envelope']['run'] for r in comparisons if r['historical_configuration_envelope']}):
        lines.append(f'- `{run}`')
    (args.output / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
