"""Compare 48 new Colab answers with the immutable HW2 cache; never replace cached texts."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

HW2 = Path(r'C:\Users\555\projects\hw2\outputs\full_v1')
ROOT = Path(r'C:\Users\555\projects\hw3')
STOP_AT = 5


def digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def load_answers(path):
    text = path.read_text(encoding='utf-8').strip()
    if path.suffix == '.jsonl':
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
        if len(lines) == 1 and 'records' in lines[0]:
            return lines[0]
        return {'records': lines}
    return json.loads(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--answers', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'cache_check')
    a = p.parse_args()
    payload = json.loads((ROOT / 'colab_upload' / 'verify_assets.json').read_text(encoding='utf-8'))
    manifest = json.loads((HW2 / 'run_manifest.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (HW2 / 'generation_log.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    cache = {(r['asset_id'], r['prompt_mode']): r['response']['text'].strip()
             for r in rows if r['prompt_mode'] in ('para', 'infer')}
    expected = {(item['asset_id'], mode): item['prompts'][mode]
                for item in payload['assets'] for mode in ('para', 'infer')}
    answers = load_answers(a.answers)
    records = answers['records']
    got = {}
    problems = []
    for r in records:
        key = (r['asset_id'], r['prompt_mode'])
        if key in got:
            problems.append(f'duplicate {key}')
        got[key] = r['response']['text'].strip()
        prompt_hash = hashlib.sha256(expected[key].encode('utf-8')).hexdigest() if key in expected else None
        if key not in expected:
            problems.append(f'unexpected asset/mode {key}')
        elif r.get('prompt_sha256') != prompt_hash:
            problems.append(f'prompt hash differs {key}')
    missing = sorted(set(expected) - set(got))
    if missing:
        problems.append(f'missing {len(missing)} answers: {missing[:5]}')
    runtime = answers.get('runtime', {})
    for field, want in (('model', manifest['model']), ('resolved_revision', manifest['revision']),
                        ('device', 'cuda'), ('dtype', 'torch.bfloat16'),
                        ('do_sample', False), ('thinking', False), ('context_limit', 8192)):
        if runtime.get(field) != want:
            problems.append(f'runtime {field}={runtime.get(field)!r} expected {want!r}')
    mismatches = [{'asset_id': k[0], 'prompt_mode': k[1],
                   'cache': cache[k], 'new': got[k]} for k in sorted(expected)
                  if k in got and got[k] != cache[k]]
    report = {
        'expected_answers': len(expected), 'received_answers': len(records),
        'exact_matches': len(expected) - len(missing) - len(mismatches),
        'mismatches': len(mismatches), 'protocol_problems': problems,
        'stop_threshold': STOP_AT,
        'stop': len(mismatches) >= STOP_AT or bool(problems),
        'new_timing_seconds': {'model_load': answers.get('load_seconds'),
                               'generation': answers.get('generation_seconds')},
        'hw2_generation_seconds_from_manifest': manifest.get('timings_seconds', {}).get('inference_seconds'),
        'note': 'Timing of the short check is separate from HW2 generation time; cached HW2 texts remain unchanged in the evaluation.',
    }
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (a.output_dir / 'mismatches.jsonl').open('w', encoding='utf-8') as s:
        for m in mismatches:
            s.write(json.dumps(m, ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if mismatches:
        print('first mismatch:', json.dumps(mismatches[0], ensure_ascii=False))
    if report['stop']:
        print('STOP: show mismatches and ask the student before using any new answer')


if __name__ == '__main__':
    main()