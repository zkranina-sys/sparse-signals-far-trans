"""Failure analysis of cached HW2 descriptions; prepares 25 assets for manual labels."""
from __future__ import annotations
import argparse, hashlib, json, random, re
from collections import Counter
from pathlib import Path

ROOT = Path(r'C:\Users\555\projects\hw3')
HW2 = Path(r'C:\Users\555\projects\hw2')


def is_echo(text, source):
    return source.strip() and source.strip() in text.replace('\n', ' ')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-log', type=Path, default=HW2/'outputs'/'full_v1'/'generation_log.jsonl')
    p.add_argument('--coverage', type=Path, default=ROOT/'outputs'/'dev_all_warm'/'coverage.json')
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs'/'error_analysis')
    p.add_argument('--seed', type=int, default=20261127)
    a = p.parse_args()
    rows = [json.loads(line) for line in a.cache_log.read_text(encoding='utf-8').splitlines() if line.strip()]
    invalid, echo, unchanged = [], [], []
    by_mode = Counter(r['prompt_mode'] for r in rows)
    for r in rows:
        text = r['response']['text'].strip()
        source = r['source'].strip()
        if not text or len(text.split()) > 40:
            invalid.append({'asset_id': r['asset_id'], 'prompt_mode': r['prompt_mode'],
                            'words': len(text.split()), 'text': text})
        if is_echo(text, source):
            echo.append({'asset_id': r['asset_id'], 'prompt_mode': r['prompt_mode'], 'text': text})
    para = {r['asset_id']: r['response']['text'].strip() for r in rows if r['prompt_mode'] == 'para'}
    infer = {r['asset_id']: r['response']['text'].strip() for r in rows if r['prompt_mode'] == 'infer'}
    source = {r['asset_id']: r['source'] for r in rows if r['prompt_mode'] == 'para'}
    for i in sorted(para):
        if infer.get(i) == para[i]:
            unchanged.append({'asset_id': i, 'text': para[i]})
    cov = json.loads(a.coverage.read_text(encoding='utf-8')) if a.coverage.exists() else {}
    assignments = cov.get('rule_assignments', {})
    random.seed(a.seed)
    candidates = sorted(i for i in source if any(
        word in infer[i].casefold() for word in ('sector', 'industry', 'region')))
    sample = sorted(random.sample(candidates, min(25, len(candidates))))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with (a.output_dir/'manual_label_25.jsonl').open('w', encoding='utf-8') as s:
        for i in sample:
            s.write(json.dumps({'asset_id': i, 'source': source[i], 'para': para[i],
                                'infer': infer[i], 'rule_fields': assignments.get(i, []),
                                'label': '', 'student_note': ''}, ensure_ascii=False) + '\n')
    report = {
        'cache_records': len(rows), 'by_mode': dict(by_mode),
        'invalid_outputs': invalid, 'invalid_count': len(invalid),
        'json_echoes': echo, 'json_echo_count': len(echo),
        'repairs_or_fallbacks': [],
        'infer_equals_para_count': len(unchanged),
        'unchanged_assets': unchanged,
        'coverage': {k: v for k, v in cov.items() if k != 'rule_assignments'},
        'manual_sample_seed': a.seed,
        'manual_sample_ids': sample,
        'manual_sample_file': str(a.output_dir/'manual_label_25.jsonl'),
        'note': 'Labels left blank for the student; no accuracy claim is made by this script.',
    }
    (a.output_dir/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('records', len(rows), 'invalid', len(invalid), 'echo', len(echo),
          'unchanged', len(unchanged), 'manual sample', len(sample))


if __name__ == '__main__':
    main()