"""HW3 development evaluation: 10,404 warm users, cached HW2 texts, frozen keyword rule."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import data_adapter
import encoder as enc


FEATURE_FIELDS = ['assetCategory', 'assetSubCategory', 'sector', 'industry', 'marketID']
MISSING = ('', 'Not_Available', 'nan')
OLD_PROTOCOL_HASH = '50efd4aeb415ba6f0c66dd0793a7c2033cdc32a42b86b58eebbca3e81eaf3417'


def digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(
        value, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def matches(name, words):
    low = name.casefold()
    flat = ' '.join(low.split())
    for word in words:
        token = word.casefold()
        if ' ' in token or '-' in token:
            if token in flat:
                return True
        elif re.search(r'\b' + re.escape(token) + r'\b', low):
            return True
    return False


def rule_suffix(rule, source_json):
    facts = json.loads(source_json)
    name = facts.get('name', '')
    found = []
    for entry in rule['sector_industry_words']:
        if matches(name, entry['words']):
            for field in ('sector', 'industry'):
                if str(facts.get(field, '')).strip() in MISSING or str(facts.get(field, '')).strip() == 'unknown':
                    found.append((field, entry[field]))
            break
    for entry in rule['region_words']:
        if matches(name, entry['words']):
            found.append(('region', entry['region']))
            break
    if not found:
        return ''
    return ' ' + '; '.join(f'{field}: {value}' for field, value in found) + '.'


def rank_from(counts, vectors, pop, eligible):
    scores = enc.history_scores(counts, vectors)
    if not any(scores.values()):
        scores = pop
    return sorted(eligible, key=lambda isin: (-scores.get(isin, 0.0), isin))[:10]


def group_users(users, assets, history):
    return [uid for uid in users if any(
        str(assets[isin].get(field, '')).strip() in MISSING
        for isin in history.get(uid, {}) for field in ('sector', 'industry'))]


def make_protocol(args, manifest, assets, users):
    return {
        'teaching_defaults_not_official_grading_protocol': True,
        'cutoff': args.cutoff,
        'end_exclusive': manifest['end_exclusive'],
        'users': users,
        'cohort_selection': args.cohort_selection,
        'eligible_assets': sorted(assets),
        'exclude_seen': False,
        'candidate_count': args.candidate_count,
        'k': 10,
        'empty_truth_policy': 'zero in primary mean; nonempty mean also reported',
        'initial_positions': 'excluded',
        'duplicate_policy': 'exact full rows only',
        'profile_use': 'display only in classical baselines',
        'asset_feature_fields': FEATURE_FIELDS,
        'source_sha256': manifest['data_sha256'],
    }


def load_cached(manifest):
    path = Path(manifest['_log_path'])
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    counts = Counter(r['prompt_mode'] for r in rows)
    if len(rows) != 1534 or counts != {'para': 767, 'infer': 767}:
        raise ValueError(f'Cached generation log unexpected: {len(rows)}, {dict(counts)}')
    para = {r['asset_id']: r['response']['text'].strip() for r in rows if r['prompt_mode'] == 'para'}
    infer = {r['asset_id']: r['response']['text'].strip() for r in rows if r['prompt_mode'] == 'infer'}
    if len(para) != 767 or len(infer) != 767:
        raise ValueError('Duplicate asset IDs in cached responses')
    return rows, para, infer


def coverage(rule, documents, para, infer):
    values = {field: set() for field in ('sector', 'industry', 'region')}
    for entry in rule['sector_industry_words']:
        values['sector'].add(entry['sector'])
        values['industry'].add(entry['industry'])
    for entry in rule['region_words']:
        values['region'].add(entry['region'])
    rule_filled, llm_approx = {}, {}
    for isin, text in para.items():
        suffix = rule_suffix(rule, documents[isin])
        rule_filled[isin] = set(re.findall(r'(sector|industry|region): ([^;]+)', suffix))
        low = infer[isin].casefold()
        llm_approx[isin] = {field for field, words in values.items()
                            if any(word.casefold() in low for word in words)}
    by_field = {}
    for field in ('sector', 'industry', 'region'):
        by_field[field] = {
            'rule_assets': sum(1 for v in rule_filled.values() if any(f == field for f, _ in v)),
            'llm_approx_assets': sum(1 for v in llm_approx.values() if field in v),
        }
    return {
        'note': 'LLM counts are a lexical approximation from the frozen rule value list; accuracy is checked manually on 25 assets.',
        'assets_total': len(para),
        'rule_any_field_assets': sum(1 for v in rule_filled.values() if v),
        'llm_approx_any_field_assets': sum(1 for v in llm_approx.values() if v),
        'by_field': by_field,
        'rule_assignments': {isin: sorted(f'{f}={v}' for f, v in val)
                             for isin, val in sorted(rule_filled.items()) if val},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(r'C:\Users\555\projects\hw2\colab_upload'))
    parser.add_argument('--starter-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--cache-log', type=Path, default=Path(r'C:\Users\555\projects\hw2\outputs\full_v1\generation_log.jsonl'))
    parser.add_argument('--manifest', type=Path, default=Path(r'C:\Users\555\projects\hw2\outputs\full_v1\run_manifest.json'))
    parser.add_argument('--rule', type=Path, default=Path(__file__).resolve().parent / 'keyword_rule.json')
    parser.add_argument('--output-dir', type=Path, default=Path(r'C:\Users\555\projects\hw3\outputs\dev_all_warm'))
    parser.add_argument('--cutoff', default='2021-06-01')
    parser.add_argument('--horizon-days', type=int, default=30)
    parser.add_argument('--max-users', type=int, default=20000)
    parser.add_argument('--cohort-selection', choices=['warm', 'warm_all'], default='warm_all')
    parser.add_argument('--candidate-count', type=int, default=50)
    args = parser.parse_args()

    started = time.perf_counter()
    args.model = 'Qwen/Qwen3-4B-Instruct-2507'
    args.revision = 'cdbee75f17c01a7cc42f958dc650907174af0554'
    args.device = 'cpu'
    args.allow_download = False
    args.on_invalid = 'fallback'
    rule = json.loads(args.rule.read_text(encoding='utf-8'))
    rule_hash = digest(args.rule.read_bytes())
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    manifest['_log_path'] = str(args.cache_log)

    source_hashes = {name: digest((args.data_dir / name).read_bytes()) for name in
                     ('transactions.csv', 'asset_information.csv',
                      'customer_information.csv', 'close_prices.csv')}
    if source_hashes != manifest['protocol']['source_sha256']:
        raise ValueError('FAR-Trans CSV hashes differ from HW2 manifest')
    code_hashes = {
        'encoder_sha256': digest(Path(enc.__file__).read_bytes()),
        'runtime_sha256': digest(Path(runtime_file()).read_bytes()),
    }
    expected_code = {k: manifest['settings_fingerprint'][k] for k in code_hashes}
    if code_hashes != expected_code:
        raise ValueError('Code hashes differ from HW2 manifest')
    rows, para, infer = load_cached(manifest)
    prompt_fp = {mode: digest(sorted(r['prompt'] for r in rows if r['prompt_mode'] == mode))
                 for mode in ('para', 'infer')}
    if prompt_fp != manifest['prompt_fingerprints']:
        raise ValueError('Prompt fingerprints differ from HW2 manifest')
    if {r['runtime']['model'] for r in rows} != {manifest['model']}:
        raise ValueError('Cached model differs from manifest')
    if {r['runtime']['resolved_revision'] for r in rows} != {manifest['revision']}:
        raise ValueError('Cached revision differs from manifest')
    print('integrity OK; source/code/prompt/model/revision match HW2')
    print('rule_hash', rule_hash)

    load_started = time.perf_counter()
    bundle = data_adapter.load(args)
    starter, assets, history, truth, contexts, baseline, events, pop, base_manifest = bundle
    users = base_manifest['users']
    load_seconds = time.perf_counter() - load_started
    print('warm users', len(users))
    if len(users) < 100:
        raise ValueError('Unexpectedly small warm cohort')
    eligible = sorted({i for c in contexts.values() for i in c['eligible_asset_ids']})
    documents = {i: enc.source_text(assets[i]) for i in eligible}
    if set(documents) != set(para):
        raise ValueError('Eligible assets differ from HW2 cache')

    feature_started = time.perf_counter()
    rule_text = {isin: para[isin] + rule_suffix(rule, documents[isin]) for isin in eligible}
    original = enc.tfidf(documents)
    views = {
        'p_para': enc.combine_views(original, enc.tfidf(para)),
        'p_infer': enc.combine_views(original, enc.tfidf(infer)),
        'p_rule': enc.combine_views(original, enc.tfidf(rule_text)),
    }
    feature_seconds = time.perf_counter() - feature_started

    protocol = make_protocol(args, base_manifest, assets, users)
    protocol_hash = digest(protocol)
    frozen = Path(r'C:\Users\555\projects\hw3\outputs\protocol\protocol_hash.txt').read_text(encoding='utf-8').strip()
    if args.cohort_selection == 'warm_all' and protocol_hash != frozen:
        raise ValueError(f'Protocol hash {protocol_hash} differs from frozen {frozen}')
    if protocol_hash == OLD_PROTOCOL_HASH and args.cohort_selection != 'warm':
        raise ValueError('Protocol hash unexpectedly equals demo-100 hash')

    group = group_users(users, assets, history)
    group_set = set(group)
    scoring_started = time.perf_counter()
    per_user, predictions, unchanged = [], [], []
    for uid in users:
        counts = history.get(uid, {})
        allowed = contexts[uid]['eligible_asset_ids']
        relevant = truth.get(uid, set()) & set(allowed)
        methods = dict(baseline[uid])
        methods['original_text_cbf'] = rank_from(counts, original, pop, allowed)
        for name, vectors in views.items():
            methods[name] = rank_from(counts, vectors, pop, allowed)
        for name, ranking in methods.items():
            starter.validate_ranking(ranking, allowed)
            per_user.append({'method': name, 'user_id': uid,
                             'relevant_count': len(relevant),
                             'ndcg_at_10': starter.ndcg10(ranking, relevant),
                             'in_missing_field_group': uid in group_set,
                             'has_future_purchase': bool(relevant)})
            predictions.append((name, uid, ranking))
        if methods['p_para'] == methods['p_infer'] or methods['p_para'] == methods['p_rule']:
            unchanged.append({'user_id': uid, 'relevant_count': len(relevant),
                              'p_para_equals_p_infer': methods['p_para'] == methods['p_infer'],
                              'p_para_equals_p_rule': methods['p_para'] == methods['p_rule'],
                              'p_infer_equals_p_rule': methods['p_infer'] == methods['p_rule'],
                              'p_para_top10': methods['p_para'],
                              'p_infer_top10': methods['p_infer'],
                              'p_rule_top10': methods['p_rule']})
    scoring_seconds = time.perf_counter() - scoring_started

    methods = {}
    for name in sorted({r['method'] for r in per_user}):
        subset = [r for r in per_user if r['method'] == name]
        methods[name] = {
            'all': starter.aggregate(subset),
            'nonempty': starter.aggregate([r for r in subset if r['has_future_purchase']]),
            'group': starter.aggregate([r for r in subset if r['in_missing_field_group']]),
        }
        methods[name]['ndcg_at_10'] = methods[name]['all']['ndcg_at_10']

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'per_user.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_user[0]))
        writer.writeheader()
        writer.writerows(per_user)
    with (args.output_dir / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'user_id', 'rank', 'ISIN'])
        for name, uid, ranking in predictions:
            writer.writerows((name, uid, rank, isin) for rank, isin in enumerate(ranking, 1))
    with (args.output_dir / 'ndcg_table.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'users_all', 'ndcg_all', 'nonempty_users',
                         'ndcg_nonempty', 'group_users', 'ndcg_group'])
        for name, row in sorted(methods.items()):
            writer.writerow([name, row['all']['users'], row['all']['ndcg_at_10'],
                             row['nonempty']['users'], row['nonempty']['ndcg_at_10'],
                             row['group']['users'], row['group']['ndcg_at_10']])
    with (args.output_dir / 'unchanged_cases.jsonl').open('w', encoding='utf-8') as stream:
        for record in unchanged:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')

    metrics = {'protocol_hash': protocol_hash, 'methods': methods,
               'fallback_counts': {},
               'note': 'Development results on all warm users; offline purchase relevance.'}
    cov = coverage(rule, documents, para, infer)
    manifest_out = {
        'protocol_hash': protocol_hash,
        'hw2_protocol_hash': OLD_PROTOCOL_HASH,
        'protocol': {k: v for k, v in protocol.items() if k != 'users'},
        'users_count': len(users),
        'group_users_count': len(group),
        'group_definition': 'at least one pre-cutoff history asset with empty sector or industry',
        'rule_path': str(args.rule), 'rule_sha256': rule_hash,
        'cache_log': str(args.cache_log),
        'cached_generation_records': len(rows),
        'cached_model': manifest['model'], 'cached_revision': manifest['revision'],
        'source_sha256': source_hashes, 'code_sha256': code_hashes,
        'prompt_fingerprints': prompt_fp,
        'timings_seconds': {'load': load_seconds, 'feature': feature_seconds,
                            'scoring': scoring_seconds, 'total': time.perf_counter() - started},
        'note': 'HW2 cached responses reused unchanged; no model inference in this evaluation.',
    }
    for name, value in (('metrics.json', metrics), ('run_manifest.json', manifest_out),
                        ('coverage.json', cov)):
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('protocol_hash', protocol_hash, 'hw2', OLD_PROTOCOL_HASH)
    print('users', len(users), 'group', len(group))
    for name, row in sorted(methods.items()):
        print('%s all=%.4f nonempty=%.4f group=%.4f buyers=%d' % (
            name, row['all']['ndcg_at_10'] or 0.0,
            row['nonempty']['ndcg_at_10_nonempty'] or 0.0,
            row['group']['ndcg_at_10'] or 0.0, row['nonempty']['users']))
    print('unchanged cases', len(unchanged))


def runtime_file():
    import runtime
    return runtime.__file__


if __name__ == '__main__':
    main()