"""POST-HOC CHECK - INVENTED AFTER SEEING RESULTS. Not part of the frozen protocol.

p_infer_missing: use p_infer text for assets with empty sector or industry,
p_para text for all other assets. Same users, cutoff, window, cache, k and
scoring rules as the frozen development evaluation. Reported separately.
"""
import csv, json, random, sys, time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(r'C:\Users\555\projects\hw3')
HW2 = Path(r'C:\Users\555\projects\hw2')
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'colab_upload'))
import data_adapter, encoder as enc, eval_dev, starter

OUT = ROOT / 'outputs' / 'posthoc_infer_missing'
BOOT_SEED = 20261126
BOOT_B = 2000


def integrity(args, rows):
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    src = {name: eval_dev.digest((args.data_dir / name).read_bytes()) for name in
           ('transactions.csv', 'asset_information.csv', 'customer_information.csv', 'close_prices.csv')}
    if src != manifest['protocol']['source_sha256']:
        raise ValueError('FAR-Trans CSV hashes differ from HW2 manifest')
    codes = {'encoder_sha256': eval_dev.digest(Path(enc.__file__).read_bytes()),
             'runtime_sha256': eval_dev.digest(Path(sys.modules['runtime'].__file__).read_bytes())}
    if codes != {k: manifest['settings_fingerprint'][k] for k in codes}:
        raise ValueError('Code hashes differ from HW2 manifest')
    fps = {mode: eval_dev.digest(sorted(r['prompt'] for r in rows if r['prompt_mode'] == mode))
           for mode in ('para', 'infer')}
    if fps != manifest['prompt_fingerprints']:
        raise ValueError('Prompt fingerprints differ from HW2 manifest')
    if {r['runtime']['model'] for r in rows} != {manifest['model']} or \
       {r['runtime']['resolved_revision'] for r in rows} != {manifest['revision']}:
        raise ValueError('Cached model or revision differs from HW2 manifest')
    return manifest, src, codes, fps


def bootstrap(diffs, seed, B):
    rng = random.Random(seed)
    n = len(diffs)
    means = [sum(diffs[i] for i in rng.choices(range(n), k=n)) / n for _ in range(B)]
    means.sort()
    return {'n': n, 'mean': sum(diffs) / n,
            'ci95': [means[int(0.025 * B)], means[int(0.975 * B)]],
            'method': 'paired resampling with replacement, random.Random(seed), B replicates, percentile 2.5/97.5',
            'seed': seed, 'replicates': B}


def main():
    started = time.perf_counter()
    args = SimpleNamespace(
        data_dir=HW2 / 'colab_upload', starter_dir=ROOT / 'code', output_dir=OUT,
        cutoff='2021-06-01', horizon_days=30, max_users=20000,
        cohort_selection='warm_all', candidate_count=50,
        model='Qwen/Qwen3-4B-Instruct-2507',
        revision='cdbee75f17c01a7cc42f958dc650907174af0554',
        device='cpu', allow_download=False, on_invalid='fallback',
        cache_log=HW2 / 'outputs' / 'full_v1' / 'generation_log.jsonl',
        manifest=HW2 / 'outputs' / 'full_v1' / 'run_manifest.json',
        rule=ROOT / 'code' / 'keyword_rule.json')
    rows, para, infer = eval_dev.load_cached({'_log_path': str(args.cache_log)})
    hw2, source_hashes, code_hashes, prompt_fp = integrity(args, rows)
    print('integrity OK', flush=True)

    t0 = time.perf_counter()
    bundle = data_adapter.load(args)
    st, assets, history, truth, contexts, baseline, events, pop, base_manifest = bundle
    users = base_manifest['users']
    load_seconds = time.perf_counter() - t0
    print('users', len(users), flush=True)

    protocol = eval_dev.make_protocol(args, base_manifest, assets, users)
    protocol_hash = eval_dev.digest(protocol)
    frozen = (ROOT / 'outputs' / 'protocol' / 'protocol_hash.txt').read_text(encoding='utf-8').strip()
    if protocol_hash != frozen:
        raise ValueError('Protocol hash differs from frozen protocol')
    print('protocol_hash', protocol_hash, flush=True)

    eligible = sorted({i for c in contexts.values() for i in c['eligible_asset_ids']})
    documents = {i: enc.source_text(assets[i]) for i in eligible}
    if set(documents) != set(para):
        raise ValueError('Eligible assets differ from HW2 cache')
    missing_assets = sorted(i for i in eligible if any(
        str(assets[i].get(f, '')).strip() in eval_dev.MISSING for f in ('sector', 'industry')))
    mixed = {i: infer[i] if i in set(missing_assets) else para[i] for i in eligible}
    print('missing_assets', len(missing_assets), 'of', len(eligible), flush=True)

    t0 = time.perf_counter()
    original = enc.tfidf(documents)
    views = {'p_para': enc.combine_views(original, enc.tfidf(para)),
             'p_infer': enc.combine_views(original, enc.tfidf(infer)),
             'p_infer_missing': enc.combine_views(original, enc.tfidf(mixed))}
    feature_seconds = time.perf_counter() - t0

    group_set = set(eval_dev.group_users(users, assets, history))
    t0 = time.perf_counter()
    per_user, preds = [], []
    for uid in users:
        counts = history.get(uid, {})
        allowed = contexts[uid]['eligible_asset_ids']
        relevant = truth.get(uid, set()) & set(allowed)
        for name, vectors in views.items():
            ranking = eval_dev.rank_from(counts, vectors, pop, allowed)
            st.validate_ranking(ranking, allowed)
            per_user.append({'method': name, 'user_id': uid, 'relevant_count': len(relevant),
                             'ndcg_at_10': st.ndcg10(ranking, relevant),
                             'in_missing_field_group': uid in group_set,
                             'has_future_purchase': bool(relevant)})
            if name == 'p_infer_missing':
                preds.append((uid, ranking))
    scoring_seconds = time.perf_counter() - t0

    old = {(r['method'], r['user_id']): float(r['ndcg_at_10']) for r in
           csv.DictReader((ROOT / 'outputs' / 'dev_all_warm' / 'per_user.csv').open(encoding='utf-8'))}
    maxerr = max(abs(old[(r['method'], r['user_id'])] - r['ndcg_at_10']) for r in per_user
                 if r['method'] in ('p_para', 'p_infer'))
    if maxerr > 1e-12:
        raise ValueError('Replicated p_para/p_infer values differ from main run: %g' % maxerr)
    print('replication maxerr', maxerr, flush=True)

    methods = {}
    for name in views:
        subset = [r for r in per_user if r['method'] == name]
        methods[name] = {
            'all': st.aggregate(subset),
            'nonempty': st.aggregate([r for r in subset if r['has_future_purchase']]),
            'group': st.aggregate([r for r in subset if r['in_missing_field_group']]),
        }
        methods[name]['ndcg_at_10'] = methods[name]['all']['ndcg_at_10']

    scores = {}
    for r in per_user:
        scores.setdefault((r['method'], r['user_id']), r['ndcg_at_10'])
    buyer_set = {r['user_id'] for r in per_user
                 if r['method'] == 'p_para' and r['has_future_purchase']}
    cuts = {
        'all': list(users),
        'buyers': [u for u in users if u in buyer_set],
        'group': [u for u in users if u in group_set],
    }
    boot = {}
    for cut, uids in cuts.items():
        d = [scores[('p_infer_missing', u)] - scores[('p_para', u)] for u in uids]
        boot[cut] = {'p_infer_missing_minus_p_para': bootstrap(d, BOOT_SEED, BOOT_B)}
    # Fresh draws for comparability: same seed and B applied independently to each cut.

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'per_user.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_user[0]))
        w.writeheader(); w.writerows(per_user)
    with (OUT / 'ndcg_table.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['method', 'users_all', 'ndcg_all', 'nonempty_users', 'ndcg_nonempty',
                    'group_users', 'ndcg_group'])
        for name, row in sorted(methods.items()):
            w.writerow([name, row['all']['users'], row['all']['ndcg_at_10'],
                        row['nonempty']['users'], row['nonempty']['ndcg_at_10_nonempty'],
                        row['group']['users'], row['group']['ndcg_at_10']])
    with (OUT / 'predictions.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['method', 'user_id', 'rank', 'ISIN'])
        for uid, ranking in preds:
            w.writerows(('p_infer_missing', uid, rank, isin) for rank, isin in enumerate(ranking, 1))
    result = {
        'POST_HOC_INVENTED_AFTER_RESULTS': True,
        'not_added_to_main_ndcg_table': True,
        'question': 'does the p_infer gain come from guesses about empty fields or from paraphrasing all texts?',
        'variant': 'p_infer text for assets with empty sector or industry; p_para text for all other assets',
        'selection_rule': 'same frozen cohort, cutoff, window, eligibility, candidate count, k, cache and scoring as dev_all_warm',
        'protocol_hash': protocol_hash,
        'missing_field_assets': len(missing_assets),
        'assets_total': len(eligible),
        'methods': methods,
        'bootstrap_vs_p_para': boot,
        'timings_seconds': {'load': load_seconds, 'feature': feature_seconds, 'scoring': scoring_seconds,
                            'total': time.perf_counter() - started},
        'integrity': {'source_sha256': source_hashes, 'code_sha256': code_hashes,
                      'prompt_fingerprints': prompt_fp, 'replication_maxerr': maxerr},
        'note': 'Development results; exploratory post-hoc check, not preregistered.',
    }
    (OUT / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'missing_field_assets.json').write_text(json.dumps(sorted(missing_assets), indent=2) + '\n', encoding='utf-8')
    for cut in ('all', 'buyers', 'group'):
        b = boot[cut]['p_infer_missing_minus_p_para']
        print('%s n=%d mean=%+.6f CI=[%+.6f, %+.6f]' % (cut, b['n'], b['mean'], b['ci95'][0], b['ci95'][1]), flush=True)
    for name, row in sorted(methods.items()):
        print('%s all=%.4f nonempty=%.4f group=%.4f' % (
            name, row['all']['ndcg_at_10'] or 0.0, row['nonempty']['ndcg_at_10_nonempty'] or 0.0,
            row['group']['ndcg_at_10'] or 0.0), flush=True)
    print('seconds', time.perf_counter() - started, flush=True)


if __name__ == '__main__':
    main()