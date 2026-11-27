"""Prepare one hand-checkable nDCG@10 case from the frozen warm-user order."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import sys

ROOT = Path(r'C:\Users\555\projects\hw3')
sys.path.insert(0, str(ROOT/'code'))


def ndcg(ranking, truth, k=10):
    gain = sum(1/math.log2(i+2) for i, x in enumerate(ranking[:k]) if x in truth)
    ideal = sum(1/math.log2(i+2) for i in range(min(k, len(truth))))
    return gain, ideal, gain/ideal if ideal else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=ROOT/'outputs'/'protocol'/'protocol.json')
    p.add_argument('--results', type=Path, default=ROOT/'outputs'/'dev_all_warm')
    p.add_argument('--data-dir', type=Path, default=Path(r'C:\Users\555\projects\hw2\colab_upload'))
    p.add_argument('--output', type=Path, default=ROOT/'outputs'/'manual_ndcg_check.json')
    p.add_argument('--rank', type=int, default=1, help='1 = first warm user in frozen order with nonempty truth')
    a = p.parse_args()
    import starter
    users = json.loads(a.protocol.read_text(encoding='utf-8'))['users']
    assets, profiles, history, truth, end, excluded = starter.load_data(a.data_dir, '2021-06-01', 30)
    rows = list(csv.DictReader((a.results/'per_user.csv').open(encoding='utf-8')))
    by_user = {}
    for r in rows:
        by_user.setdefault(r['user_id'], []).append(r)
    chosen = []
    for uid in users:
        rec = next((r for r in by_user.get(uid, []) if r['method'] == 'p_para'), None)
        if rec and int(rec['relevant_count']) > 0:
            chosen.append(uid)
            if len(chosen) >= a.rank:
                break
    if not chosen:
        raise ValueError('No user with nonempty truth found')
    uid = chosen[-1]
    preds = list(csv.DictReader((a.results/'predictions.csv').open(encoding='utf-8')))
    relevant = truth.get(uid, set())
    methods = {}
    for method in sorted({r['method'] for r in preds}):
        ranking = [r['ISIN'] for r in sorted(
            (x for x in preds if x['method'] == method and x['user_id'] == uid),
            key=lambda x: int(x['rank']))]
        if len(ranking) != 10:
            raise ValueError(f'{method}: expected 10 items')
        gain, ideal, value = ndcg(ranking, relevant)
        methods[method] = {'ranking': ranking, 'hits': [x for x in ranking if x in relevant],
                           'gain': gain, 'ideal': ideal, 'ndcg_at_10': value}
    out = {'selection_rule': 'first warm user in frozen protocol order with relevant_count > 0',
           'rank_in_selection': a.rank, 'user_id': uid,
           'future_purchase_assets_in_eligible_universe': sorted(relevant),
           'methods': methods, 'note': 'Prepared for the student hand calculation; not a claim of correctness.'}
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('user', uid, 'relevant', len(relevant))
    for name, value in methods.items():
        print(name, f"{value['ndcg_at_10']:.4f}", 'hits', len(value['hits']))


if __name__ == '__main__':
    main()