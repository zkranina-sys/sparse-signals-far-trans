"""Prespecified paired bootstrap for the HW3 development results.

Settings used by the command that produced outputs/dev_all_warm/bootstrap.json:
- clients are sorted by ID; 2000 bootstrap replicates per cut and comparison;
- random.Random(20261130) for p_infer - p_para and random.Random(20261131)
  for p_rule - p_para, created fresh for every (cut, comparison) pair;
- one replicate draws n client indices with random.randrange(n) with replacement;
- the interval bounds are the 50th and the 1950th sorted replicate means,
  i.e. Python list indices 49 and 1949.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PER_USER = ROOT / 'outputs' / 'dev_all_warm' / 'per_user.csv'
DEFAULT_OUTPUT = ROOT / 'outputs' / 'dev_all_warm' / 'bootstrap.json'
SEED = {'p_infer': 20261130, 'p_rule': 20261131}
REPLICATES = 2000
LOW_INDEX = 49
HIGH_INDEX = 1949


def read_scores(path):
    by = {}
    with path.open(encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            record = by.setdefault(row['user_id'], {})
            record[row['method']] = float(row['ndcg_at_10'])
            record['_buyer'] = row['has_future_purchase'] == 'True'
            record['_group'] = row['in_missing_field_group'] == 'True'
    return by


def ci95(diffs, seed):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(REPLICATES))
    return [means[LOW_INDEX], means[HIGH_INDEX]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-user', type=Path, default=PER_USER)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    scores = read_scores(args.per_user)
    ids = sorted(scores)
    cuts = {
        'all': ids,
        'buyers': [uid for uid in ids if scores[uid]['_buyer']],
        'group': [uid for uid in ids if scores[uid]['_group']],
    }
    result = {}
    for name, uids in cuts.items():
        infer = [scores[uid]['p_infer'] - scores[uid]['p_para'] for uid in uids]
        rule = [scores[uid]['p_rule'] - scores[uid]['p_para'] for uid in uids]
        result[name] = {
            'n': len(uids),
            'p_infer_minus_p_para_mean': sum(infer) / len(infer),
            'p_rule_minus_p_para_mean': sum(rule) / len(rule),
            'p_infer_minus_para_ci95': ci95(infer, SEED['p_infer']),
            'p_rule_minus_para_ci95': ci95(rule, SEED['p_rule']),
        }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                           encoding='utf-8')
    print('wrote', args.output)


if __name__ == '__main__':
    main()