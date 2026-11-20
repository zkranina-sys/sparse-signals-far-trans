#!/usr/bin/env python3
"""Check the scientific effect of history weights without calling a model.

This is an AI-agent check, not evidence of student personal verification.
It checks score invariance under positive scaling, the identity ablation,
masking of actual full histories, and (when supplied) saved response effects.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predictions(path):
    result = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            result.setdefault((row['method'], row['user_id']), {})[int(row['rank'])] = row['ISIN']
    return {key: [value[i] for i in sorted(value)] for key, value in result.items()}


def ratios_changed(counts, weights, tolerance=1e-12):
    total_count = math.fsum(counts.values())
    total_weight = math.fsum(weights.values())
    if total_weight <= 0:
        return True
    return any(abs(count / total_count - weights.get(key, 0) / total_weight) > tolerance
               for key, count in counts.items())


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, default=root / 'outputs/baseline')
    parser.add_argument('--data-dir', type=Path, default=root / 'data/FAR-Trans')
    parser.add_argument('--encoder-source', type=Path, default=root / 'src/history_encoder.py')
    parser.add_argument('--responses', type=Path)
    parser.add_argument('--change-dir', type=Path)
    parser.add_argument('--report', type=Path, default=root / 'evidence/encoder_effect_check.json')
    args = parser.parse_args()
    source = root / 'source/starter.py'
    starter = load_module('effect_check_starter', source)
    encoder = load_module('effect_check_encoder', args.encoder_source)
    protocol = json.loads((args.baseline_dir / 'protocol.json').read_text())
    contexts = read_jsonl(args.baseline_dir / 'contexts.jsonl')
    saved_predictions = predictions(args.baseline_dir / 'predictions.csv')
    assets, _, history, _, _, _ = starter.load_data(args.data_dir, protocol['cutoff'], 30)
    features = {key: starter.tokens(asset) for key, asset in assets.items()}
    popularity = starter.popularity_scores(history, assets)
    failures = []

    # A controlled nondegenerate fixture: swapping relative weights must swap
    # the order of assets sharing different observed sectors.
    synthetic_features = {
        'A': ['assetCategory=stock', 'sector=utilities'],
        'B': ['assetCategory=stock', 'sector=technology'],
        'C': ['assetCategory=stock', 'sector=utilities'],
        'D': ['assetCategory=bond', 'sector=government'],
        'E': ['assetCategory=stock', 'sector=technology'],
    }
    count_scores = starter.content_scores({'A': 3, 'B': 1}, synthetic_features)
    changed_scores = starter.content_scores({'A': 1, 'B': 3}, synthetic_features)
    synthetic_effect = count_scores['C'] > count_scores['E'] and changed_scores['E'] > changed_scores['C']
    if not synthetic_effect:
        failures.append('Synthetic relative-weight change failed to change the expected order')

    identity_mismatches, masking_mismatches = [], []
    scale_score_max_error, scale_order_changes, near_tie_changes = 0.0, [], []
    for context in contexts:
        uid = context['user_id']
        counts = {item['id']: item['buy_count'] for item in context['history']}
        rank = starter.encode_ranking(counts, context, features, popularity)
        if rank != saved_predictions[('cbf', uid)]:
            identity_mismatches.append(uid)
        prompt, labels = encoder.build_prompt(context)
        if len(labels) != len(counts) or any(key in prompt for key in counts) or uid in prompt:
            masking_mismatches.append(uid)
        original = starter.content_scores(counts, features)
        scaled = starter.content_scores({key: 7 * value for key, value in counts.items()}, features)
        error = max(abs(original[key] - scaled[key]) for key in original)
        scale_score_max_error = max(scale_score_max_error, error)
        scaled_order, original_order = starter.order(scaled), starter.order(original)
        if scaled_order[:10] != original_order[:10]:
            scale_order_changes.append(uid)
            moved = set(scaled_order[:10]) | set(original_order[:10])
            pairs = [(a, b) for a in moved for b in moved
                     if (original_order.index(a) < original_order.index(b)) !=
                        (scaled_order.index(a) < scaled_order.index(b))]
            near_tie_changes.append({'user_id': uid, 'swapped_pairs': len(pairs),
                                    'max_original_gap': max((abs(original[a] - original[b]) for a,b in pairs), default=0.0)})
    if identity_mismatches:
        failures.append('Count-identity ablation did not reproduce saved CBF rankings')
    if masking_mismatches:
        failures.append('Full-history identifier masking check failed')
    if scale_score_max_error > 1e-12:
        failures.append('Positive-scaling score error exceeded numerical tolerance')
    if any(row['max_original_gap'] > 1e-12 for row in near_tie_changes):
        failures.append('Positive-scaling rank changes were not restricted to numerical near ties')

    result = {
        'verification_kind': 'AI-agent scientific mechanism check; no model call',
        'starter_sha256': sha(source), 'encoder_sha256': sha(args.encoder_source),
        'protocol_hash': protocol['protocol_hash'], 'users': len(contexts),
        'synthetic_relative_weight_change_affects_expected_order': synthetic_effect,
        'synthetic_count_scores': count_scores, 'synthetic_changed_scores': changed_scores,
        'identity_ablation_mismatched_users': identity_mismatches,
        'full_history_masking_mismatched_users': masking_mismatches,
        'positive_scaling_factor': 7,
        'positive_scaling_max_absolute_score_error': scale_score_max_error,
        'positive_scaling_changed_top10_users': scale_order_changes,
        'positive_scaling_near_tie_diagnostics': near_tie_changes,
        'floating_point_caution': 'Common positive scaling preserves normalized profiles mathematically. Binary floating-point summation may change near-tied order; do not interpret such changes as a substantive encoder effect.',
    }
    if args.responses:
        responses = read_jsonl(args.responses)
        by_user = {row['user_id']: row for row in responses}
        if set(by_user) != {row['user_id'] for row in contexts} or len(by_user) != len(responses):
            failures.append('Saved responses do not uniquely cover all frozen cohort users')
        change = predictions(args.change_dir / 'predictions.csv') if args.change_dir else None
        cases = []
        for context in contexts:
            uid = context['user_id']
            if uid not in by_user:
                continue
            response = by_user[uid]
            counts = {item['id']: item['buy_count'] for item in context['history']}
            weights = response['history_weights']
            if response.get('protocol_hash') != protocol['protocol_hash']:
                failures.append(f'Response protocol mismatch: {uid}')
            rank = starter.encode_ranking(weights, context, features, popularity)
            scores = starter.content_scores(weights, features)
            count_scores = starter.content_scores(counts, features)
            score_change = max(abs(scores[key] - count_scores[key]) for key in scores)
            different_ratios = ratios_changed(counts, weights)
            different_ranking = rank != saved_predictions[('cbf', uid)]
            row = {'user_id': uid, 'history_assets': len(counts),
                   'relative_weight_distribution_changed': different_ratios,
                   'max_absolute_score_change': score_change,
                   'top10_changed': different_ranking}
            cases.append(row)
            if change is not None and rank != change.get(('custom', uid)):
                failures.append(f'Saved custom ranking differs from saved response scoring: {uid}')
            if len(counts) < 4 and (different_ratios or different_ranking):
                failures.append(f'Outside-subgroup identity policy violated: {uid}')
        result['saved_response_analysis'] = {
            'response_file': str(args.responses.resolve()), 'response_sha256': sha(args.responses),
            'users_with_changed_relative_weights': sum(row['relative_weight_distribution_changed'] for row in cases),
            'users_with_changed_scores_over_1e_minus_12_tolerance': sum(row['max_absolute_score_change'] > 1e-12 for row in cases),
            'users_with_changed_top10': sum(row['top10_changed'] for row in cases),
            'cases': cases,
        }
    result['failures'] = failures
    result['status'] = 'passed' if not failures else 'failed'
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'users': len(contexts),
                      'scaling_max_score_error': scale_score_max_error,
                      'identity_mismatches': len(identity_mismatches),
                      'report': str(args.report.resolve())}))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
