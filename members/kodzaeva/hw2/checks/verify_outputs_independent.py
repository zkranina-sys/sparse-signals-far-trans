#!/usr/bin/env python3
"""Independently audit the fixed HW8 teaching experiment using only stdlib.

This is an AI-agent verification. It does not establish that the student has
personally inspected or recomputed anything. No starter code is imported.
Raw data and all reports remain local; this script makes no network calls.

Run from the workspace after both experiment folders are complete:
    python3 evidence/verify_outputs_independent.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


CLASSICAL = {"popularity", "cbf", "latest_buy", "latest_buy_or_sell"}
FEATURE_FIELDS = (
    "assetCategory", "assetSubCategory", "sector", "industry", "marketID"
)
SOURCES = (
    "transactions.csv", "asset_information.csv", "customer_information.csv",
    "close_prices.csv",
)


class Audit:
    def __init__(self):
        self.counts = Counter()
        self.failures = []

    def check(self, condition, name, detail=None):
        self.counts[name] += 1
        if not condition:
            failure = {"check": name}
            if detail is not None:
                failure["detail"] = detail
            self.failures.append(failure)
        return bool(condition)


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_data(data_dir, cutoff, end):
    """Rebuild eligibility, retained purchases and labels from the raw CSVs.

    Eligibility needs one metadata record strictly before cutoff and at least
    one positive price strictly before cutoff. It does not mean latest price
    must be positive. Exact transaction duplicates are removed across ALL raw
    columns before negative transaction IDs are excluded. Shared positive IDs
    on nonidentical rows are retained.
    """
    latest_assets = {}
    for record in read_csv(data_dir / "asset_information.csv"):
        if record["timestamp"][:10] >= cutoff:
            continue
        key = record["ISIN"]
        if key not in latest_assets or record["timestamp"] > latest_assets[key]["timestamp"]:
            latest_assets[key] = record

    priced = set()
    for record in read_csv(data_dir / "close_prices.csv"):
        if record["timestamp"][:10] < cutoff and float(record["closePrice"]) > 0:
            priced.add(record["ISIN"])
    assets = {key: value for key, value in latest_assets.items() if key in priced}

    profiles = {}
    for record in read_csv(data_dir / "customer_information.csv"):
        if record["timestamp"][:10] >= cutoff:
            continue
        key = record["customerID"]
        if key not in profiles or record["timestamp"] > profiles[key]["timestamp"]:
            profiles[key] = record

    history = defaultdict(Counter)
    truth = defaultdict(set)
    seen_rows = set()
    excluded = Counter()
    raw_rows = retained_past_buys = retained_label_buys = 0
    for record in read_csv(data_dir / "transactions.csv"):
        raw_rows += 1
        exact_record = tuple(record.items())
        if exact_record in seen_rows:
            excluded["exact_duplicate_rows"] += 1
            continue
        seen_rows.add(exact_record)
        if int(record["transactionID"]) < 0:
            excluded["reconstructed_initial_position_rows"] += 1
            continue
        if record["transactionType"] != "Buy" or record["ISIN"] not in assets:
            continue
        day = record["timestamp"][:10]
        if day < cutoff:
            history[record["customerID"]][record["ISIN"]] += 1
            retained_past_buys += 1
        elif cutoff <= day < end:
            truth[record["customerID"]].add(record["ISIN"])
            retained_label_buys += 1

    diagnostics = {
        "raw_transaction_rows": raw_rows,
        "retained_historical_buy_rows": retained_past_buys,
        "retained_evaluation_buy_rows": retained_label_buys,
        "eligible_assets": len(assets),
        "asof_customer_profiles": len(profiles),
        "excluded_rows": dict(excluded),
        "selection_uses_future_truth": False,
    }
    return assets, profiles, history, truth, diagnostics


def attribute_tokens(record):
    answer = set()
    for key in FEATURE_FIELDS:
        value = record.get(key, "").strip()
        if value not in {"", "Not_Available", "nan"}:
            answer.add(f"{key}={value.casefold()}")
    return sorted(answer)


def load_contexts(path, audit, run_name):
    result = {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            record = json.loads(line)
            uid = record["user_id"]
            audit.check(uid not in result, "unique_context_user", {"run": run_name, "line": number})
            result[uid] = record
    return result


def verify_contexts(contexts, users, assets, profiles, history, cutoff, candidate_count, audit, run_name):
    audit.check(set(contexts) == set(users), "context_user_coverage", run_name)
    universe = set(assets)
    for uid in users:
        if uid not in contexts:
            continue
        context = contexts[uid]
        where = {"run": run_name, "user_id": uid}
        audit.check(context.get("cutoff") == cutoff, "context_cutoff", where)
        allowed = context.get("eligible_asset_ids", [])
        audit.check(len(allowed) == len(set(allowed)) and set(allowed) == universe,
                    "context_eligible_universe", where)
        audit.check(context.get("profile", {}).get("timestamp") == profiles[uid]["timestamp"],
                    "context_asof_customer_profile", where)

        reported_history = context.get("history", [])
        history_ids = [item.get("id") for item in reported_history]
        audit.check(len(history_ids) == len(set(history_ids)), "context_unique_history", where)
        counts = {item.get("id"): item.get("buy_count") for item in reported_history}
        audit.check(counts == dict(history.get(uid, {})), "context_history_matches_raw_past_buys", where)

        candidates = context.get("candidate_assets", [])
        candidate_ids = [item.get("id") for item in candidates]
        audit.check(len(candidates) == min(candidate_count, len(universe)), "context_candidate_count", where)
        audit.check(len(candidate_ids) == len(set(candidate_ids)) and set(candidate_ids) <= universe,
                    "context_unique_eligible_candidates", where)
        for item in reported_history + candidates:
            isin = item.get("id")
            if not audit.check(isin in assets, "context_asset_is_asof_eligible", where):
                continue
            raw = assets[isin]
            detail = dict(where, asset_id=isin)
            audit.check(item.get("metadata_timestamp") == raw["timestamp"] and raw["timestamp"][:10] < cutoff,
                        "context_latest_past_metadata", detail)
            audit.check(item.get("attributes") == attribute_tokens(raw), "context_shared_cbf_attributes", detail)
            audit.check(item.get("name") == raw.get("assetName"), "context_asset_name_matches_source", detail)


def load_predictions(path, users, eligible, audit, run_name):
    by_pair = defaultdict(dict)
    methods = set()
    for row in read_csv(path):
        uid, method = row["user_id"], row["method"]
        methods.add(method)
        position = int(row["rank"])
        key = (method, uid)
        where = {"run": run_name, "method": method, "user_id": uid, "rank": position}
        audit.check(uid in users, "prediction_user_in_fixed_cohort", where)
        audit.check(position not in by_pair[key], "prediction_rank_unique", where)
        by_pair[key][position] = row["ISIN"]

    expected = CLASSICAL if run_name == "baseline" else CLASSICAL | {"custom"}
    audit.check(methods == expected, "required_method_set", {"run": run_name, "found": sorted(methods)})
    rankings = {}
    for method in expected:
        audit.check({uid for name, uid in by_pair if name == method} == set(users),
                    "prediction_complete_user_coverage", {"run": run_name, "method": method})
        for uid in users:
            positions = by_pair.get((method, uid), {})
            where = {"run": run_name, "method": method, "user_id": uid}
            audit.check(set(positions) == set(range(1, 11)), "exactly_top10_ranks", where)
            ranking = [isin for _, isin in sorted(positions.items())]
            audit.check(len(ranking) == 10 and len(set(ranking)) == 10, "unique_top10_assets", where)
            audit.check(set(ranking) <= eligible, "top10_assets_eligible_at_cutoff", where)
            rankings[(method, uid)] = ranking
    return rankings, expected


def independent_ndcg(ranking, relevant):
    # Binary gains. Empty future truth is retained and explicitly scores zero.
    if not relevant:
        return 0.0
    discount = [1.0 / math.log2(rank + 1) for rank in range(1, 11)]
    actual = math.fsum(discount[index] for index, isin in enumerate(ranking[:10]) if isin in relevant)
    ideal = math.fsum(discount[: min(10, len(relevant))])
    return actual / ideal


def summarize(users, values, truth):
    nonempty = [uid for uid in users if truth.get(uid)]
    return {
        "users": len(users),
        "nonempty_truth_users": len(nonempty),
        "empty_truth_users": len(users) - len(nonempty),
        "ndcg_at_10": math.fsum(values[uid] for uid in users) / len(users) if users else None,
        "ndcg_at_10_nonempty": math.fsum(values[uid] for uid in nonempty) / len(nonempty) if nonempty else None,
    }


def compare_aggregate(actual, expected, audit, name, detail):
    for key, value in expected.items():
        found = actual.get(key)
        if isinstance(value, float):
            good = isinstance(found, (int, float)) and math.isfinite(found) and math.isclose(found, value, rel_tol=1e-12, abs_tol=1e-12)
        else:
            good = found == value
        audit.check(good, name, dict(detail, field=key, found=found, expected=value))


def verify_run(folder, run_name, users, assets, profiles, history, truth, diagnostics, groups, cutoff, end, source_hashes, audit):
    protocol = read_json(folder / "protocol.json")
    audit.check(protocol.get("cutoff") == cutoff, "fixed_protocol_cutoff", run_name)
    audit.check(protocol.get("end_exclusive") == end, "fixed_protocol_label_window", run_name)
    audit.check(protocol.get("users") == users, "deterministic_warm_cohort_order", run_name)
    audit.check(protocol.get("cohort_selection") == "warm", "declared_warm_cohort", run_name)
    audit.check(protocol.get("eligible_assets") == sorted(assets), "protocol_eligibility_rebuilt_from_raw", run_name)
    audit.check(protocol.get("exclude_seen") is False, "repeat_purchases_allowed", run_name)
    audit.check(protocol.get("k") == 10, "protocol_k10", run_name)
    audit.check(protocol.get("initial_positions") == "excluded", "protocol_negative_id_policy", run_name)
    audit.check(protocol.get("duplicate_policy") == "exact full rows only", "protocol_exact_duplicate_policy", run_name)
    audit.check(protocol.get("asset_feature_fields") == list(FEATURE_FIELDS), "protocol_fixed_feature_fields", run_name)
    audit.check(protocol.get("source_sha256") == source_hashes, "protocol_source_hashes_match_local_raw_files", run_name)
    prehash = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    rebuilt_hash = hashlib.sha256(json.dumps(prehash, sort_keys=True).encode()).hexdigest()
    audit.check(protocol.get("protocol_hash") == rebuilt_hash, "protocol_hash_recomputed", run_name)

    contexts = load_contexts(folder / "contexts.jsonl", audit, run_name)
    verify_contexts(contexts, users, assets, profiles, history, cutoff,
                    protocol.get("candidate_count", 50), audit, run_name)
    rankings, methods = load_predictions(folder / "predictions.csv", users, set(assets), audit, run_name)
    per_user = {}
    for row in read_csv(folder / "per_user.csv"):
        key = (row["method"], row["user_id"])
        audit.check(key not in per_user, "per_user_metric_row_unique", {"run": run_name, "pair": list(key)})
        per_user[key] = row
    audit.check(set(per_user) == {(method, uid) for method in methods for uid in users},
                "per_user_metric_complete_coverage", run_name)

    metrics = read_json(folder / "metrics.json")
    audit.check(metrics.get("protocol_hash") == rebuilt_hash, "metrics_protocol_hash", run_name)
    audit.check(metrics.get("excluded_rows") == diagnostics["excluded_rows"], "excluded_counts_match_raw_rebuild", run_name)
    audit.check(set(metrics.get("methods", {})) == methods, "aggregate_method_coverage", run_name)

    independently_computed = {}
    values_by_method = {}
    for method in sorted(methods):
        values = {uid: independent_ndcg(rankings[(method, uid)], truth.get(uid, set())) for uid in users}
        values_by_method[method] = values
        for uid in users:
            row = per_user.get((method, uid), {})
            where = {"run": run_name, "method": method, "user_id": uid}
            try:
                reported = float(row["ndcg_at_10"])
                relevant_count = int(row["relevant_count"])
            except (KeyError, ValueError, TypeError):
                audit.check(False, "per_user_metric_parseable", where)
                continue
            audit.check(relevant_count == len(truth.get(uid, set())), "per_user_truth_count_recomputed", where)
            audit.check(math.isfinite(reported) and math.isclose(reported, values[uid], rel_tol=1e-12, abs_tol=1e-12),
                        "per_user_ndcg_recomputed", where)
        independently_computed[method] = {name: summarize(members, values, truth) for name, members in groups.items()}
        compare_aggregate(metrics.get("methods", {}).get(method, {}),
                          independently_computed[method]["all_users"], audit,
                          "reported_aggregate_matches_independent_ndcg", {"run": run_name, "method": method})

    custom_failures = metrics.get("custom_failures", [])
    audit.check(isinstance(custom_failures, list), "failure_log_is_list", run_name)
    if isinstance(custom_failures, list):
        for failure in custom_failures:
            uid = failure.get("user_id")
            audit.check(uid in users, "fallback_user_in_fixed_cohort", {"run": run_name, "user_id": uid})
            if "custom" in methods and uid in users:
                audit.check(rankings[("custom", uid)] == rankings[("cbf", uid)],
                            "logged_custom_failure_uses_count_cbf_fallback", {"run": run_name, "user_id": uid})

    changed = None
    if "custom" in methods:
        changed = {}
        for name, members in groups.items():
            diffs = [values_by_method["custom"][uid] - values_by_method["cbf"][uid] for uid in members]
            changed[name] = {
                "users": len(members),
                "mean_paired_custom_minus_cbf": math.fsum(diffs) / len(diffs) if diffs else None,
                "positive_user_differences": sum(value > 1e-12 for value in diffs),
                "negative_user_differences": sum(value < -1e-12 for value in diffs),
                "zero_user_differences": sum(abs(value) <= 1e-12 for value in diffs),
                "changed_top10_lists": sum(rankings[("custom", uid)] != rankings[("cbf", uid)] for uid in members),
            }
    result = {
        "folder": str(folder.resolve()),
        "protocol_hash": rebuilt_hash,
        "methods": independently_computed,
        "custom_minus_cbf": changed,
        "logged_custom_failure_count": len(custom_failures) if isinstance(custom_failures, list) else None,
    }
    return protocol, contexts, rankings, result


def main():
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=workspace / "data/FAR-Trans")
    parser.add_argument("--baseline-dir", type=Path, default=workspace / "outputs/baseline")
    parser.add_argument("--change-dir", type=Path, default=workspace / "outputs/change")
    parser.add_argument("--report", type=Path, default=workspace / "evidence/independent_verification.json")
    parser.add_argument("--expected-users", type=int, default=100)
    parser.add_argument("--cutoff", default="2021-06-01")
    parser.add_argument("--end-exclusive", default="2021-07-01")
    args = parser.parse_args()
    audit = Audit()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verification_kind": "AI-agent independent local raw-data and metric verification",
        "no_starter_code_imported": True,
        "network_used_by_this_script": False,
        "student_checks": {
            "status": "not established by this script",
            "note": "Agent computations must not be described as student personal checks.",
        },
        "source_script_sha256": file_hash(Path(__file__)),
        "parameters": {"cutoff": args.cutoff, "end_exclusive": args.end_exclusive,
                       "expected_users": args.expected_users, "cohort": "warm",
                       "repeat_purchases_allowed": True, "k": 10},
    }
    try:
        audit.check(args.cutoff == "2021-06-01" and args.end_exclusive == "2021-07-01",
                    "approved_june_2021_teaching_window")
        audit.check(args.expected_users == 100, "approved_100_user_teaching_sample")
        assets, profiles, history, truth, diagnostics = reconstruct_data(args.data_dir, args.cutoff, args.end_exclusive)
        warm_users = [uid for uid in profiles if history.get(uid)]
        warm_users.sort(key=lambda uid: (hashlib.sha256(("42:" + uid).encode()).hexdigest(), uid))
        users = warm_users[:args.expected_users]
        audit.check(len(users) == args.expected_users, "expected_fixed_cohort_size")
        audit.check(all(history.get(uid) for uid in users), "all_selected_users_are_historically_warm")
        target = [uid for uid in users if len(history.get(uid, {})) >= 4]
        groups = {
            "all_users": users,
            "all_nonempty_truth": [uid for uid in users if truth.get(uid)],
            "history_4plus": target,
            "history_4plus_nonempty_truth": [uid for uid in target if truth.get(uid)],
            "history_1to3": [uid for uid in users if 1 <= len(history.get(uid, {})) <= 3],
            "history_4to9": [uid for uid in users if 4 <= len(history.get(uid, {})) <= 9],
            "history_10plus": [uid for uid in users if len(history.get(uid, {})) >= 10],
        }
        report["raw_data_reconstruction"] = diagnostics
        report["cohort_counts"] = {
            name: {"users": len(members), "nonempty_truth_users": sum(bool(truth.get(uid)) for uid in members)}
            for name, members in groups.items()
        }
        source_hashes = {name: file_hash(args.data_dir / name) for name in SOURCES}
        report["source_sha256"] = source_hashes
        baseline = verify_run(args.baseline_dir, "baseline", users, assets, profiles, history, truth,
                              diagnostics, groups, args.cutoff, args.end_exclusive, source_hashes, audit)
        change = verify_run(args.change_dir, "change", users, assets, profiles, history, truth,
                            diagnostics, groups, args.cutoff, args.end_exclusive, source_hashes, audit)
        audit.check(baseline[0] == change[0], "both_runs_use_identical_full_protocol")
        audit.check(baseline[1] == change[1], "both_runs_use_identical_contexts")
        for method in sorted(CLASSICAL):
            audit.check(all(baseline[2][(method, uid)] == change[2][(method, uid)] for uid in users),
                        "classical_predictions_identical_between_runs", method)
        report["runs"] = {"baseline": baseline[3], "change": change[3]}
    except Exception as error:
        audit.check(False, "verification_completed_without_exception",
                    {"exception": type(error).__name__, "message": str(error)})

    report["check_counts"] = dict(sorted(audit.counts.items()))
    report["total_check_instances"] = sum(audit.counts.values())
    report["failures"] = audit.failures
    report["status"] = "passed" if not audit.failures else "failed"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["total_check_instances"],
                      "failures": len(audit.failures), "report": str(args.report.resolve())}, ensure_ascii=False))
    return 0 if not audit.failures else 1


if __name__ == "__main__":
    sys.exit(main())
