#!/usr/bin/env python3
"""Replay or independently verify the frozen FAR-Trans teaching baseline.

This program uses only Python's standard library. It never modifies frozen_outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "frozen_outputs"
RAW_ARCHIVE = ROOT / "raw" / "FAR-Trans.zip"
SOURCE = ROOT / "source" / "starter.py"
CSV_NAMES = (
    "transactions.csv",
    "asset_information.csv",
    "customer_information.csv",
    "close_prices.csv",
)
TOLERANCE = 1e-12


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_supported_python():
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required")


def prepare_raw(output_dir, protocol):
    work = output_dir / "work"
    if work.exists():
        raise SystemExit(f"Refusing to reuse existing work directory: {work}")
    work.mkdir(parents=True)
    with zipfile.ZipFile(RAW_ARCHIVE) as archive:
        archive.extractall(work)
    data_dir = work / "FAR-Trans"
    if not data_dir.is_dir():
        raise SystemExit("Archive did not contain FAR-Trans/")
    actual = {name: sha256(data_dir / name) for name in CSV_NAMES}
    expected = protocol["source_sha256"]
    if actual != expected:
        raise SystemExit("Raw CSV SHA-256 mismatch: " + json.dumps(
            {name: {"expected": expected[name], "actual": actual[name]}
             for name in CSV_NAMES if expected[name] != actual[name]}, sort_keys=True))
    return data_dir, actual


def ndcg10(ranking, truth):
    if not truth:
        return 0.0
    dcg = sum(1 / math.log2(rank + 1)
              for rank, asset in enumerate(ranking[:10], start=1) if asset in truth)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(truth)) + 1))
    return dcg / ideal


def future_truth(data_dir, protocol):
    """Derive binary future-Buy truth with the documented row exclusions.

    The frozen protocol supplies the as-of eligible asset universe. This independent
    check scans raw future transactions, removes exact duplicates and negative IDs,
    and intersects retained Buys with that universe.
    """
    users = set(protocol["users"])
    eligible = set(protocol["eligible_assets"])
    cutoff, end = protocol["cutoff"], protocol["end_exclusive"]
    seen, truth = set(), defaultdict(set)
    with (data_dir / "transactions.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = tuple(sorted(row.items()))
            if key in seen:
                continue
            seen.add(key)
            if int(row["transactionID"]) < 0:
                continue
            stamp = row["timestamp"][:10]
            if (row["customerID"] in users and row["transactionType"] == "Buy"
                    and row["ISIN"] in eligible and cutoff <= stamp < end):
                truth[row["customerID"]].add(row["ISIN"])
    return truth


def load_predictions(path):
    predictions = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            predictions[(row["user_id"], row["method"])].append(
                (int(row["rank"]), row["ISIN"]))
    return {key: [asset for _, asset in sorted(rows)] for key, rows in predictions.items()}


def load_per_user(path):
    values = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            values[(row["user_id"], row["method"])] = {
                "relevant_count": int(row["relevant_count"]),
                "ndcg_at_10": float(row["ndcg_at_10"]),
            }
    return values


def compare_per_user(expected_path, actual_path):
    expected, actual = load_per_user(expected_path), load_per_user(actual_path)
    if set(expected) != set(actual):
        raise SystemExit("Per-user keys differ")
    failures = []
    for key in expected:
        if expected[key]["relevant_count"] != actual[key]["relevant_count"]:
            failures.append((key, "relevant_count"))
        if abs(expected[key]["ndcg_at_10"] - actual[key]["ndcg_at_10"]) > TOLERANCE:
            failures.append((key, "ndcg_at_10"))
    if failures:
        raise SystemExit("Per-user comparison failed: " + repr(failures[:5]))
    return len(expected)


def compare_metrics(expected, actual):
    if expected["protocol_hash"] != actual["protocol_hash"]:
        raise SystemExit("metrics.json protocol_hash differs")
    for method, expected_row in expected["methods"].items():
        actual_row = actual["methods"].get(method)
        if actual_row is None:
            raise SystemExit(f"metrics.json missing method: {method}")
        for field, value in expected_row.items():
            observed = actual_row.get(field)
            if isinstance(value, float):
                if observed is None or abs(value - observed) > TOLERANCE:
                    raise SystemExit(f"metrics mismatch: {method}.{field}")
            elif value != observed:
                raise SystemExit(f"metrics mismatch: {method}.{field}")


def verify_frozen(data_dir, protocol):
    truth = future_truth(data_dir, protocol)
    predictions = load_predictions(FROZEN / "predictions.csv")
    stored = load_per_user(FROZEN / "per_user.csv")
    failures = []
    for (user_id, method), row in stored.items():
        calculated_truth = truth.get(user_id, set())
        calculated = ndcg10(predictions[(user_id, method)], calculated_truth)
        if row["relevant_count"] != len(calculated_truth):
            failures.append((user_id, method, "relevant_count"))
        if abs(row["ndcg_at_10"] - calculated) > TOLERANCE:
            failures.append((user_id, method, "ndcg_at_10"))
    if failures:
        raise SystemExit("Independent frozen-label check failed: " + repr(failures[:5]))
    return {"per_user_rows_checked": len(stored), "max_tolerance": TOLERANCE}


def replay(data_dir, output_dir, protocol):
    rerun = output_dir / "rerun_outputs"
    if rerun.exists():
        raise SystemExit(f"Refusing to overwrite rerun outputs: {rerun}")
    horizon_days = (date.fromisoformat(protocol["end_exclusive"])
                    - date.fromisoformat(protocol["cutoff"])).days
    command = [sys.executable, str(SOURCE), "run", "--data-dir", str(data_dir),
               "--output-dir", str(rerun), "--cutoff", protocol["cutoff"],
               "--horizon-days", str(horizon_days), "--max-users", str(len(protocol["users"])),
               "--cohort", protocol["cohort_selection"], "--candidate-count",
               str(protocol["candidate_count"])]
    if protocol["exclude_seen"]:
        command.append("--exclude-seen")
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    subprocess.run(command, check=True, env=environment)
    rerun_protocol = read_json(rerun / "protocol.json")
    frozen_metrics = read_json(FROZEN / "metrics.json")
    rerun_metrics = read_json(rerun / "metrics.json")
    if rerun_protocol != protocol:
        raise SystemExit("Rerun protocol differs from frozen protocol")
    compare_metrics(frozen_metrics, rerun_metrics)
    per_user_rows = compare_per_user(FROZEN / "per_user.csv", rerun / "per_user.csv")
    return {"rerun_output_dir": str(rerun), "protocol_matched": True,
            "metrics_matched_within_tolerance": True, "per_user_rows_checked": per_user_rows,
            "metrics_tolerance": TOLERANCE, "prediction_rows_not_compared": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check frozen predictions against raw future Buy labels; do not rerun.")
    parser.add_argument("--result-json", type=Path,
                        help="Optional path for a machine-readable result outside frozen_outputs.")
    args = parser.parse_args()
    require_supported_python()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    protocol = read_json(FROZEN / "protocol.json")
    args.output_dir.mkdir(parents=True)
    try:
        data_dir, raw_hashes = prepare_raw(args.output_dir, protocol)
        result = {"raw_csv_sha256_verified": raw_hashes,
                  "frozen_label_check": verify_frozen(data_dir, protocol),
                  "mode": "verify-only" if args.verify_only else "replay"}
        if not args.verify_only:
            result["replay"] = replay(data_dir, args.output_dir, protocol)
        print("PASS: " + json.dumps(result, sort_keys=True))
        if args.result_json:
            args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    except Exception:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
