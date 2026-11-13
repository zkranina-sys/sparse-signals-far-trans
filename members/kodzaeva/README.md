# Maria Kodzaeva — Session 1: History-Weight Encoder

Team: **Sparse Signals**. Project: *Filling the Gaps: Where LLMs Improve a Repeat-Purchase Recommender on FAR-Trans*.

This folder contains the reproducible classical baseline and the individual research plan. The proposed LLM history-weight Encoder has **not** been run. The PDF report and native AI-session export are submitted through the homework channel separately.

## Contents

- `study_plan.md`: research question, same-information comparison, proposed integration and individual contribution.
- `source/`: unchanged teaching starter at upstream commit `51337032503c56e3a28ef646a24aa016782e1bd3`.
- `reproduce.py`: standard-library replay and independent frozen-label verification.
- `frozen_outputs/`: exact saved protocol, four-method metrics, per-user scores and Top-10 predictions.
- `evidence/manual_ndcg_check.md` and `.json`: one retraceable real-user calculation and the scope of personal verification.
- `evidence/source_snapshot_check.json`: hashes and equality checks against the pinned upstream source.
- `package_check.json`: result of the supported-runtime replay used to verify this publication layout.
- `manifest.json`: SHA-256 of the published files, excluding the manifest itself.

## Reproduce

Use Python **3.10+**; the verified replay used **3.12.14**. No third-party Python packages, model key or model call are needed. From this folder, download the official public data archive into the ignored `raw/` directory:

```bash
python3 --version
mkdir -p raw
curl --fail --location https://researchdata.gla.ac.uk/1658/1/FAR-Trans.zip --output raw/FAR-Trans.zip
python3 reproduce.py --output-dir package_replay --result-json replay_check.json
```

To check the saved predictions against eligible future Buy labels without re-running the starter:

```bash
python3 reproduce.py --verify-only --output-dir package_verify --result-json verify_check.json
```

Use a new output-directory name for each invocation. The reproducer refuses to overwrite existing output directories or frozen outputs. It extracts the archive into the chosen work directory and verifies the four required raw CSV hashes against the frozen protocol. The archive itself is not committed.

Replay uses the unchanged starter with the recorded cutoff, horizon, cohort and candidate count, plus `PYTHONHASHSEED=0`. It checks identical protocol content and per-method/per-user scores within `1e-12`. Exact recommendation-row identity across environments is not asserted: floating near-ties in CBF can change ordering while displayed metrics remain equal. The saved predictions are the evidence for the reported run.

## Actual teaching result

The shared teaching settings are cutoff `2021-06-01`, future interval `[2021-06-01, 2021-07-01)`, a deterministic warm sample of 100 users, candidate count 50 and repeat purchases allowed. These are not the final project benchmark protocol. Empty truth scores zero in the all-user mean: 89 users have empty truth and 11 have future purchases.

| Method | nDCG@10: all 100 | nDCG@10: nonempty 11 |
|---|---:|---:|
| Popularity | 0.0286 | 0.2599 |
| CBF | 0.0201 | 0.1827 |
| Latest buy | 0.0277 | 0.2520 |
| Latest buy or sell | 0.0271 | 0.2464 |

The predeclared 4+ distinct historical-asset subgroup has 37 users, 10 with nonempty future truth. Its CBF means are 0.0543 over all 37 and 0.2010 over the nonempty 10. These are baseline feasibility findings, not an LLM effect estimate.

A supported-runtime replay of the complete four-method command took **2.62 seconds** on macOS arm64. This is whole-run timing, not per-method timing or an inference-cost measurement. The saved package check verified all 400 per-user method/user rows and the raw hashes.

## Attribution and data

The teaching source is preserved unchanged from [dryjins/RecSys-LLMs, pinned FAR-Trans starter](https://github.com/dryjins/RecSys-LLMs/tree/51337032503c56e3a28ef646a24aa016782e1bd3/project/far-trans). Original source ownership is retained; the folder is an attributed snapshot, not a claim that the student authored the starter. The study plan, audit materials and reproducer were prepared with disclosed AI assistance.

Data: Javier Sanz-Cruzado, Nikolaos Droukas and Richard McCreadie, *FAR-Trans: An Investment Dataset for Financial Asset Recommendation*, IJCAI-2024 Fin-RecSys Workshop. The [University of Glasgow release](https://researchdata.gla.ac.uk/1658/) is distributed under **CC BY 4.0**, DOI [10.5525/gla.researchdata.1658](https://doi.org/10.5525/gla.researchdata.1658). The anonymized example does not attach or infer a real identity. Purchase relevance is not investment profitability or suitability.
