# POST-HOC CHECK - INVENTED AFTER SEEING RESULTS

This folder is separate from the main development results. Nothing here was added to
outputs/dev_all_warm/ndcg_table.csv.

Variant p_infer_missing: use the p_infer asset description for assets whose sector or
industry is empty in the source data (411 of 767 assets), and the p_para description for
all other assets. Everything else is identical to the frozen protocol: 10,404 warm users,
cutoff 2021-06-01, window [2021-06-01, 2021-07-01), same HW2 cache, candidate count 50,
k=10, empty truth scores zero. Protocol hash:
12ccb71cd29f052609c79e7a5f5bd5137238fe6f0196802983a75178a8f83759

The saved p_para and p_infer per-user values from the main run were reproduced with
maximum absolute error 0.0 before the new variant was scored.

Files:
- metrics.json: main results, paired bootstrap against p_para, integrity hashes.
- ndcg_table.csv: three methods only; post-hoc, not the main table.
- per_user.csv and predictions.csv are deliberately NOT copied into the upload folder.
- missing_field_assets.json: the 411 asset IDs used for the p_infer text (no client IDs).
- metrics.json is the single source of intervals: bootstrap_vs_p_para uses clients
  sorted by ID, random.Random(20261126), B=2000, rng.choices sampling and sorted
  replicate means at indices int(0.025*B)=50 and int(0.975*B)=1950.

Interpretation limit: TF-IDF is refit jointly for each text corpus, so this is not a clean
additive decomposition into inferred facts versus paraphrasing. Treat all statements here
as exploratory.
