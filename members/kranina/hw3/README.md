# HW3 development study - upload folder

For: members/kranina/hw3. The experiment commit hash is given in the HW3 report.

## Research question

Does letting an LLM-Encoder infer empty asset fields from the asset name improve
nDCG@10 over a paraphrase-only encoder, and does a frozen keyword rule give the
same benefit? All numbers are development results, not an official benchmark.

## Contents

- code/: evaluation and auxiliary scripts plus the frozen rule. Includes
  bootstrap_main.py, which reproduces bootstrap.json byte-for-byte, and
  posthoc_missing_variant.py, the exploratory p_infer_missing check.
- protocol_hash.txt, protocol_summary.md: frozen protocol hash and settings,
  with NO client IDs.
- keyword_rule.json: frozen rule, identical to code/keyword_rule.json.
- ndcg_table.csv, metrics.json: all eight methods on all warm users.
- bootstrap.json: prespecified paired intervals (all, buyers, missing-field group).
- bootstrap_additional_posthoc.json: EXTRA cuts invented after seeing results;
  outside-group and buyers-inside-group. Do not present as preregistered.
- coverage.json: rule and approximate LLM field coverage.
- run_manifest.json: code/data hashes, settings, timings, protocol hashes.
- error_analysis_report.json: cached-output validity and unchanged cases.
- cache_check_report.json: 48 fresh Colab/T4 answers compared with the HW2 cache;
  48 exact matches, 0 mismatches, 0 protocol problems.
- demo100_ndcg_table.csv: exact reproduction of the HW2 100-user demo.
- posthoc_infer_missing/: POST-HOC check invented after seeing results.
  Summary tables, an asset-ID list and code only; no per-client files.
  See posthoc_infer_missing/README.md.
- manual_label_25.csv, manual_label_25_summary.json, manual_label_summary.csv:
  student's manual accuracy labels for 25 asset descriptions. Asset ISINs
  only, no client IDs.

## Protocol

Cutoff 2021-06-01; window [2021-06-01, 2021-07-01); 10,404 warm users;
767 eligible assets; candidate count 50; k=10; repeat purchases allowed.
No client IDs are published: the list of 10,404 clients is uniquely
recoverable from the source data by the frozen selection code.
Users with empty truth score zero in the primary mean. Protocol hash:
12ccb71cd29f052609c79e7a5f5bd5137238fe6f0196802983a75178a8f83759.
Rule hash: a558a6d5bfad2835f105542bfec4c05b46cc612ed491577226fe368908eaa2f4.

## Methods

Popularity, CBF, Latest buy, Latest buy or sell, original_text_cbf, p_para,
p_infer, p_rule (ablation: p_para plus frozen keyword values in the generated
TF-IDF channel). Same users, cutoff, assets, labels and scoring rules.

## Headline numbers (development)

p_infer 0.0427 vs p_para 0.0424 on all users; paired difference +0.0003,
95% bootstrap [-0.0001, +0.0008] includes zero. In the prespecified
missing-field subgroup (5,641 users) the difference is +0.0007 with interval
[+0.00008, +0.0014]. p_rule 0.0423, no improvement over p_para.

## Prespecified bootstrap

code/bootstrap_main.py reproduces bootstrap.json byte-for-byte (verified by
SHA-256; the original file was not modified). Settings: random.Random(20261130)
for p_infer - p_para and random.Random(20261131) for p_rule - p_para, 2000
replicates, clients sorted by ID, interval bounds at the 50th and 1950th
sorted replicate means.

## Post-hoc check: p_infer_missing (INVENTED AFTER SEEING RESULTS)

Use p_infer text for the 411 of 767 assets with empty sector or industry and
p_para text for the other assets; all other protocol settings are unchanged.
NOT added to the main ndcg_table.csv. nDCG@10 (4 decimals): 0.0424 all users,
0.3749 buyers, 0.0532 missing-field group. Paired p_infer_missing - p_para:
+0.000070 [-0.000186, +0.000327] all; +0.000617 [-0.001720, +0.002940]
buyers; +0.000254 [-0.000126, +0.000677] group; all intervals include zero.
The variant retains about 21% of the full p_infer point estimate (about 35% in the
group). TF-IDF is refit for each corpus, so this is not a causal decomposition.
All quoted post-hoc intervals come from posthoc_infer_missing/metrics.json only
(clients sorted by ID, random.Random(20261126), B=2000, rng.choices and sorted
replicate means at indices 50 and 1950). A second protocol-order file with slightly
different intervals was removed from this folder.

## Failure analysis

1,534 cached answers: 0 invalid outputs, 0 JSON echoes, 0 fallbacks;
2 assets gave identical para and infer text. Rule fills at least one field for
228 of 767 assets; the LLM coverage figure is a lexical approximation.
Student checked 25 descriptions by hand: 4 correct, 18 unchanged, 2 invented
and 1 wrong overall; among the 10 assets with empty sector or industry,
2 correct, 5 unchanged, 2 invented and 1 wrong. A fresh Colab/T4 check
regenerated 24 assets x 2 prompts: all 48 answers matched the HW2 cache
verbatim, with 0 protocol problems; cached texts were not replaced. Its
timing is reported separately from HW2 generation time.

## Deliberately not included

FAR-Trans source data (transactions, asset_information, customer_information,
close_prices), per-client files (per_user.csv, predictions.csv,
unchanged_cases.jsonl, and the per-client files of the post-hoc check),
and protocol.json with its 10,404 client IDs. The included manual label
sheet contains asset ISINs but no client identifiers. run_manifest.json is
kept only after verifying that it contains no client IDs.

## Contribution and verification

Student: research decisions, approval of runs, manual labels and hand nDCG.
Assistant: code, frozen rule and protocol, local evaluation, draft text.
AI usage is disclosed in the report PDF. No AI-generated citation was accepted
without checking the original source. FAR-Trans data are not committed.
