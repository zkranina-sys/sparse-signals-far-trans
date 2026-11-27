# Frozen protocol - summary without client IDs

This file intentionally contains NO client identifiers. The exact list of
10,404 client IDs is uniquely recoverable from the FAR-Trans source data by
re-running the frozen selection code in code/eval_dev.py with the settings
below. The published hash fixes the resulting cohort:

- protocol hash: 12ccb71cd29f052609c79e7a5f5bd5137238fe6f0196802983a75178a8f83759
- rule hash: a558a6d5bfad2835f105542bfec4c05b46cc612ed491577226fe368908eaa2f4
- cohort selection: warm_all, deterministic order, max_users=20000
- users: 10,404 warm users; missing-field subgroup: 5,641 users
- cutoff: 2021-06-01; evaluation window: [2021-06-01, 2021-07-01)
- eligible assets: 767; candidate count: 50; k: 10; repeats allowed
- primary mean: users with empty truth score zero; buyers mean reported separately
- description cache: 1,534 HW2 answers reused unchanged; no new inference in scoring

Descriptor files with counts and hashes are safe to publish; the client list
itself is not included in this upload folder.
