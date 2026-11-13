# One-user nDCG@10 check

This example uses the public, anonymized [FAR-Trans release](https://researchdata.gla.ac.uk/1658/) (CC BY 4.0) and the saved CBF ranking in `frozen_outputs/predictions.csv`. It is a teaching-split check, not an LLM experiment or the final project benchmark.

- User: `5087D23F907F90F3DDD1`.
- Cutoff: `2021-06-01`; future window: `[2021-06-01, 2021-07-01)`.
- Previously purchased assets may be recommended again.
- The eligible future-purchase set contains one asset: `GRS003003035`.
- Raw `transactions.csv`, physical line **85882** including the header: transaction `9049523`, `Buy`, `2021-06-08`, asset `GRS003003035` for this user.
- The ten CBF prediction records are physical lines **2732–2741** of the frozen predictions file.

| Rank | ISIN | Binary hit | DCG contribution |
|---:|---|---:|---:|
| 1 | `GRS117123000` | 0 | 0 |
| 2 | `GRS118003003` | 0 | 0 |
| 3 | `GRS224003004` | 0 | 0 |
| 4 | `GRS314003005` | 0 | 0 |
| 5 | `GRS371113002` | 0 | 0 |
| 6 | `GRS385113006` | 0 | 0 |
| 7 | `GRS438003006` | 0 | 0 |
| 8 | `GRS469003024` | 0 | 0 |
| 9 | `100974271034` | 0 | 0 |
| 10 | `GRS003003035` | 1 | `1 / log2(11)` |

With binary relevance, `DCG@10 = sum(hit_r / log2(r + 1))`. The sole hit is at rank 10:

```text
DCG@10  = 1 / log2(11) = 0.2890648263178879
IDCG@10 = 1 / log2(2)  = 1
nDCG@10 = DCG / IDCG   = 0.2890648263178879
```

The ideal ranking places that one relevant asset first. The result matches `frozen_outputs/per_user.csv`, physical line **275**, which stores `relevant_count=1` and `ndcg_at_10=0.2890648263178879`.

## Scope of verification

The student independently recomputed the arithmetic (rounded result `0.2890648263`) and visually matched the asset ID at position 10 in displayed purchase/ranking extracts. She did not independently load the complete raw CSVs or establish completeness of the future-purchase set. The raw-record match, eligibility and complete truth-set checks were performed by AI-assisted validation. This distinction is retained rather than presenting agent checks as personal work.

The standard-library reproducer can rederive eligible future Buy labels and verify all 400 frozen user/method scores with `--verify-only`; see the member README for commands and data provenance.
