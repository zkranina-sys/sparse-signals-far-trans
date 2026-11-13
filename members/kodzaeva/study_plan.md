# Individual Study Plan: LLM-Based History Reweighting

**Author:** Maria Kodzaeva. **Team:** Sparse Signals. **Stage:** Session 1 / Define.

## Research question and importance

Does LLM reweighting of observed historical Buy assets improve nDCG@10 over the unchanged count-weighted CBF scorer for FAR-Trans users with at least four distinct retained assets bought before the cutoff? The intervention is relative history weights, the comparator is the provided CBF, the outcome is ranking accuracy and the population is a predeclared historical subgroup. The LLM intervention is planned, not implemented.

The question isolates profile construction from changing item features and candidate reranking. A recommender product owner could use a positive, null or negative finding to decide whether an extra profile-generation stage warrants its coverage, latency, cost and failure handling. No profit, suitability or user-acceptance effect is claimed.

## Related studies and bounded gap

| Study | Relevant method and protocol | Connection and limitation |
|---|---|---|
| Sanz-Cruzado, Droukas and McCreadie, FAR-Trans (2024), IJCAI Fin-RecSys Workshop | Financial transactions/metadata; 61 temporal settings, six-month horizons, nDCG@10 and ROI@10. | Dataset and protocol reference. The paper excludes previously bought assets; the course permits repeats, so scores are not directly comparable. |
| Xi et al., KAR (2023), arXiv:2306.10933v4 | LLM factorized preference reasoning plus item knowledge; encoded vectors and a learned adaptor. MovieLens-1M/Amazon-Books, CTR and reranking evaluation. | Nearest mechanism, but changes more than weights. It does not establish a gain for our fixed-scorer financial subgroup. |
| Li et al., Recformer (2023), KDD | Trained textual sequential representations, pretraining and finetuning; Amazon leave-one-out/full-item evaluation. | An architectural alternative rather than a frozen prompted weight generator. We do not reproduce its training or transfer its gains. |

Within these reviewed studies we did not find the exact proposed weights-only comparison under this FAR-Trans condition. This is a bounded review statement, not a global novelty claim.

## Controlled method

The existing CBF aggregates normalized categorical asset vectors with `buy_count` weights. The proposed open-weight `Qwen/Qwen2.5-3B-Instruct` Encoder (planned revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`) would return finite nonnegative weights only for observed historical assets. Restoring `buy_count` is the direct ablation. A common positive rescaling cannot change the normalized profile; relative weights must change.

The model receives counts and the same five CBF attributes: `assetCategory`, `assetSubCategory`, `sector`, `industry`, `marketID`. Temporary history keys replace IDs; names, risk-profile fields, future metadata and candidate descriptions are excluded from this first prompt. Invalid/missing/all-zero weights fall back to the count control and are counted as failures. Both arms use the same users, information, cutoff, eligible universe, repeat policy, scoring and fallback.

Initial generation plan: temperature 0.2, top_p 0.9, maximum 512 output tokens, three repeats. Preserve full retained history in both arms; apply any necessary truncation identically. Pilot three users, then cap the teaching experiment at 111 profiles and 60 minutes of inference. This is a proposed budget, not measured throughput. Model licence and hardware suitability will be checked before implementation.

## Evaluation and evidence

The actual classical teaching run and its reproducible files are described in [README.md](README.md). The 4+ subgroup is defined from past history, not future outcomes: 37 of 100 users, with 10 nonempty future truth sets. The primary research contrast is mean nDCG@10 over all 37, including 27 empty truth users. Retain the common all-100 mean separately; the subgroup nonempty-10 mean is diagnostic. Empty truth scores zero. Keep all four provided baselines.

Record generation/whole-run runtime separately, compute or token cost, failed outputs and fallbacks. Save three independent inference outputs and report the mean/range of paired differences. The [manual real-user example](evidence/manual_ndcg_check.md) has one relevant asset at rank 10, DCG=0.2890648263, IDCG=1 and nDCG=0.2890648263.

## Team discussion and individual contribution

Zlata and I shared our proposed methods, and I outlined the common evaluation and control setup for planned comparisons. The shared protocol does not imply identical component controls: my profile component uses count-weighted CBF, Zlata's retrieval studies its content-based filling control, and Vladislav's reranker compares against an unchanged candidate order.

My assigned implementation responsibility is the history-weight Encoder. The first weights-only experiment is independent of unfinished team components. For later integration I propose a fixed union of latest_buy Top-10 and count-CBF Top-50, scored by `0.5 / latest_buy_rank + 0.5 * CBF_cosine`, with zero recency outside the latest_buy list. The pool, recency and blend are identical in the count-weight and LLM-weight arms. Filling empty slots alone cannot affect 10+ histories; the blend supplies a route for weights to change membership/order before a separately ablated team reranker. This is a proposed extension, not implemented or confirmed as a team agreement.

## Verification scope and limitations

The student personally checked the cited method/evaluation sections, subgroup arithmetic and the nDCG arithmetic/asset match in displayed extracts. Complete raw CSV processing, truth-set completeness, hashes and all-user recomputation were AI-assisted agent checks. The source code and frozen results are preserved with hashes. Native AI-session logs and the PDF are submitted separately, not included in this public folder.

Only ten subgroup users have future labels, limiting the pilot's evidence. As-of filtering does not prove unknown collection times. Estimated/missing profile fields are excluded from the first comparison. Pseudonymous histories should not be reidentified. Temporary keys and categorical inputs reduce direct asset lookup but cannot prove absence of pretraining contamination. LLM reasoning or an LLM judge is not ground truth. RAG and finetuning are optional later extensions.

## Primary sources

1. [FAR-Trans paper](https://arxiv.org/abs/2407.08692), relevant §§3 and 5.
2. [KAR, v4](https://arxiv.org/abs/2306.10933v4), relevant §§4–5.
3. [Recformer, KDD 2023](https://cseweb.ucsd.edu/~jmcauley/pdfs/kdd23b.pdf), relevant §§2–3, DOI 10.1145/3580305.3599519.
4. [Pinned teaching starter](https://github.com/dryjins/RecSys-LLMs/tree/51337032503c56e3a28ef646a24aa016782e1bd3/project/far-trans).
5. [Qwen model revision](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/tree/aa8e72537993ba99e69dfaafa59ed015b17504d1).
