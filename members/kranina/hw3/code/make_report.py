"""Build the HW3 draft PDF from measured outputs only; no invented numbers."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(r'C:\Users\555\projects\hw3')
HW2_MANIFEST = Path(r'C:\Users\555\projects\hw2\outputs\full_v1\run_manifest.json')


def f4(value):
    return '-' if value is None else f'{value:.4f}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT/'outputs'/'dev_all_warm')
    p.add_argument('--coverage', type=Path, default=ROOT/'outputs'/'dev_all_warm'/'coverage.json')
    p.add_argument('--errors', type=Path, default=ROOT/'outputs'/'error_analysis'/'report.json')
    p.add_argument('--cache-check', type=Path, default=ROOT/'outputs'/'cache_check'/'report.json')
    p.add_argument('--manual', type=Path, default=ROOT/'outputs'/'manual_ndcg_check.json')
    p.add_argument('--output', type=Path, default=ROOT/'hw3_draft_report.pdf')
    p.add_argument('--name', default='Zlata Kranina')
    p.add_argument('--team', default='Sparse Signals')
    p.add_argument('--repository', default='https://github.com/zkranina-sys/sparse-signals-far-trans (members/kranina)')
    p.add_argument('--commit', default='not created: code and outputs are recorded in the exported session')
    p.add_argument('--bootstrap', type=Path, default=ROOT/'outputs'/'dev_all_warm'/'bootstrap.json')
    p.add_argument('--posthoc', type=Path, default=ROOT/'outputs'/'posthoc_infer_missing'/'metrics.json')
    p.add_argument('--manual-labels', type=Path, default=ROOT/'outputs'/'manual_label_25_summary.json')
    a = p.parse_args()
    metrics = json.loads((a.results/'metrics.json').read_text(encoding='utf-8'))
    manifest = json.loads((a.results/'run_manifest.json').read_text(encoding='utf-8'))
    hw2 = json.loads(HW2_MANIFEST.read_text(encoding='utf-8'))
    coverage = json.loads(a.coverage.read_text(encoding='utf-8')) if a.coverage.exists() else {}
    errors = json.loads(a.errors.read_text(encoding='utf-8')) if a.errors.exists() else {}
    cache = json.loads(a.cache_check.read_text(encoding='utf-8')) if a.cache_check.exists() else {}
    manual = json.loads(a.manual.read_text(encoding='utf-8')) if a.manual.exists() else {}
    bootstrap = json.loads(a.bootstrap.read_text(encoding='utf-8')) if a.bootstrap.exists() else {}
    posthoc = json.loads(a.posthoc.read_text(encoding='utf-8')) if a.posthoc.exists() else {}
    manual_labels = json.loads(a.manual_labels.read_text(encoding='utf-8')) if a.manual_labels.exists() else {}
    pdfmetrics.registerFont(TTFont('Arial', r'C:\Windows\Fonts\arial.ttf'))
    pdfmetrics.registerFont(TTFont('Arial-Bold', r'C:\Windows\Fonts\arialbd.ttf'))
    ss = getSampleStyleSheet()
    for key in ('Normal', 'Title', 'Heading1', 'Heading2'):
        ss[key].fontName = 'Arial'
    ss['Title'].fontName = 'Arial-Bold'
    ss['Heading1'].fontName = 'Arial-Bold'
    ss['Heading2'].fontName = 'Arial-Bold'
    ss['Normal'].fontSize = 9.2
    ss['Normal'].leading = 12.2
    story = []
    def para(text, style='Normal'):
        story.append(Paragraph(text, ss[style]))
    def head(text, level=1):
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(text, ss[f'Heading{level}']))
    para('Does Keyword Filling Match LLM Field Inference in a Financial Asset Encoder? Development Evidence from FAR-Trans', 'Title')
    para(f'{a.name} | HSE University | Team: {a.team}<br/>Repository: {a.repository}<br/>Experiment commit: {a.commit}')
    head('Abstract')
    para('We continue a FAR-Trans study of an LLM-as-Encoder: asset descriptions are produced by an open-weight model and embedded with TF-IDF before classical recommendation scoring. The research question is whether filling missing sector, industry and region fields from the asset name improves nDCG@10, and whether a frozen general keyword rule gives the same benefit as model inference. All results below are development results, not official benchmark scores. The cached HW2 descriptions are reused unchanged; the protocol and rule were frozen before comparison. We report all users and the subgroup with future purchases, four baselines, original-text CBF, p_para, p_infer and the keyword-rule ablation, together with failure analysis and a hand-checkable case.')
    head('Introduction')
    para('FAR-Trans contains retail investment transactions and asset metadata. In the eligible universe, sector/industry fields are often empty although the asset name is informative. The HW2 demo of 100 warm users suggested a p_infer advantage but the bootstrap interval included zero, so the effect was not established. This homework tests the same question on all warm users and asks whether the model adds anything beyond simple keywords. We use the LLM strictly as an Encoder: it changes asset text, not the ranking stage.')
    head('Related Work')
    para('FAR-Trans provides the dataset and eleven classical baselines [1], [2]. LLM-Rec introduced prompt-based item-description enrichment and showed gains with simple recommenders [3]. The course encoder adapts that idea to FAR-Trans with paraphrase and TF-IDF [4]. Qwen3 is the open-weight model family used for generation [5]. Our change is the comparison of model-inferred fields with a keyword rule under one frozen evaluation protocol.')
    head('Method')
    para(f"Universe: {manifest.get('users_count')} warm users and {coverage.get('assets_total', 'all')} eligible assets, cutoff {manifest.get('protocol', {}).get('cutoff')}, window ending {manifest.get('protocol', {}).get('end_exclusive')}, candidate count {manifest.get('protocol', {}).get('candidate_count')}, k=10, repeat purchases allowed. Protocol hash: {manifest.get('protocol_hash')}; this differs from the HW2 demo hash {manifest.get('hw2_protocol_hash')}. Users with empty truth enter the primary mean as zero; their count is reported separately.")
    para('Descriptions are the immutable HW2 cache. p_para paraphrases supplied facts and keeps unknown fields; p_infer replaces unknown fields by name-based guesses. The ablation p_rule appends the frozen name_keyword_fill_v1 values to p_para in the generated TF-IDF channel. Original facts form a separate TF-IDF view. User profiles are count-weighted sums of purchased-asset vectors; cosine scores produce the top ten, with popularity used for zero profiles. The rule dictionary was frozen before metrics and contains only general name tokens and region/country tokens, no per-asset entries.')
    para(f"Model: {hw2.get('model')}, revision {hw2.get('revision')}, greedy decoding, thinking off, 8192 context, at most 160 new tokens. HW2 generation time {f4(hw2.get('timings_seconds', {}).get('inference_seconds'))} s was measured on Colab/T4 and is separate from recommendation scoring. This development evaluation performs no new model inference; its measured scoring time is {f4(manifest.get('timings_seconds', {}).get('scoring'))} s.")
    head('Experiments')
    order = ['popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell', 'original_text_cbf', 'p_para', 'p_infer', 'p_rule']
    rows = [['Method', 'nDCG@10 all', 'nDCG@10 buyers', 'Buyers']]
    by = metrics['methods']
    for name in order:
        if name not in by:
            continue
        rows.append([name, f4(by[name]['all']['ndcg_at_10']), f4(by[name]['nonempty']['ndcg_at_10']),
                     str(by[name]['nonempty']['users'])])
    table = Table(rows, colWidths=[55*mm, 30*mm, 32*mm, 20*mm])
    table.setStyle(TableStyle([('FONTNAME', (0,0), (-1,-1), 'Arial'), ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#DCE6F1')), ('GRID', (0,0), (-1,-1), 0.3, colors.grey),
        ('ALIGN', (1,1), (-1,-1), 'RIGHT')]))
    story.append(table)
    para('Table 1. Development results, mean nDCG@10 across all users and across users with future purchases. Empty truth counts as zero in the primary column. Values are printed to four decimals.')
    rule_cov = coverage.get('by_field', {})
    para(f"Rule coverage: any field filled for {coverage.get('rule_any_field_assets')} assets; sector {rule_cov.get('sector', {}).get('rule_assets')}, industry {rule_cov.get('industry', {}).get('rule_assets')}, region {rule_cov.get('region', {}).get('rule_assets')}. The LLM coverage estimate is lexical and approximate; accuracy is checked by the student on 25 assets, not claimed here.")
    if cache:
        para(f"Cache verification: {cache.get('exact_matches')} of {cache.get('expected_answers')} new answers matched verbatim; mismatches {cache.get('mismatches')}, protocol problems {len(cache.get('protocol_problems', []))}. The verification timing is separate and does not alter HW2 generation time.")
    else:
        para('Cache verification with new model answers is pending; no claim about reproduction is made.')
    if manual:
        para(f"Hand-checkable case: user {manual.get('user_id')}, selected as the first user with future purchases in the frozen order. Gain, ideal gain and nDCG are listed per method in outputs/manual_ndcg_check.json.")
    b_all = bootstrap.get('all', {})
    b_group = bootstrap.get('group', {})
    mean_all = b_all.get('p_infer_minus_p_para_mean')
    mean_group = b_group.get('p_infer_minus_p_para_mean')
    ci_group = b_group.get('p_infer_minus_para_ci95', [])
    half_group = (ci_group[1] - ci_group[0]) / 2 if len(ci_group) == 2 else None
    ph = posthoc.get('bootstrap_vs_p_para', {})
    ph_all = ph.get('all', {}).get('p_infer_missing_minus_p_para', {})
    ph_buyers = ph.get('buyers', {}).get('p_infer_missing_minus_p_para', {})
    ph_group = ph.get('group', {}).get('p_infer_missing_minus_p_para', {})
    known = manual_labels.get('groups', {}).get('other', {})
    known_assets = known.get('assets')
    known_new = known.get('labels', {}).get('верно')
    head('Discussion')
    para('The controls in this comparison are original_text_cbf and p_para; p_rule is an ablation, not a '
         'second control. The frozen rule filled region on '
         f"{rule_cov.get('region', {}).get('rule_assets')} of {coverage.get('assets_total')} assets, "
         f"against sector on {rule_cov.get('sector', {}).get('rule_assets')} and industry on "
         f"{rule_cov.get('industry', {}).get('rule_assets')}, so it almost only adds region. The ablation "
         'isolates who filled the empty fields: the frozen rule or the LLM. If p_rule matches p_para, the '
         'added lexical material does not change recommendation rankings on this sample; if it changes '
         'scores without improving nDCG, enrichment alone is not evidence of better recommendations. '
         'Because asset names can carry model knowledge about specific assets, these development results '
         'cannot identify the source of that knowledge. The official evaluation set was not used; every '
         'number in this report is a development result on the June 2021 window, not an official '
         'benchmark score.')
    head('Limitations and Next Steps', level=2)
    para('The post-hoc p_infer_missing check, invented after seeing the results, splits the gain by '
         'asset group rather than by type of text change: p_infer_missing - p_para is '
         f"{f4(ph_all.get('mean'))} on all users, {f4(ph_buyers.get('mean'))} for buyers and "
         f"{f4(ph_group.get('mean'))} in the missing-field group, and all three bootstrap intervals "
         'include zero. The group interval from metrics.json is '
         f"[{ph_group['ci95'][0]:+.6f}, {ph_group['ci95'][1]:+.6f}]. "
         'This retains roughly one fifth of the all-user point estimate '
         f"({f4(mean_all)}) and about a third of the group estimate ({f4(mean_group)}). In the "
         f"student's 25-asset audit, {known_new} of {known_assets} assets with known fields also "
         'gained new information, so an asset-group split is not a split by change type. The precise '
         'limitation is that no separately preregistered phrasing-only control existed: we did not '
         'first measure how much nDCG changes from any other phrasing. Next steps: freeze a '
         'phrasing-only control in advance, measure the nDCG change from other phrasings first, and '
         'test it on data not yet examined. We do not run this now because the expected differences, '
         f"{f4(mean_all)}-{f4(mean_group)}, are comparable to the missing-field group interval "
         f"half-width about {f4(half_group)}, so one additional month would most likely not decide "
         'the question. No new instruction, protocol or run was created for this note.')
    para(f"Failure analysis: {errors.get('invalid_count', 'not measured')} invalid cached outputs, {errors.get('json_echo_count', 'not measured')} JSON echoes and {errors.get('infer_equals_para_count', 'not measured')} assets whose p_infer equals p_para. Repairs and fallbacks were recorded explicitly; no failed answer was deleted. Cases where ranking did not change are retained in outputs/dev_all_warm/unchanged_cases.jsonl.")
    head('AI Usage Disclosure')
    para('The assistant wrote code, prepared the frozen rule and protocol, ran local evaluation, checked references and drafted this text in the exported OpenCode session. The student made the research decisions, approved the runs, will label 25 generated descriptions and will compute one nDCG value by hand. No AI-generated citation was accepted without checking the original source; FAR-Trans data were not committed.')
    head('References')
    refs = [
        '[1] J. Sanz-Cruzado, N. Droukas, and R. McCreadie, "FAR-Trans: An investment dataset for financial asset recommendation," in Proc. IJCAI-2024 Workshop on Recommender Systems in Finance (Fin-RecSys), 2024. [Online]. Available: https://arxiv.org/abs/2407.08692',
        '[2] University of Glasgow, "FAR-Trans: An investment dataset for financial asset recommendation" (dataset), 2024, doi: 10.5525/gla.researchdata.1658. [Online]. Available: https://researchdata.gla.ac.uk/1658/',
        '[3] H. Lyu et al., "LLM-Rec: Personalized recommendation via prompting large language models," in Findings of NAACL 2024, pp. 583-612, doi: 10.18653/v1/2024.findings-naacl.39.',
        '[4] S. Jin, "RecSys-LLMs: FAR-Trans starter and LLM encoder/ranker baselines," GitHub repository, commit f685086. [Online]. Available: https://github.com/dryjins/RecSys-LLMs',
        '[5] Qwen Team, "Qwen3 technical report," arXiv:2505.09388, 2025. [Online]. Available: https://arxiv.org/abs/2505.09388',
    ]
    for r in refs:
        para(r)
    doc = SimpleDocTemplate(str(a.output), pagesize=A4, leftMargin=16*mm, rightMargin=16*mm,
                            topMargin=14*mm, bottomMargin=14*mm, title='HW3 draft report')
    doc.build(story)
    print('wrote', a.output, 'bytes', a.output.stat().st_size)


if __name__ == '__main__':
    main()