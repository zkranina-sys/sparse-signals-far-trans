"""Post-hoc diagnostic control: square-root purchase-count weights in4+.

Same fields, scorer, cohort and eligibility; no LLM and no fitted parameters.
"""
import math


def encode(context):
    history=context['history']
    return {item['id']:(math.sqrt(item['buy_count']) if len(history)>=4 else item['buy_count'])
            for item in history}
