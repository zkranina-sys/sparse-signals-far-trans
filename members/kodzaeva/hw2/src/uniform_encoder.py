"""Post-hoc non-LLM mechanism control, selected after primary results.

Set all retained history assets to equal weights within the predeclared 4+
group; preserve count weighting elsewhere. No model or new information.
"""

def encode(context):
    history = context['history']
    return {item['id']: 1 if len(history) >= 4 else item['buy_count']
            for item in history}
