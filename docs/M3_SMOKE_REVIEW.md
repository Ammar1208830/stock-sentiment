# M3 authored-example review

Reviewed on 26 October 2026 using actual local CPU FinBERT inference. These six texts were authored for the project and are not sourced financial events, an independent labeled dataset, or a stock-direction test. Expected labels are illustrative human interpretations.

Model: `ProsusAI/finbert` at `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`. Runtime: Torch 2.14.1, Transformers 4.57.6, CPU float32. Full text and scores are in the local `artifacts/m3-smoke/scores.jsonl` output and the packaged `sentiment_smoke.jsonl` fixture.

| Example | Expected | Model label | P(positive) | P(negative) | P(neutral) | Review |
| --- | --- | --- | --- | --- | --- | --- |
| Record profits and higher forecast | Positive | Positive | 0.953 | 0.021 | 0.026 | Direction of stated earnings news recognized |
| Substantial loss and lower forecast | Negative | Negative | 0.007 | 0.972 | 0.021 | Explicit loss/forecast language recognized |
| Results scheduled for Tuesday | Neutral | Neutral | 0.019 | 0.069 | 0.912 | Factual scheduling announcement recognized |
| Revenue declines 30%, weaker demand | Negative | Negative | 0.009 | 0.976 | 0.016 | Explicit deterioration recognized |
| Earnings beat, revenue rises | Positive | Positive | 0.957 | 0.021 | 0.022 | Explicit financial improvement recognized |
| “Great, another quarter of losses…” | Negative | Neutral | 0.089 | 0.390 | 0.521 | Sarcasm example does not match expected negative label |

Probabilities are rounded, so displayed values may not sum exactly to one. Five straightforward examples match the expectations. The sarcastic example shows a relevant limitation for investor discussions: conflicting surface wording and implied meaning can change the predicted label. Its negative probability remains substantial, which is another reason to retain the full probability vector rather than only a hard label.

No tuning or special-case correction was applied after observing this error. Do not interpret five matches out of six authored examples as a general sentiment accuracy estimate or as evidence about future stock movement. Real-source error review remains pending until permitted text data is available.
