# Stock Sentiment Lab dashboard

A static dashboard of the saved project snapshot. It displays historical vendor prices, validation and frozen held-out metrics, paired uncertainty, source readiness and project milestones. It does not fetch live quotes or retrain models. The Final results view links the local report and reproducible handoff.

From the main project directory:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory website/dist
```

Open <http://127.0.0.1:8765>. Stop the server with Ctrl+C when finished.

The browser reads `dist/data.json`, containing historical prices and development summaries/metrics from the local M2–M5 artifacts. No API keys, raw social text, test outcomes or model weights are included.

The chart supports equity and date-range selection. Model comparisons can be filtered; each row opens its parameters and confusion matrix. Export buttons download validation metrics or vendor price data locally.

Hosting status: a private Sites project was registered, but no project data or dashboard source was uploaded. Automatic approval review blocked external synchronization because the request to view the website did not explicitly authorize uploading the project-derived price snapshot and validation metrics to Sites. Local preview remains available. Publishing requires that specific approval.
