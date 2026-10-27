# Three-stock dashboard

React, TypeScript, Vite, Tailwind, shadcn/ui and Lucide. Four views: Overview, News & sentiment, Price comparison and Market data. The primary comparison uses the actual combined FinBERT + price Ridge model, with dollar/percentage errors and two baselines.

The browser reads `/api/live` from the local Python service; there are no API keys in the frontend. Build and serve from the project root:

```bash
cd website
npm ci
npm test
npm run build
cd ..
.venv/bin/python -m sentiment_lab.live serve --port 8765
```

To develop, keep the Python service on 8765 and run `npm run dev` here. Vite proxies `/api` to Python. Collection and model commands are described in the [project README](../README.md). A plain static HTTP server cannot provide the data API.

The icon-only sun/moon control saves the selected theme. Theme colors transition together, GSAP animates short entrances, and Anime.js draws the price chart. Reduced-motion preferences are respected. Old research snapshots, reports and archive downloads are no longer used.
