# Stock Sentiment Lab frontend

React + TypeScript + Vite dashboard, with actual [shadcn/ui](https://github.com/shadcn-ui/ui) components, Tailwind CSS and [Lucide](https://github.com/lucide-icons/lucide) icons. The custom SVG logo represents a price/sentiment signal. The Python research pipeline is independent of the frontend.

## Develop

Use Node.js 22.19+ and npm. From this directory:

```bash
npm ci
npm run dev
```

Open the URL printed by Vite. To verify and create a static build:

```bash
npm test
npm run build
npm run preview
```

The build in `dist/` also works with the existing project-root command:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory website/dist
```

## Research snapshot

The dashboard reads `public/data.json`. This saved local research snapshot is intentionally excluded from Git, along with datasets, reports and trained models. A fresh clone shows an explicit empty state rather than fabricated results. Use **Open research snapshot** to choose a compatible JSON snapshot; it is parsed locally in the browser. With an existing snapshot, the same action is in **Export data**.

For the built-in local snapshot, copy the saved dashboard JSON to `public/data.json`. Optional local `public/report.html` and `public/handoff.zip` enable the report/download actions. The build checks whether those files exist, so missing assets are not shown as working links. Rebuild after adding or changing files. Refresh M6 results in an existing snapshot from the project root with `python3 website/export_dashboard.py`.

## Included behavior

- Overview, market data, model benchmarks, final results and research pipeline.
- Equity and date-range controls, historical chart hover, and price CSV export.
- Separate held-out/validation tabs, family/feature/search filters, baseline toggle and accessible model dialogs.
- Correct partition labels in metrics exports; overview stays on held-out results independently of the benchmark tab selection.
- Actual saved uncertainty intervals and the source-data limitation remain visible. No live quotes, invented sentiment scores, or investment recommendations.
- Mobile navigation, keyboard-accessible Radix controls and reduced-motion support.
- Icon-only sun/moon toggle, saved light/dark preference, and muted mint, blue, violet and amber accents. The initial theme follows the operating system until a choice is saved.
- GSAP handles short, staggered page/card entrances using `@gsap/react`. Anime.js draws the price line and fades its area when the ticker or date range changes. The libraries control separate elements; scopes are reverted on changes/unmount, and both respond to reduced-motion preferences.

Components live in `src/components/ui/` and are copied from the official shadcn registry with its CLI. `components.json` preserves the setup for adding more. Lucide icons are imported individually from `lucide-react`. Third-party license notices are in `THIRD_PARTY_NOTICES.md`; exact dependency versions are recorded in `package-lock.json`.

The frontend is previewed locally. No Sites deployment is part of this change.
