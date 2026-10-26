import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import {
  Activity,
  ArrowDown,
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  ChartNoAxesCombined,
  Check,
  ChevronRight,
  CircleHelp,
  Database,
  ExternalLink,
  FileText,
  FlaskConical,
  GitBranch,
  CodeXml,
  Layers,
  LayoutDashboard,
  ListChecks,
  LockKeyhole,
  Menu,
  MessageSquare,
  Newspaper,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  TrendingUp,
  Upload,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  companyNames,
  dateLabel,
  download,
  featureNames,
  filterMetrics,
  formatMetric,
  metricsCsv,
  metricsFor,
  modelName,
  pricesCsv,
  usd,
  validateSnapshot,
  type Metric,
  type Partition,
  type Price,
  type ResearchData,
  type View,
} from "@/lib/research";
import { cn } from "@/lib/utils";

const navigation = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "market", label: "Market data", icon: ChartNoAxesCombined },
  { id: "models", label: "Model benchmarks", icon: SlidersHorizontal },
  { id: "results", label: "Final results", icon: FlaskConical },
  { id: "pipeline", label: "Research pipeline", icon: GitBranch },
] as const;
const pageCopy: Record<View, [string, string]> = {
  overview: [
    "Your research, in focus.",
    "A clearer picture of the market. A more honest view of the signal.",
  ],
  market: [
    "The market, in context.",
    "Five liquid equities. One shared historical timeline.",
  ],
  models: [
    "Every model needs a benchmark.",
    "Compare the evidence, inspect the parameters, and question the result.",
  ],
  results: [
    "Evidence over expectation.",
    "The frozen result, its uncertainty, and what we can actually conclude.",
  ],
  pipeline: [
    "From language to evidence.",
    "A reproducible experiment, with the timeline kept intact.",
  ],
};
function validView(value: string): value is View {
  return navigation.some((item) => item.id === value);
}

function Brand({ onClick }: { onClick: () => void }) {
  return (
    <a
      href="#overview"
      onClick={onClick}
      className="brand"
      aria-label="Sentiment Lab overview"
    >
      <img src="logo.svg" alt="" width="42" height="42" />
      <span>
        sentiment<span className="brand-light">lab</span>
        <small>RESEARCH WORKSPACE</small>
      </span>
    </a>
  );
}
function Sidebar({
  view,
  navigate,
}: {
  view: View;
  navigate: (view: View) => void;
}) {
  return (
    <div className="sidebar-inner">
      <Brand onClick={() => navigate("overview")} />
      <div className="nav-caption">WORKSPACE</div>
      <nav aria-label="Main navigation">
        {navigation.map((item) => (
          <button
            key={item.id}
            onClick={() => navigate(item.id)}
            className={cn("nav-link", view === item.id && "is-active")}
            aria-current={view === item.id ? "page" : undefined}
          >
            <item.icon size={18} strokeWidth={1.7} />
            <span>{item.label}</span>
            {view === item.id && <span className="nav-active-dot" />}
          </button>
        ))}
      </nav>
      <div className="sidebar-study">
        <div className="study-icon">
          <Activity size={17} />
        </div>
        <div>
          <strong>Stock direction study</strong>
          <span>5 equities · next-session forecast</span>
        </div>
      </div>
      <div className="sidebar-bottom">
        <div className="sidebar-note">
          <FlaskConical size={22} strokeWidth={1.5} />
          <h3>Curiosity, with a control.</h3>
          <p>Does financial sentiment add value beyond price alone?</p>
          <button onClick={() => navigate("results")}>
            Explore the findings <ArrowUpRight size={14} />
          </button>
        </div>
        <div className="sidebar-meta">
          <span className="workspace-avatar">SL</span>
          <div>
            <strong>Stock Sentiment Lab</strong>
            <span>Research prototype · v0.8</span>
          </div>
          <LockKeyhole size={13} />
        </div>
      </div>
    </div>
  );
}
function StatusBadge({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: "green" | "amber" | "neutral";
}) {
  return (
    <Badge variant="outline" className={cn("status-badge", `status-${tone}`)}>
      <span className="status-dot" />
      {children}
    </Badge>
  );
}
function SectionHeading({
  title,
  description,
  action,
  tag,
}: {
  title: string;
  description?: string;
  action?: React.ReactNode;
  tag?: string;
}) {
  return (
    <CardHeader className="section-heading">
      <div>
        {tag && <div className="section-kicker">{tag}</div>}
        <CardTitle>
          <h2>{title}</h2>
        </CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </div>
      {action}
    </CardHeader>
  );
}
function MetricCards({ data }: { data: ResearchData }) {
  const cards = [
    {
      name: "Price observations",
      value: Object.values(data.prices).reduce(
        (sum, rows) => sum + rows.length,
        0,
      ),
      caption: `${Object.keys(data.prices).length} liquid US equities`,
      icon: ChartNoAxesCombined,
      tag: "Historical",
    },
    {
      name: "Model-ready rows",
      value: data.features.retained_rows,
      caption: "Chronologically aligned",
      icon: Layers,
      tag: "Leakage checked",
    },
    {
      name: "Frozen models",
      value: data.summary.trained_models,
      caption: "2 classifiers · 5 feature sets",
      icon: LockKeyhole,
      tag: "Protocol frozen",
    },
    {
      name: "Held-out observations",
      value: data.evaluation?.test_rows ?? data.summary.holdout_rows_reserved,
      caption: data.evaluation
        ? `${data.evaluation.distinct_test_sessions} unseen trading sessions`
        : "Reserved for evaluation",
      icon: FlaskConical,
      tag: data.evaluation ? "Evaluated" : "Reserved",
    },
  ];
  return (
    <section className="metric-cards" aria-label="Research summary">
      {cards.map((card) => (
        <Card key={card.name} className="metric-card">
          <CardContent>
            <div className="metric-card-top">
              <span>{card.name}</span>
              <div className="stat-icon">
                <card.icon size={17} />
              </div>
            </div>
            <div className="stat-value">{card.value.toLocaleString()}</div>
            <div className="stat-caption">
              <span>{card.caption}</span>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span tabIndex={0} aria-label={card.tag}>
                    <ShieldCheck size={13} />
                  </span>
                </TooltipTrigger>
                <TooltipContent>{card.tag}</TooltipContent>
              </Tooltip>
            </div>
          </CardContent>
        </Card>
      ))}
    </section>
  );
}
function PriceChart({ rows, ticker }: { rows: Price[]; ticker: string }) {
  const [hover, setHover] = useState<number | null>(null);
  useEffect(() => setHover(null), [rows]);
  const W = 920,
    H = 285,
    left = 4,
    right = 62,
    top = 20,
    bottom = 34;
  const low = Math.min(...rows.map((row) => row.close)),
    high = Math.max(...rows.map((row) => row.close)),
    pad = (high - low) * 0.15 || 1;
  const min = low - pad,
    max = high + pad;
  const x = (index: number) =>
    left + (index / Math.max(1, rows.length - 1)) * (W - left - right);
  const y = (value: number) =>
    top + ((max - value) / (max - min)) * (H - top - bottom);
  const path = rows
    .map(
      (row, index) => `${index === 0 ? "M" : "L"}${x(index)},${y(row.close)}`,
    )
    .join(" ");
  const selected = hover == null ? null : rows[hover];
  return (
    <div className="chart-wrap">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="price-chart"
        role="img"
        aria-label={`${ticker} historical closing prices from ${rows[0].date} to ${rows.at(-1)!.date}`}
        onPointerMove={(event) => {
          const bounds = event.currentTarget.getBoundingClientRect();
          const position = ((event.clientX - bounds.left) / bounds.width) * W;
          setHover(
            Math.min(
              rows.length - 1,
              Math.max(
                0,
                Math.round(
                  ((position - left) / (W - left - right)) * (rows.length - 1),
                ),
              ),
            ),
          );
        }}
        onPointerLeave={() => setHover(null)}
      >
        <defs>
          <linearGradient id="price-area" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#2d8e6e" stopOpacity=".17" />
            <stop offset="100%" stopColor="#2d8e6e" stopOpacity=".005" />
          </linearGradient>
        </defs>
        {Array.from({ length: 5 }, (_, i) => {
          const value = min + ((max - min) * i) / 4;
          return (
            <g key={i}>
              <line
                x1={left}
                x2={W - right + 8}
                y1={y(value)}
                y2={y(value)}
                stroke="#e8ede8"
                strokeDasharray="3 5"
              />
              <text
                x={W - right + 18}
                y={y(value) + 4}
                fill="#69775f"
                fontSize="11"
              >
                ${value.toFixed(0)}
              </text>
            </g>
          );
        })}
        <path
          d={`${path} L${x(rows.length - 1)},${H - bottom} L${left},${H - bottom} Z`}
          fill="url(#price-area)"
        />
        <path
          d={path}
          fill="none"
          stroke="#2b8061"
          strokeWidth="2.4"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {[0, 1, 2, 3, 4].map((i) => {
          const index = Math.round(((rows.length - 1) * i) / 4);
          return (
            <text
              key={i}
              x={x(index)}
              y={H - 5}
              textAnchor={i === 0 ? "start" : i === 4 ? "end" : "middle"}
              fill="#69775f"
              fontSize="11"
            >
              {new Date(rows[index].date + "T12:00:00Z").toLocaleDateString(
                "en-US",
                { month: "short", year: "2-digit" },
              )}
            </text>
          );
        })}
        {selected && hover != null ? (
          <g>
            <line
              x1={x(hover)}
              x2={x(hover)}
              y1={top}
              y2={H - bottom}
              stroke="#9ab3a1"
              strokeDasharray="4 4"
            />
            <circle
              cx={x(hover)}
              cy={y(selected.close)}
              r="5"
              fill="#236a50"
              stroke="white"
              strokeWidth="2.5"
            />
          </g>
        ) : (
          <circle
            cx={x(rows.length - 1)}
            cy={y(rows.at(-1)!.close)}
            r="4"
            fill="#236a50"
            stroke="white"
            strokeWidth="2"
          />
        )}
      </svg>
      {selected && (
        <div className="chart-tooltip">
          <span>{dateLabel(selected.date)}</span>
          <strong>{usd(selected.close)}</strong>
        </div>
      )}
    </div>
  );
}
function MarketChart({ data }: { data: ResearchData }) {
  const tickers = Object.keys(data.prices);
  const [ticker, setTicker] = useState(
    tickers.includes("AAPL") ? "AAPL" : tickers[0],
  );
  const [range, setRange] = useState(252);
  const rows = useMemo(
    () => data.prices[ticker].slice(-range),
    [data, ticker, range],
  );
  const last = rows.at(-1)!,
    change = (last.close / rows[0].close - 1) * 100;
  return (
    <Card className="market-chart-card">
      <SectionHeading
        title="Market overview"
        description="The price behind the prediction"
        action={
          <Badge variant="outline" className="quiet-badge">
            USD
          </Badge>
        }
      />
      <div className="ticker-tabs" aria-label="Select equity">
        {tickers.map((symbol) => (
          <button
            key={symbol}
            className={cn("ticker-button", symbol === ticker && "is-selected")}
            aria-pressed={symbol === ticker}
            onClick={() => setTicker(symbol)}
          >
            <span
              className={cn(
                "ticker-monogram",
                `ticker-${symbol.toLowerCase()}`,
              )}
            >
              {symbol[0]}
            </span>
            {symbol}
          </button>
        ))}
      </div>
      <div className="chart-summary">
        <div>
          <div className="company-label">
            {companyNames[ticker] ?? ticker}
            <span>NASDAQ</span>
          </div>
          <div className="stock-price">
            {usd(last.close)}
            <span className={cn("price-change", change < 0 && "is-negative")}>
              {change < 0 ? (
                <ArrowDown size={13} />
              ) : (
                <ArrowUpRight size={14} />
              )}
              {Math.abs(change).toFixed(2)}%
            </span>
          </div>
          <span className="period-label">
            {range === 9999
              ? "Across available history"
              : range === 252
                ? "Over the past trading year"
                : range === 63
                  ? "Over the past 3 trading months"
                  : "Over the past trading month"}
          </span>
        </div>
        <div className="range-buttons" aria-label="Chart date range">
          {[
            [21, "1M"],
            [63, "3M"],
            [252, "1Y"],
            [9999, "All"],
          ].map(([value, label]) => (
            <button
              key={value}
              aria-pressed={range === value}
              className={cn(range === value && "is-selected")}
              onClick={() => setRange(Number(value))}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      <PriceChart rows={rows} ticker={ticker} />
      <div className="chart-footer">
        <span>
          <i />
          Vendor closing price
        </span>
        <span>
          As of {dateLabel(data.asOf)}
          <span className="desktop-only"> · Historical, not live</span>
        </span>
      </div>
    </Card>
  );
}
function SourceReadiness({
  data,
  navigate,
}: {
  data: ResearchData;
  navigate: (view: View) => void;
}) {
  const sources = [
    {
      name: "Market prices",
      provider: "yfinance · OHLCV history",
      icon: TrendingUp,
      available: Object.keys(data.prices).length > 0,
    },
    {
      name: "Financial news",
      provider: "NewsAPI · historical headlines",
      icon: Newspaper,
      available: (data.features.sentiment_covered_rows.newsapi ?? 0) > 0,
    },
    {
      name: "Reddit discussions",
      provider: "Approved research discussion",
      icon: MessageSquare,
      available: (data.features.sentiment_covered_rows.reddit ?? 0) > 0,
    },
  ];
  const count = sources.filter((source) => source.available).length;
  return (
    <Card className="source-card">
      <SectionHeading
        title="Source readiness"
        description="Better evidence starts here"
        action={
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                size="icon"
                variant="ghost"
                aria-label="About source readiness"
              >
                <CircleHelp size={16} />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              Availability reflects the saved research snapshot.
            </TooltipContent>
          </Tooltip>
        }
      />
      <div className="readiness-summary">
        <strong>
          {count}
          <span>/ 3</span>
        </strong>
        <div>
          <span>sources available</span>
          <small>Coverage before conviction.</small>
        </div>
      </div>
      <div
        className="readiness-track"
        aria-label={`${count} of 3 sources available`}
      >
        {sources.map((source) => (
          <span
            key={source.name}
            className={cn(source.available && "is-ready")}
          />
        ))}
      </div>
      <div className="sources-list">
        {sources.map((source) => (
          <div className="source-row" key={source.name}>
            <div className={cn("source-icon", source.available && "is-ready")}>
              <source.icon size={18} strokeWidth={1.6} />
            </div>
            <div className="source-copy">
              <strong>{source.name}</strong>
              <small>{source.provider}</small>
            </div>
            <StatusBadge tone={source.available ? "green" : "amber"}>
              {source.available ? "Ready" : "Missing"}
            </StatusBadge>
          </div>
        ))}
      </div>
      <div className="source-note">
        <ShieldCheck size={16} />
        <p>
          Missing text stays missing.
          <br />
          Sentiment scores are never fabricated.
        </p>
      </div>
      <button className="card-link" onClick={() => navigate("pipeline")}>
        View research pipeline <ArrowRight size={15} />
      </button>
    </Card>
  );
}
function BenchmarksTable({
  rows,
  partition,
  inspect,
}: {
  rows: Metric[];
  partition: Partition;
  inspect: (metric: Metric, partition: Partition) => void;
}) {
  return (
    <Table className="benchmarks-table">
      <TableHeader>
        <TableRow>
          <TableHead>Model</TableHead>
          <TableHead>Feature set</TableHead>
          <TableHead className="number-cell">ROC-AUC</TableHead>
          <TableHead className="number-cell">F1-score</TableHead>
          <TableHead className="number-cell">Accuracy</TableHead>
          <TableHead className="inspect-column">
            <span className="sr-only">Inspect model</span>
          </TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((metric) => (
          <TableRow
            key={metric.model_id}
            onClick={() => inspect(metric, partition)}
            className="model-row"
          >
            <TableCell>
              <div className="model-cell">
                <span
                  className={cn(
                    "model-initial",
                    metric.family === "baseline" && "baseline-initial",
                  )}
                >
                  {metric.family === "logistic_regression"
                    ? "LR"
                    : metric.family === "random_forest"
                      ? "RF"
                      : "B"}
                </span>
                <span>
                  {modelName(metric)}
                  <small>
                    {metric.family === "baseline"
                      ? "Naive benchmark"
                      : "Frozen classifier"}
                  </small>
                </span>
              </div>
            </TableCell>
            <TableCell>
              <Badge
                variant="outline"
                className={cn(
                  "feature-badge",
                  metric.feature_set === "combined" && "combined-badge",
                )}
              >
                {featureNames[metric.feature_set] ?? metric.feature_set}
              </Badge>
            </TableCell>
            <TableCell className="number-cell strong-number">
              {formatMetric(metric.roc_auc)}
            </TableCell>
            <TableCell className="number-cell">
              {formatMetric(metric.f1)}
            </TableCell>
            <TableCell className="number-cell muted-number">
              {(metric.accuracy * 100).toFixed(1)}%
            </TableCell>
            <TableCell>
              <Button
                variant="ghost"
                size="icon"
                aria-label={`Inspect ${modelName(metric)}, ${featureNames[metric.feature_set] ?? metric.feature_set}, ${partition}`}
                onClick={(event) => {
                  event.stopPropagation();
                  inspect(metric, partition);
                }}
              >
                <ArrowUpRight size={16} />
              </Button>
            </TableCell>
          </TableRow>
        ))}
        {!rows.length && (
          <TableRow>
            <TableCell colSpan={6} className="empty-table">
              <Search size={22} />
              <strong>No matching models</strong>
              <span>Try another feature set or reset your filters.</span>
            </TableCell>
          </TableRow>
        )}
      </TableBody>
    </Table>
  );
}
function Overview({
  data,
  navigate,
  inspect,
}: {
  data: ResearchData;
  navigate: (view: View) => void;
  inspect: (metric: Metric, partition: Partition) => void;
}) {
  const partition = data.testMetrics ? "test" : "validation";
  const rows = [
    "logistic_regression:price_only",
    "logistic_regression:combined",
    "random_forest:price_only",
    "baseline:always_up",
  ]
    .map((id) => metricsFor(data, partition).find((row) => row.model_id === id))
    .filter((row): row is Metric => !!row);
  return (
    <>
      <div className="overview-grid">
        <MarketChart data={data} />
        <SourceReadiness data={data} navigate={navigate} />
      </div>
      <Card className="benchmark-preview">
        <SectionHeading
          title="A signal is only as good as its baseline."
          description={
            partition === "test"
              ? "Frozen held-out results · same observations, same threshold"
              : "Validation results · held-out metrics not available"
          }
          action={
            <Button
              variant="outline"
              size="sm"
              onClick={() => navigate("models")}
            >
              Explore models <ArrowUpRight size={14} />
            </Button>
          }
        />
        <BenchmarksTable rows={rows} partition={partition} inspect={inspect} />
        <div className="table-footer">
          <span>
            <span className="small-dot" />
            {partition === "test"
              ? "Held-out evaluation"
              : "Validation evaluation"}
          </span>
          <span>
            Sentiment contribution remains inconclusive
            <CircleHelp size={13} />
          </span>
        </div>
      </Card>
      <div className="principles-strip">
        {[
          {
            number: "01",
            title: "Respect the timeline",
            body: "Train first. Hold out the future.",
            icon: GitBranch,
          },
          {
            number: "02",
            title: "Measure the difference",
            body: "Price-only, sentiment, and naive baselines.",
            icon: SlidersHorizontal,
          },
          {
            number: "03",
            title: "Report what you find",
            body: "An inconclusive result is still a result.",
            icon: ShieldCheck,
          },
        ].map((item) => (
          <div key={item.number}>
            <span className="principle-number">{item.number}</span>
            <div>
              <strong>{item.title}</strong>
              <span>{item.body}</span>
            </div>
            <item.icon size={19} strokeWidth={1.5} />
          </div>
        ))}
      </div>
    </>
  );
}
function Sparkline({ rows }: { rows: Price[] }) {
  const values = rows.slice(-63).map((row) => row.close),
    low = Math.min(...values),
    high = Math.max(...values);
  return (
    <svg width="100" height="30" viewBox="0 0 100 30" aria-hidden="true">
      <polyline
        points={values
          .map(
            (value, index) =>
              `${(index / (values.length - 1)) * 100},${27 - ((value - low) / (high - low || 1)) * 24}`,
          )
          .join(" ")}
        fill="none"
        stroke="#418464"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}
function Market({ data }: { data: ResearchData }) {
  return (
    <>
      <Card>
        <SectionHeading
          title="Five equities. One shared timeline."
          description={`Vendor prices through ${dateLabel(data.asOf)} · USD`}
          action={
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                download("sentiment-lab-prices.csv", pricesCsv(data.prices))
              }
            >
              <ArrowDownToLine size={15} />
              Export prices
            </Button>
          }
        />
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Equity</TableHead>
              <TableHead>Last close</TableHead>
              <TableHead>3-month history</TableHead>
              <TableHead className="number-cell">Observations</TableHead>
              <TableHead>Date range</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {Object.entries(data.prices).map(([ticker, rows]) => (
              <TableRow key={ticker}>
                <TableCell>
                  <div className="model-cell">
                    <span
                      className={cn(
                        "ticker-monogram",
                        `ticker-${ticker.toLowerCase()}`,
                      )}
                    >
                      {ticker[0]}
                    </span>
                    <span>
                      {ticker}
                      <small>{companyNames[ticker] ?? ticker}</small>
                    </span>
                  </div>
                </TableCell>
                <TableCell className="strong-number">
                  {usd(rows.at(-1)!.close)}
                </TableCell>
                <TableCell>
                  <Sparkline rows={rows} />
                </TableCell>
                <TableCell className="number-cell">
                  {rows.length.toLocaleString()}
                </TableCell>
                <TableCell className="muted-number">
                  {dateLabel(rows[0].date)} — {dateLabel(rows.at(-1)!.date)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>
      <div className="two-columns">
        <InfoCard icon={Database} title="A clear availability assumption">
          Historical bars are available at close + 1 minute. Vendor history can
          include retrospective split adjustments and revisions.
        </InfoCard>
        <InfoCard icon={GitBranch} title="Corporate actions, accounted for">
          Split target sessions are excluded. Technical histories restart at
          splits and gaps; {data.features.dropped_rows} warmup or reset rows
          were removed.
        </InfoCard>
      </div>
    </>
  );
}
function Models({
  data,
  partition,
  setPartition,
  inspect,
}: {
  data: ResearchData;
  partition: Partition;
  setPartition: (partition: Partition) => void;
  inspect: (metric: Metric, partition: Partition) => void;
}) {
  const [family, setFamily] = useState("all"),
    [feature, setFeature] = useState("all"),
    [baselines, setBaselines] = useState(false),
    [query, setQuery] = useState("");
  const rows = filterMetrics(
    metricsFor(data, partition),
    family,
    feature,
    baselines,
    query,
  );
  return (
    <Card>
      <SectionHeading
        title="Model comparison"
        description="AUC measures ranking. F1 uses the frozen 0.5 classification threshold."
        action={<StatusBadge tone="green">Protocol frozen</StatusBadge>}
      />
      <Tabs
        value={partition}
        onValueChange={(value) => setPartition(value as Partition)}
      >
        <div className="model-tabs">
          <TabsList>
            <TabsTrigger value="test" disabled={!data.testMetrics}>
              Held-out test
            </TabsTrigger>
            <TabsTrigger value="validation">Validation</TabsTrigger>
          </TabsList>
          <span>{rows.length} comparisons</span>
        </div>
        <div className="model-filters">
          <div className="search-field">
            <Search size={15} />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search models…"
              aria-label="Search models"
            />
          </div>
          <Select value={family} onValueChange={setFamily}>
            <SelectTrigger aria-label="Model family">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All classifiers</SelectItem>
              <SelectItem value="logistic_regression">
                Logistic Regression
              </SelectItem>
              <SelectItem value="random_forest">Random Forest</SelectItem>
              <SelectItem value="baseline">Baselines</SelectItem>
            </SelectContent>
          </Select>
          <Select value={feature} onValueChange={setFeature}>
            <SelectTrigger aria-label="Feature set">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All feature sets</SelectItem>
              {Object.entries(featureNames)
                .filter(([key]) => key !== "matched_sample")
                .map(([key, name]) => (
                  <SelectItem key={key} value={key}>
                    {name}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
          <label className="baseline-switch">
            <Switch
              checked={baselines}
              onCheckedChange={setBaselines}
              aria-label="Include baselines"
            />
            Baselines
          </label>
          {(query || family !== "all" || feature !== "all" || baselines) && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setQuery("");
                setFamily("all");
                setFeature("all");
                setBaselines(false);
              }}
            >
              <X size={14} />
              Reset
            </Button>
          )}
        </div>
        {(["test", "validation"] as const).map((value) => (
          <TabsContent key={value} value={value}>
            <BenchmarksTable rows={rows} partition={value} inspect={inspect} />
          </TabsContent>
        ))}
      </Tabs>
      <div className="table-footer">
        <span>
          <LockKeyhole size={13} />
          {partition === "test"
            ? "Frozen test predictions · no refitting"
            : "Development validation · used for model selection"}
        </span>
        <span>Primary: combined LR vs. price-only LR</span>
      </div>
    </Card>
  );
}
function InfoCard({
  icon: Icon,
  title,
  children,
}: {
  icon: typeof Database;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Card className="info-card">
      <CardContent>
        <span className="info-icon">
          <Icon size={20} strokeWidth={1.6} />
        </span>
        <h3>{title}</h3>
        <p>{children}</p>
      </CardContent>
    </Card>
  );
}
function Results({
  data,
  assets,
}: {
  data: ResearchData;
  assets: { report: boolean; handoff: boolean };
}) {
  const evaluation = data.evaluation;
  if (!evaluation)
    return (
      <InfoCard
        icon={FlaskConical}
        title="Held-out evaluation is not available"
      >
        This snapshot contains development results. Final conclusions require a
        frozen held-out evaluation.
      </InfoCard>
    );
  return (
    <>
      <Card className="results-card">
        <div className="results-hero">
          <span className="result-symbol">
            <FlaskConical size={29} strokeWidth={1.5} />
          </span>
          <StatusBadge tone="amber">
            Research conclusion: inconclusive
          </StatusBadge>
          <h2>The question is still open.</h2>
          <p>
            {data.conclusion?.statement ??
              "The sentiment hypothesis requires eligible historical text."}
          </p>
        </div>
        <div className="result-metrics">
          <div>
            <span>Price-only LR · test AUC</span>
            <strong>
              {formatMetric(evaluation.primary_control_metrics.roc_auc, 4)}
            </strong>
            <small>
              {evaluation.test_rows.toLocaleString()} held-out observations
            </small>
          </div>
          <div>
            <span>Combined − price-only AUC</span>
            <strong>
              {formatMetric(evaluation.primary_delta_auc.estimate, 4)}
            </strong>
            <small>Primary paired comparison</small>
          </div>
          <div>
            <span>Paired 95% interval</span>
            <strong>
              [
              {evaluation.primary_delta_auc.interval
                .map((value) => formatMetric(value, 4))
                .join(", ")}
              ]
            </strong>
            <small>
              {evaluation.primary_delta_auc.valid_replicates.toLocaleString()}{" "}
              bootstrap replicates
            </small>
          </div>
        </div>
        <div className="result-explanation">
          <ShieldCheck size={18} />
          <p>
            Identical predictions with missing sentiment produce a zero
            interval. This does not establish that sentiment is useless—or
            valuable.
          </p>
        </div>
        <div className="report-actions">
          {assets.report && (
            <Button asChild>
              <a href="report.html" target="_blank" rel="noopener">
                Read the final report
                <ArrowUpRight size={15} />
              </a>
            </Button>
          )}
          {assets.handoff && (
            <Button asChild variant="outline">
              <a href="handoff.zip" download>
                <ArrowDownToLine size={15} />
                Reproducible handoff
              </a>
            </Button>
          )}
          {!assets.report && !assets.handoff && (
            <span className="asset-note">
              Report and handoff files are not included in this snapshot.
            </span>
          )}
        </div>
      </Card>
      <div className="two-columns">
        <InfoCard icon={Check} title="Software delivery complete">
          Collection contracts, FinBERT, chronological features, interpretable
          classifiers, baselines, paired uncertainty, and a reproducible final
          handoff.
        </InfoCard>
        <InfoCard icon={Newspaper} title="The next research step">
          Obtain permitted, correctly timestamped news and Reddit history. Audit
          its coverage and reserve a fresh holdout before a new confirmatory
          study.
        </InfoCard>
      </div>
      <Card>
        <SectionHeading
          title="Uncertainty, checked three ways"
          description="Paired moving-block bootstrap · primary Logistic Regression comparison"
        />
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Block length</TableHead>
              <TableHead>Role</TableHead>
              <TableHead className="number-cell">Δ AUC</TableHead>
              <TableHead className="number-cell">95% interval</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.uncertainty?.results.map((result) => {
              const comparison = result.comparisons.find(
                (item) => item.role === "primary",
              );
              if (!comparison) return null;
              return (
                <TableRow key={result.block_sessions}>
                  <TableCell>
                    {result.block_sessions} trading session
                    {result.block_sessions === 1 ? "" : "s"}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline" className="feature-badge">
                      {result.block_sessions ===
                      data.uncertainty?.primary_block_sessions
                        ? "Primary"
                        : "Sensitivity"}
                    </Badge>
                  </TableCell>
                  <TableCell className="number-cell">
                    {formatMetric(comparison.delta_auc.estimate, 4)}
                  </TableCell>
                  <TableCell className="number-cell">
                    [
                    {comparison.delta_auc.interval
                      .map((value) => formatMetric(value, 4))
                      .join(", ")}
                    ]
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </Card>
    </>
  );
}
function Pipeline({ data }: { data: ResearchData }) {
  const descriptions = [
    "Hypothesis, data contracts and a chronological protocol.",
    "Historical prices, news and approved discussion imports.",
    "Pinned FinBERT inference with ticker-level aggregation.",
    "Technical indicators and train-only preprocessing.",
    "Logistic Regression, Random Forest and naive controls.",
    "Frozen predictions and paired bootstrap uncertainty.",
    "Final report, data/model cards and a reproducible handoff.",
  ];
  const count = data.milestones.filter((milestone) => milestone.done).length;
  return (
    <>
      <Card>
        <SectionHeading
          title="Built to be questioned. Designed to be repeated."
          description="Software delivery and research feasibility are separate milestones."
          action={
            <StatusBadge tone="green">
              {count} / {data.milestones.length} implemented
            </StatusBadge>
          }
        />
        <div className="milestone-list">
          {data.milestones.map((milestone, index) => (
            <div className="milestone-row" key={milestone.id}>
              <span
                className={cn(
                  "milestone-check",
                  milestone.done && "is-complete",
                )}
              >
                {milestone.done ? <Check size={17} /> : index + 1}
              </span>
              <span className="milestone-id">{milestone.id}</span>
              <div>
                <h3>{milestone.title}</h3>
                <p>{descriptions[index]}</p>
              </div>
              <Badge variant="outline" className="feature-badge">
                {milestone.done ? "Complete" : "Pending"}
              </Badge>
            </div>
          ))}
        </div>
      </Card>
      <div className="two-columns">
        <InfoCard icon={LockKeyhole} title="The experiment is frozen">
          Model grids, feature ordering, selected settings, seeds and the
          evaluation plan are bound together. Changed artifacts block
          prediction.<code className="freeze-code">{data.freezeId}</code>
        </InfoCard>
        <InfoCard
          icon={FlaskConical}
          title="A reproducible result is an honest result"
        >
          The original test period has been evaluated. A future sentiment study
          needs suitable source history and a newly reserved holdout.
        </InfoCard>
      </div>
    </>
  );
}
function ModelDetails({
  detail,
}: {
  detail: { metric: Metric; partition: Partition };
}) {
  const { metric, partition } = detail;
  return (
    <>
      <DialogHeader>
        <div className="dialog-eyebrow">
          MODEL INSPECTION ·{" "}
          {partition === "test" ? "HELD-OUT TEST" : "VALIDATION"}
        </div>
        <DialogTitle>{modelName(metric)}</DialogTitle>
        <DialogDescription>
          {featureNames[metric.feature_set] ?? metric.feature_set} ·{" "}
          {metric.rows.toLocaleString()} observations · threshold 0.5
        </DialogDescription>
      </DialogHeader>
      <div className="inspection-metrics">
        <div>
          <span>ROC-AUC</span>
          <strong>{formatMetric(metric.roc_auc)}</strong>
        </div>
        <div>
          <span>F1-score</span>
          <strong>{formatMetric(metric.f1)}</strong>
        </div>
        <div>
          <span>Accuracy</span>
          <strong>{(metric.accuracy * 100).toFixed(1)}%</strong>
        </div>
      </div>
      <div className="matrix-heading">
        <h3>Confusion matrix</h3>
        <span>Actual × predicted</span>
      </div>
      <div className="confusion-grid">
        {["True non-up", "False up", "False non-up", "True up"].map(
          (label, index) => (
            <div
              className={cn((index === 0 || index === 3) && "correct-cell")}
              key={label}
            >
              <span>{label}</span>
              <strong>
                {metric.confusion_matrix[Math.floor(index / 2)][index % 2]}
              </strong>
            </div>
          ),
        )}
      </div>
      <p className="inspection-note">
        Positive-class prevalence:{" "}
        {(metric.positive_prevalence * 100).toFixed(1)}%. No eligible sentiment
        observations are present in the original study.
      </p>
      {metric.parameters && (
        <details className="parameter-details">
          <summary>Selected model parameters</summary>
          <pre>{JSON.stringify(metric.parameters, null, 2)}</pre>
        </details>
      )}
    </>
  );
}

export default function App() {
  const [view, setView] = useState<View>(() =>
    validView(location.hash.slice(1))
      ? (location.hash.slice(1) as View)
      : "overview",
  );
  const [data, setData] = useState<ResearchData | null>(null),
    [loading, setLoading] = useState(true),
    [error, setError] = useState("");
  const [partition, setPartition] = useState<Partition>("test"),
    [detail, setDetail] = useState<{
      metric: Metric;
      partition: Partition;
    } | null>(null),
    [help, setHelp] = useState(false),
    [mobile, setMobile] = useState(false);
  const [assets, setAssets] = useState({ report: false, handoff: false });
  const fileInput = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const controller = new AbortController();
    fetch("data.json", { signal: controller.signal })
      .then((response) => {
        if (!response.ok)
          throw new Error("No saved research snapshot was found.");
        return response.json();
      })
      .then((value) => {
        const snapshot = validateSnapshot(value);
        setData(snapshot);
        if (!snapshot.testMetrics) setPartition("validation");
      })
      .catch((reason) => {
        if (reason.name !== "AbortError")
          setError(
            "No saved research snapshot was found. Open a snapshot to explore your results.",
          );
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    fetch("snapshot-assets.json", { signal: controller.signal })
      .then((response) => response.json())
      .then((value) =>
        setAssets({
          report: value.report === true,
          handoff: value.handoff === true,
        }),
      )
      .catch(() => {});
    return () => controller.abort();
  }, []);
  useEffect(() => {
    const handleHash = () => {
      const name = location.hash.slice(1);
      if (validView(name)) setView(name);
    };
    addEventListener("hashchange", handleHash);
    return () => removeEventListener("hashchange", handleHash);
  }, []);
  useEffect(() => {
    document.title = `${navigation.find((item) => item.id === view)?.label} · Sentiment Lab`;
  }, [view]);
  const navigate = (name: View) => {
    setView(name);
    location.hash = name;
    setMobile(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const inspect = (metric: Metric, selectedPartition: Partition) =>
    setDetail({ metric, partition: selectedPartition });
  const importSnapshot = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      if (file.size > 15 * 1024 * 1024)
        throw new Error("Choose a snapshot smaller than 15 MB.");
      const snapshot = validateSnapshot(JSON.parse(await file.text()));
      setData(snapshot);
      setPartition(snapshot.testMetrics ? "test" : "validation");
      setError("");
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Unable to open this snapshot.",
      );
    }
    event.target.value = "";
  };
  const exportPartition =
    view === "models" ? partition : data?.testMetrics ? "test" : "validation";
  return (
    <TooltipProvider delayDuration={200}>
      <div className="app-shell">
        <aside className="desktop-sidebar">
          <Sidebar view={view} navigate={navigate} />
        </aside>
        <div className="page-shell">
          <header className="topbar">
            <div className="breadcrumb">
              <Sheet open={mobile} onOpenChange={setMobile}>
                <SheetTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="mobile-menu"
                    aria-label="Open navigation"
                  >
                    <Menu size={20} />
                  </Button>
                </SheetTrigger>
                <SheetContent side="left" className="mobile-sidebar">
                  <SheetHeader className="sr-only">
                    <SheetTitle>Research workspace navigation</SheetTitle>
                    <SheetDescription>
                      Explore Stock Sentiment Lab.
                    </SheetDescription>
                  </SheetHeader>
                  <Sidebar view={view} navigate={navigate} />
                </SheetContent>
              </Sheet>
              <img
                className="mobile-brand-mark"
                src="logo.svg"
                alt="Sentiment Lab"
                width="25"
                height="25"
              />
              <span className="breadcrumb-workspace">Workspace</span>
              <ChevronRight size={13} className="breadcrumb-chevron" />
              <strong>
                {navigation.find((item) => item.id === view)?.label}
              </strong>
            </div>
            <div className="topbar-actions">
              <span className="saved-status">
                <span className="small-dot" />
                Saved research snapshot
              </span>
              <span className="topbar-divider" />
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setHelp(true)}
                    aria-label="About this dashboard"
                  >
                    <CircleHelp size={18} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>About the experiment</TooltipContent>
              </Tooltip>
              <Button
                asChild
                variant="ghost"
                size="icon"
                aria-label="Open GitHub repository"
              >
                <a
                  href="https://github.com/Ammar1208830/stock-sentiment"
                  target="_blank"
                  rel="noopener"
                >
                  <CodeXml size={18} />
                </a>
              </Button>
            </div>
          </header>
          <main>
            <div className="page-heading">
              <div>
                <div className="eyebrow">
                  <span />
                  FINANCIAL NLP RESEARCH
                </div>
                <h1>{pageCopy[view][0]}</h1>
                <p>{pageCopy[view][1]}</p>
              </div>
              <div className="heading-actions">
                <input
                  ref={fileInput}
                  type="file"
                  accept="application/json,.json"
                  onChange={importSnapshot}
                  className="sr-only"
                  aria-label="Import research snapshot"
                />
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      className="export-trigger"
                      disabled={loading}
                    >
                      <ArrowDownToLine size={15} />
                      <span>Export data</span>
                      <ChevronRight size={13} className="export-chevron" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem
                      disabled={!data}
                      onSelect={() =>
                        data &&
                        download(
                          `sentiment-lab-${exportPartition}.csv`,
                          metricsCsv(
                            metricsFor(data, exportPartition),
                            exportPartition,
                          ),
                        )
                      }
                    >
                      <ListChecks size={15} />
                      {exportPartition === "test"
                        ? "Held-out"
                        : "Validation"}{" "}
                      metrics CSV
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      disabled={!data}
                      onSelect={() =>
                        data &&
                        download(
                          "sentiment-lab-prices.csv",
                          pricesCsv(data.prices),
                        )
                      }
                    >
                      <TrendingUp size={15} />
                      Historical prices CSV
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onSelect={() => fileInput.current?.click()}
                    >
                      <Upload size={15} />
                      Open a local snapshot
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>
            {loading ? (
              <div
                className="loading-grid"
                aria-label="Loading research snapshot"
              >
                {[1, 2, 3, 4].map((index) => (
                  <Skeleton key={index} className="loading-card" />
                ))}
                <Skeleton className="loading-chart" />
              </div>
            ) : !data ? (
              <Card className="snapshot-empty">
                <div className="empty-symbol">
                  <Database size={30} />
                </div>
                <h2>Your research starts with evidence.</h2>
                <p role="status">{error}</p>
                <Button onClick={() => fileInput.current?.click()}>
                  <Upload size={16} />
                  Open research snapshot
                </Button>
                <small>
                  JSON stays in your browser. No upload to a server.
                </small>
              </Card>
            ) : (
              <>
                {error && (
                  <div role="alert" className="import-error">
                    {error}
                    <Button
                      size="icon"
                      variant="ghost"
                      aria-label="Dismiss import error"
                      onClick={() => setError("")}
                    >
                      <X size={15} />
                    </Button>
                  </div>
                )}
                <div className="research-notice">
                  <span className="notice-flask">
                    <FlaskConical size={17} />
                  </span>
                  <p>
                    <strong>Evidence before conviction.</strong>{" "}
                    {data.evaluation
                      ? "Frozen evaluation is complete. "
                      : "Development snapshot. "}
                    {Object.values(data.features.sentiment_covered_rows).some(
                      (count) => count > 0,
                    )
                      ? "Research conclusions depend on eligible source coverage and the frozen study protocol."
                      : "News and Reddit history are missing, so the sentiment hypothesis remains open."}
                  </p>
                  <button onClick={() => navigate("results")}>
                    View findings <ArrowRight size={14} />
                  </button>
                </div>
                {view === "overview" && <MetricCards data={data} />}
                <div className="view-content" key={view}>
                  {view === "overview" ? (
                    <Overview
                      data={data}
                      navigate={navigate}
                      inspect={inspect}
                    />
                  ) : view === "market" ? (
                    <Market data={data} />
                  ) : view === "models" ? (
                    <Models
                      data={data}
                      partition={partition}
                      setPartition={setPartition}
                      inspect={inspect}
                    />
                  ) : view === "results" ? (
                    <Results data={data} assets={assets} />
                  ) : (
                    <Pipeline data={data} />
                  )}
                </div>
              </>
            )}
            <footer>
              <div>
                <img src="logo.svg" alt="" width="19" height="19" />
                <span>
                  Sentiment Lab<span className="footer-divider">/</span>
                  <span className="footer-tagline">
                    An empirical study of financial language
                  </span>
                </span>
              </div>
              <span>
                Built on evidence.<span className="footer-dot">·</span>Open to
                questions.
                <ArrowUpRight size={12} />
              </span>
            </footer>
          </main>
        </div>
      </div>
      <Dialog
        open={!!detail}
        onOpenChange={(open) => {
          if (!open) setDetail(null);
        }}
      >
        <DialogContent className="model-dialog">
          {detail && <ModelDetails detail={detail} />}
        </DialogContent>
      </Dialog>
      <Dialog open={help} onOpenChange={setHelp}>
        <DialogContent className="help-dialog">
          <DialogHeader>
            <span className="help-logo">
              <img src="logo.svg" alt="" width="48" height="48" />
            </span>
            <DialogTitle>A hypothesis, under test.</DialogTitle>
            <DialogDescription>
              Does financial language help predict next-session stock direction
              beyond what price alone can tell us?
            </DialogDescription>
          </DialogHeader>
          <p>
            This dashboard displays saved historical prices and real validation
            or frozen held-out model results. No live quotes or investment
            recommendations are provided.
          </p>
          <div className="help-callout">
            <ShieldCheck size={18} />
            <p>
              Eligible news and Reddit history are missing from the original
              study. Sentiment’s predictive value remains inconclusive.
            </p>
          </div>
          <div className="help-built-with">
            <span>Built with</span>
            <a
              href="https://github.com/shadcn-ui/ui"
              target="_blank"
              rel="noopener"
            >
              shadcn/ui
              <ExternalLink size={12} />
            </a>
            <a
              href="https://github.com/lucide-icons/lucide"
              target="_blank"
              rel="noopener"
            >
              Lucide
              <ExternalLink size={12} />
            </a>
          </div>
        </DialogContent>
      </Dialog>
    </TooltipProvider>
  );
}
