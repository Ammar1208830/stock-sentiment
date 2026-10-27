import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowUpRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import {
  companyNames,
  dateLabel,
  usd,
  type Price,
  type ResearchData,
} from "@/lib/research";
import { cn } from "@/lib/utils";
import { usePriceChartMotion } from "@/hooks/use-motion";
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
function PriceChart({ rows, ticker }: { rows: Price[]; ticker: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const chartRoot = useRef<HTMLDivElement>(null);
  usePriceChartMotion(
    chartRoot,
    `${ticker}:${rows[0].date}:${rows.at(-1)!.date}:${rows.length}:${rows.at(-1)!.close}`,
  );
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
    <div className="chart-wrap" ref={chartRoot}>
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
            <stop offset="0%" stopColor="var(--chart-area)" stopOpacity=".17" />
            <stop
              offset="100%"
              stopColor="var(--chart-area)"
              stopOpacity=".005"
            />
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
                stroke="var(--chart-grid)"
                strokeDasharray="3 5"
              />
              <text
                x={W - right + 18}
                y={y(value) + 4}
                fill="var(--chart-label)"
                fontSize="11"
              >
                ${value.toFixed(0)}
              </text>
            </g>
          );
        })}
        <path
          data-chart-area=""
          d={`${path} L${x(rows.length - 1)},${H - bottom} L${left},${H - bottom} Z`}
          fill="url(#price-area)"
        />
        <path
          data-chart-line=""
          d={path}
          fill="none"
          stroke="var(--chart-line)"
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
              fill="var(--chart-label)"
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
              stroke="var(--chart-line)"
              strokeDasharray="4 4"
            />
            <circle
              cx={x(hover)}
              cy={y(selected.close)}
              r="5"
              fill="var(--chart-line)"
              stroke="var(--card)"
              strokeWidth="2.5"
            />
          </g>
        ) : (
          <circle
            cx={x(rows.length - 1)}
            cy={y(rows.at(-1)!.close)}
            r="4"
            fill="var(--chart-line)"
            stroke="var(--card)"
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
export function MarketChart({
  data,
}: {
  data: Pick<ResearchData, "prices" | "asOf">;
}) {
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
          As of {dateLabel(last.date)}
          <span className="desktop-only"> · Historical, not live</span>
        </span>
      </div>
    </Card>
  );
}
