import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { companyNames, dateLabel, usd } from "@/lib/research";
import type { LiveFeed } from "@/hooks/use-live-feed";

export function PriceComparison({ feed }: { feed: LiveFeed | null }) {
  const result = feed?.price_evaluation;
  if (!result)
    return (
      <Card className="comparison-empty">
        <CardHeader>
          <CardTitle>Price comparison</CardTitle>
          <CardDescription>
            {feed?.price_evaluation_error ??
              (feed?.price_evaluation_status === "running"
                ? "Collecting historical headlines and training the combined model…"
                : "The first price comparison is being prepared.")}
          </CardDescription>
        </CardHeader>
      </Card>
    );
  return (
    <Card className="price-comparison">
      <CardHeader className="section-heading">
        <div>
          <CardTitle>
            <h2>Predicted vs. actual</h2>
          </CardTitle>
          <CardDescription>
            Closing prices for {dateLabel(result.target_session)} · price
            history + FinBERT sentiment
          </CardDescription>
        </div>
        <span className="comparison-date">{result.rows.length} stocks</span>
      </CardHeader>
      <Table className="comparison-table">
        <TableHeader>
          <TableRow>
            <TableHead>Stock</TableHead>
            <TableHead>Predicted close</TableHead>
            <TableHead>Actual close</TableHead>
            <TableHead>Dollar error ↓</TableHead>
            <TableHead>Error % ↓</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {result.rows.map((row) => (
            <TableRow key={row.ticker}>
              <TableCell>
                <strong>{row.ticker}</strong>
                <small className="comparison-subtitle">
                  {companyNames[row.ticker]}
                </small>
              </TableCell>
              <TableCell className="prediction-price">
                {usd(row.predicted_close)}
              </TableCell>
              <TableCell>
                {row.actual_close == null
                  ? "Awaiting close"
                  : usd(row.actual_close)}
              </TableCell>
              <TableCell>
                {row.absolute_error == null ? "—" : usd(row.absolute_error)}
              </TableCell>
              <TableCell>
                {row.percentage_error == null
                  ? "—"
                  : `${row.percentage_error.toFixed(2)}%`}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <div className="tracker-summary">
        <div>
          <span>Mean dollar error</span>
          <strong>{result.mae == null ? "—" : usd(result.mae)}</strong>
        </div>
        <div>
          <span>Mean percentage error</span>
          <strong>
            {result.mape == null ? "—" : `${result.mape.toFixed(2)}%`}
          </strong>
        </div>
        <div>
          <span>Price-only model error</span>
          <strong>
            {result.price_only_mae == null ? "—" : usd(result.price_only_mae)}
          </strong>
        </div>
        <div>
          <span>Unchanged-price guess error</span>
          <strong>
            {result.baseline_mae == null ? "—" : usd(result.baseline_mae)}
          </strong>
        </div>
      </div>
      <details className="comparison-inputs">
        <summary>See the inputs and baselines</summary>
        <p>
          Trained on {result.training_rows} stock-days across{" "}
          {result.training_sessions} trading sessions (
          {dateLabel(result.training_start)}–{dateLabel(result.training_end)}).{" "}
          {result.training_news_rows} training rows contain scored news. The
          comparison day's prices are withheld from training.
        </p>
        <Table className="comparison-table">
          <TableHeader>
            <TableRow>
              <TableHead>Stock</TableHead>
              <TableHead>Previous close</TableHead>
              <TableHead>News headlines used</TableHead>
              <TableHead>Price-only prediction</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {result.rows.map((row) => (
              <TableRow key={row.ticker}>
                <TableCell>{row.ticker}</TableCell>
                <TableCell>
                  {usd(row.reference_close)}
                  <small className="comparison-subtitle">
                    {dateLabel(row.reference_session)}
                  </small>
                </TableCell>
                <TableCell>
                  {row.news_count}
                  {!row.news_count && (
                    <small className="comparison-subtitle">
                      No eligible headlines
                    </small>
                  )}
                </TableCell>
                <TableCell>{usd(row.price_only_close)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </details>
      <p className="news-disclaimer">
        Lower error is better. One comparison day and a short news history are
        not enough to establish predictive accuracy.
        {result.truncated_queries > 0
          ? " Some headline searches were capped, so sentiment reflects a sample."
          : ""}
      </p>
      {feed?.price_evaluation_error && (
        <p role="status" className="feed-issues">
          {feed.price_evaluation_error}
        </p>
      )}
    </Card>
  );
}
