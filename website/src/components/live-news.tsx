import { useState } from "react";
import {
  ArrowUpRight,
  Clock3,
  ExternalLink,
  Newspaper,
  RefreshCw,
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { activeTickers, companyNames, dateLabel, usd } from "@/lib/research";
import type { LiveFeed, NewsArticle } from "@/hooks/use-live-feed";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

const localTime = (value: string | null) =>
  value
    ? new Date(value).toLocaleString([], {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "Not refreshed yet";

function SentimentBadge({ article }: { article: NewsArticle }) {
  const s = article.sentiment;
  const score =
    s.label === "positive"
      ? s.positive
      : s.label === "negative"
        ? s.negative
        : s.neutral;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge
          tabIndex={0}
          variant="outline"
          className={cn("headline-sentiment", s.label)}
        >
          {s.label} {(score * 100).toFixed(0)}%
        </Badge>
      </TooltipTrigger>
      <TooltipContent>
        <p>
          Positive {(s.positive * 100).toFixed(1)}% · Neutral{" "}
          {(s.neutral * 100).toFixed(1)}% · Negative{" "}
          {(s.negative * 100).toFixed(1)}%
        </p>
      </TooltipContent>
    </Tooltip>
  );
}
function TickerSelect({
  value,
  onChange,
}: {
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger
        aria-label="Filter by stock"
        className="news-ticker-select"
      >
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="all">All three stocks</SelectItem>
        {activeTickers.map((ticker) => (
          <SelectItem key={ticker} value={ticker}>
            {ticker} · {companyNames[ticker]}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
export function NewsStatus({
  feed,
  error,
  reload,
}: {
  feed: LiveFeed | null;
  error: string;
  reload: () => void;
}) {
  return (
    <div className="live-status">
      <div>
        <span
          className={cn(
            "small-dot",
            feed?.last_news_success_at && "feed-ready",
          )}
        />
        <strong>Prices + news</strong>
        <span>
          {feed
            ? `Updates every ${feed.refresh_hours}h · articles delayed ${feed.delay_hours}h`
            : error || "Connecting to the news service…"}
        </span>
      </div>
      <Button
        variant="ghost"
        size="icon"
        onClick={reload}
        aria-label="Reload latest feed results"
      >
        <RefreshCw size={14} />
      </Button>
    </div>
  );
}
export function NewsFeed({
  feed,
  error,
  reload,
  compact = false,
}: {
  feed: LiveFeed | null;
  error: string;
  reload: () => void;
  compact?: boolean;
}) {
  const [ticker, setTicker] = useState("all");
  const [limit, setLimit] = useState(20);
  if (!feed)
    return (
      <Card className="news-empty">
        <CardContent>
          <Newspaper size={26} />
          <h2>Financial news</h2>
          <p>{error || "Connecting to the news service…"}</p>
          <Button variant="outline" onClick={reload}>
            Check connection
          </Button>
        </CardContent>
      </Card>
    );
  const articles = feed.articles.filter(
    (article) => ticker === "all" || article.ticker === ticker,
  );
  return (
    <div className="news-workspace">
      <Card>
        <CardHeader className="section-heading">
          <div>
            <CardTitle>
              <h2>What the headlines are saying</h2>
            </CardTitle>
            <CardDescription>
              FinBERT scores the language in each headline. These are sentiment
              scores, not stock-price probabilities.
            </CardDescription>
          </div>
          {!compact && (
            <TickerSelect
              value={ticker}
              onChange={(value) => {
                setTicker(value);
                setLimit(20);
              }}
            />
          )}
        </CardHeader>
        <div className="news-freshness">
          <span>
            <Clock3 size={13} />
            Collected {localTime(feed.last_news_success_at)}
          </span>
          <span>Next refresh due {localTime(feed.next_refresh_at)}</span>
        </div>
        <div className="sentiment-cards">
          {feed.sentiment
            .filter((row) => ticker === "all" || row.ticker === ticker)
            .map((row) => (
              <div className="sentiment-card" key={row.ticker}>
                <span>{row.ticker}</span>
                <strong className={cn("sentiment-label", row.label)}>
                  {row.label === "unavailable" ? "No headlines" : row.label}
                </strong>
                <div
                  className="sentiment-meter"
                  aria-label={`${row.ticker}: ${row.articles} scored headlines`}
                >
                  {row.positive != null && (
                    <>
                      <i
                        className="positive"
                        style={{ width: `${row.positive * 100}%` }}
                      />
                      <i
                        className="neutral"
                        style={{ width: `${(row.neutral ?? 0) * 100}%` }}
                      />
                      <i
                        className="negative"
                        style={{ width: `${(row.negative ?? 0) * 100}%` }}
                      />
                    </>
                  )}
                </div>
                <small>{row.articles} scored headlines · 7-day window</small>
              </div>
            ))}
        </div>
        <div className="sentiment-legend">
          <span>
            <i className="positive" />
            Positive
          </span>
          <span>
            <i className="neutral" />
            Neutral
          </span>
          <span>
            <i className="negative" />
            Negative
          </span>
        </div>
        {feed.issues.length > 0 && (
          <div className="feed-issues" role="status">
            {feed.issues.join(" ")}
          </div>
        )}
        {feed.truncated_queries > 0 && (
          <p className="news-disclaimer">
            Some search results were capped. Sentiment reflects the collected
            sample, not every article.
          </p>
        )}
      </Card>
      {!compact && (
        <Card className="headline-list">
          <CardHeader className="section-heading">
            <div>
              <CardTitle>
                <h2>Latest collected headlines</h2>
              </CardTitle>
              <CardDescription>
                {articles.length} financial ticker-headline records · focus or
                hover a score for all three probabilities. Factual headlines can
                be neutral.
              </CardDescription>
            </div>
            <Badge variant="outline" className="feature-badge">
              NewsAPI · delayed feed
            </Badge>
          </CardHeader>
          {articles.slice(0, limit).map((article) => (
            <article className="headline-row" key={article.id + article.ticker}>
              <div className="headline-symbol">
                <Newspaper size={17} />
              </div>
              <div className="headline-copy">
                <div>
                  <Badge variant="outline" className="feature-badge">
                    {article.ticker}
                  </Badge>
                  <span>{article.publisher}</span>
                  <time dateTime={article.published_at}>
                    {localTime(article.published_at)}
                  </time>
                </div>
                <a href={article.url} target="_blank" rel="noopener noreferrer">
                  {article.title}
                  <ExternalLink size={12} />
                </a>
              </div>
              <SentimentBadge article={article} />
            </article>
          ))}
          {!articles.length && (
            <p className="news-empty-copy">
              No matching news has been collected yet. No neutral scores are
              substituted for missing headlines.
            </p>
          )}
          {articles.length > limit && (
            <Button
              className="show-more"
              variant="ghost"
              onClick={() => setLimit((value) => value + 20)}
            >
              Show more headlines
            </Button>
          )}
        </Card>
      )}
    </div>
  );
}
export function DataWalkthrough({ feed }: { feed: LiveFeed | null }) {
  const article = feed?.walkthrough_article;
  return (
    <Card className="data-walkthrough">
      <CardHeader className="section-heading">
        <div>
          <CardTitle>
            <h2>From data to a prediction</h2>
          </CardTitle>
          <CardDescription>
            One simple pipeline for Apple, NVIDIA and Amazon.
          </CardDescription>
        </div>
      </CardHeader>
      <ol className="walkthrough-steps">
        <li>
          <span>01</span>
          <div>
            <h3>Collect prices and news</h3>
            <p>
              yfinance supplies daily price history. NewsAPI supplies financial
              headlines with a 24-hour delay. We use only data available before
              the prediction cutoff.
            </p>
            {article && (
              <blockquote>
                <a href={article.url} target="_blank" rel="noopener noreferrer">
                  {article.title}
                </a>
                <small>
                  {article.ticker} · Published {localTime(article.published_at)}
                </small>
              </blockquote>
            )}
          </div>
        </li>
        <li>
          <span>02</span>
          <div>
            <h3>Read the headline with FinBERT</h3>
            <p>
              {article
                ? `This headline scored ${(article.sentiment.positive * 100).toFixed(1)}% positive, ${(article.sentiment.neutral * 100).toFixed(1)}% neutral and ${(article.sentiment.negative * 100).toFixed(1)}% negative.`
                : "FinBERT assigns positive, neutral and negative scores to each headline."}{" "}
              We average eligible sentiment for each stock. These scores
              describe the language, not the chance the stock will rise.
            </p>
          </div>
        </li>
        <li>
          <span>03</span>
          <div>
            <h3>Combine sentiment and price history</h3>
            <p>
              Recent price changes, volatility and RSI are combined with news
              sentiment and headline count. Ridge regression learns from earlier
              trading days and estimates the target day's return. Predicted
              close = previous close × (1 + predicted return).
            </p>
          </div>
        </li>
        <li>
          <span>04</span>
          <div>
            <h3>Compare with the actual close</h3>
            <p>
              Reveal the actual closing price and measure the dollar and
              percentage error. Smaller error is better. Compare with a
              price-only model and the simple guess that the price stays at its
              previous close.
            </p>
          </div>
        </li>
      </ol>
    </Card>
  );
}
