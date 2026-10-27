import { useCallback, useEffect, useRef, useState } from "react";

export interface NewsArticle {
  id: string;
  ticker: string;
  title: string;
  url: string;
  publisher: string;
  published_at: string;
  first_observed_at: string;
  sentiment: {
    positive: number;
    negative: number;
    neutral: number;
    polarity: number;
    label: string;
  };
}
export interface LiveFeed {
  walkthrough_article: Pick<
    NewsArticle,
    "ticker" | "title" | "url" | "published_at" | "sentiment"
  > | null;
  price_evaluation_status: string;
  price_evaluation_error: string | null;
  price_evaluation: PriceEvaluation | null;
  tickers: string[];
  status: string;
  delay_hours: number;
  refresh_hours: number;
  last_attempt_at: string | null;
  last_completed_at: string | null;
  last_news_success_at: string | null;
  next_refresh_at: string | null;
  total_article_records: number;
  truncated_queries: number;
  issues: string[];
  articles: NewsArticle[];
  sentiment: {
    ticker: string;
    articles: number;
    positive: number | null;
    negative: number | null;
    neutral: number | null;
    polarity: number | null;
    label: string;
  }[];
  market: {
    as_of: string;
    observed_at: string;
    prices: Record<string, { date: string; close: number; volume: number }[]>;
  } | null;
}

export interface PriceEvaluation {
  target_session: string;
  evaluated_at: string;
  training_rows: number;
  training_sessions: number;
  training_start: string;
  training_end: string;
  training_news_rows: number;
  news_records: number;
  news_start: string;
  news_end: string;
  mae: number | null;
  mape: number | null;
  baseline_mae: number | null;
  price_only_mae: number | null;
  truncated_queries: number;
  rows: {
    ticker: string;
    session: string;
    decision_at: string;
    reference_session: string;
    reference_close: number;
    actual_close: number | null;
    predicted_close: number;
    price_only_close: number;
    baseline_close: number;
    absolute_error: number | null;
    percentage_error: number | null;
    baseline_error: number | null;
    price_only_error: number | null;
    news_count: number;
    news_polarity: number | null;
    news_latest_published_at: string | null;
  }[];
}

export function useLiveFeed() {
  const [feed, setFeed] = useState<LiveFeed | null>(null);
  const [error, setError] = useState("");
  const controller = useRef<AbortController | null>(null);
  const reload = useCallback(async () => {
    controller.current?.abort();
    const request = new AbortController();
    controller.current = request;
    try {
      const response = await fetch("/api/live", {
        cache: "no-store",
        signal: request.signal,
      });
      if (!response.ok) throw new Error("The news service is not available.");
      const value = await response.json();
      if (!Array.isArray(value.articles) || !Array.isArray(value.sentiment))
        throw new Error("The news service returned an invalid response.");
      setFeed(value);
      setError("");
    } catch (reason) {
      if (!(reason instanceof DOMException && reason.name === "AbortError"))
        setError(
          "The news service is offline. Last received results are retained.",
        );
    }
  }, []);
  useEffect(() => {
    void reload();
    const visible = () => {
      if (document.visibilityState === "visible") void reload();
    };
    const timer = setInterval(visible, 60_000);
    document.addEventListener("visibilitychange", visible);
    return () => {
      clearInterval(timer);
      controller.current?.abort();
      document.removeEventListener("visibilitychange", visible);
    };
  }, [reload]);
  return { feed, error, reload };
}
