export type Partition = "test" | "validation";
export type View =
  | "overview"
  | "market"
  | "news"
  | "tracker"
  | "models"
  | "results"
  | "pipeline";
export interface Price {
  date: string;
  close: number;
  volume: number;
}
export interface Metric {
  model_id: string;
  family: string;
  feature_set: string;
  rows: number;
  roc_auc: number | null;
  f1: number;
  accuracy: number;
  precision: number;
  recall: number;
  positive_prevalence: number;
  confusion_matrix: number[][];
  parameters?: Record<string, unknown>;
}
export interface Interval {
  estimate: number;
  interval: [number, number];
  valid_replicates: number;
}
export interface ResearchData {
  asOf: string;
  prices: Record<string, Price[]>;
  metrics: Metric[];
  testMetrics?: Metric[];
  summary: {
    trained_models: number;
    attempted_candidates: number;
    holdout_rows_reserved: number;
  };
  features: {
    retained_rows: number;
    dropped_rows: number;
    partition_rows: Record<string, number>;
    sentiment_covered_rows: Record<string, number>;
  };
  freezeId: string;
  milestones: { id: string; title: string; done: boolean }[];
  evaluation?: {
    primary_control_metrics: Metric;
    primary_augmented_metrics: Metric;
    primary_delta_auc: Interval;
    primary_delta_f1: Interval;
    distinct_test_sessions: number;
    test_start: string;
    test_end: string;
    test_rows: number;
  };
  uncertainty?: {
    primary_block_sessions: number;
    results: {
      block_sessions: number;
      comparisons: { role: string; delta_auc: Interval }[];
    }[];
  };
  conclusion?: { statement: string };
}
export const companyNames: Record<string, string> = {
  AAPL: "Apple Inc.",
  MSFT: "Microsoft",
  NVDA: "NVIDIA",
  AMZN: "Amazon",
  TSLA: "Tesla",
};
export const activeTickers = ["AAPL", "NVDA", "AMZN"] as const;
export const featureNames: Record<string, string> = {
  price_only: "Price only",
  price_news: "Price + news",
  price_reddit: "Price + Reddit",
  sentiment_only: "Sentiment only",
  combined: "Combined",
  matched_sample: "Matched baseline",
};
export const formatMetric = (value: number | null | undefined, digits = 3) =>
  value == null ? "—" : value.toFixed(digits);
export const usd = (value: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(
    value,
  );
export const dateLabel = (value: string, short = false) =>
  new Date(value + "T12:00:00Z").toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    ...(short ? {} : { year: "numeric" }),
  });
export function modelName(metric: Metric) {
  if (metric.family === "logistic_regression") return "Logistic Regression";
  if (metric.family === "random_forest") return "Random Forest";
  const id = metric.model_id.split(":")[1];
  return (
    (
      {
        training_majority: "Training majority",
        always_up: "Always up",
        zero_drift_no_change: "No-change forecast",
        momentum: "Momentum",
      } as Record<string, string>
    )[id] || `Seeded coin · ${id?.split("_").at(-1)}`
  );
}
export function metricsFor(data: ResearchData, partition: Partition): Metric[] {
  // Never silently substitute validation results for a held-out comparison.
  return partition === "test" ? (data.testMetrics ?? []) : data.metrics;
}
export function filterMetrics(
  rows: Metric[],
  family = "all",
  features = "all",
  baselines = false,
  query = "",
) {
  return rows.filter(
    (row) =>
      (family === "all"
        ? row.family !== "baseline" || baselines
        : row.family === family) &&
      (features === "all" || row.feature_set === features) &&
      (!row.model_id.includes("coin_seed_") ||
        row.model_id.endsWith("coin_seed_42")) &&
      `${modelName(row)} ${featureNames[row.feature_set] ?? row.feature_set}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
}
export function metricsCsv(rows: Metric[], partition: Partition) {
  const keys = [
    "model_id",
    "feature_set",
    "roc_auc",
    "f1",
    "accuracy",
    "precision",
    "recall",
  ] as const;
  return [
    "partition," + keys.join(","),
    ...rows.map((row) =>
      [partition, ...keys.map((key) => row[key] ?? "")].map(csvCell).join(","),
    ),
  ].join("\n");
}
export function csvCell(value: unknown) {
  const text = String(value);
  // Prevent user-imported text from becoming a spreadsheet formula on export.
  return (
    '"' +
    (/^[=+@\-\t\r]/.test(text) ? "'" : "") +
    text.replaceAll('"', '""') +
    '"'
  );
}
export function pricesCsv(prices: Record<string, Price[]>) {
  return [
    "ticker,date,vendor_close,volume",
    ...Object.entries(prices).flatMap(([ticker, rows]) =>
      rows.map((row) =>
        [ticker, row.date, row.close, row.volume].map(csvCell).join(","),
      ),
    ),
  ].join("\n");
}
export function download(name: string, contents: string, type = "text/csv") {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function validateSnapshot(value: unknown): ResearchData {
  if (!value || typeof value !== "object")
    throw new Error("Please choose a valid research snapshot.");
  const data = value as ResearchData;
  if (
    !data.prices ||
    !Object.keys(data.prices).length ||
    !Array.isArray(data.metrics) ||
    !data.summary ||
    !data.features ||
    !Array.isArray(data.milestones)
  )
    throw new Error(
      "This file is missing the research prices, metrics or experiment summary.",
    );
  const validCount = (count: unknown) =>
    Number.isInteger(count) && Number(count) >= 0;
  const validDate = (date: unknown) =>
    typeof date === "string" &&
    /^\d{4}-\d{2}-\d{2}$/.test(date) &&
    !Number.isNaN(Date.parse(date));
  const validInterval = (interval: Interval) =>
    interval &&
    Number.isFinite(interval.estimate) &&
    Array.isArray(interval.interval) &&
    interval.interval.length === 2 &&
    interval.interval.every(Number.isFinite) &&
    interval.interval[0] <= interval.interval[1] &&
    validCount(interval.valid_replicates);
  if (
    !validDate(data.asOf) ||
    typeof data.freezeId !== "string" ||
    !validCount(data.summary.trained_models) ||
    !validCount(data.summary.holdout_rows_reserved) ||
    !validCount(data.features.retained_rows) ||
    !validCount(data.features.dropped_rows) ||
    !data.features.sentiment_covered_rows ||
    !data.features.partition_rows ||
    data.milestones.some(
      (milestone) =>
        !milestone ||
        typeof milestone.id !== "string" ||
        typeof milestone.title !== "string" ||
        typeof milestone.done !== "boolean",
    ) ||
    (data.testMetrics !== undefined && !Array.isArray(data.testMetrics))
  )
    throw new Error("The snapshot contains an invalid experiment summary.");
  for (const rows of Object.values(data.prices)) {
    if (
      !Array.isArray(rows) ||
      rows.length < 2 ||
      rows.some(
        (row, index) =>
          !row ||
          !validDate(row.date) ||
          !Number.isFinite(row.close) ||
          row.close <= 0 ||
          !Number.isFinite(row.volume) ||
          row.volume < 0 ||
          (index > 0 && row.date <= rows[index - 1].date),
      )
    )
      throw new Error(
        "The snapshot contains invalid or unordered price observations.",
      );
  }
  for (const row of [...data.metrics, ...(data.testMetrics ?? [])]) {
    if (
      !row ||
      typeof row.model_id !== "string" ||
      typeof row.family !== "string" ||
      typeof row.feature_set !== "string" ||
      [
        row.accuracy,
        row.f1,
        row.precision,
        row.recall,
        row.positive_prevalence,
      ].some((value) => !Number.isFinite(value) || value < 0 || value > 1) ||
      (row.roc_auc !== null &&
        (!Number.isFinite(row.roc_auc) ||
          row.roc_auc < 0 ||
          row.roc_auc > 1)) ||
      !validCount(row.rows) ||
      !Array.isArray(row.confusion_matrix) ||
      row.confusion_matrix.length !== 2 ||
      row.confusion_matrix.some(
        (cells) =>
          !Array.isArray(cells) ||
          cells.length !== 2 ||
          cells.some((cell) => !validCount(cell)),
      )
    )
      throw new Error("The snapshot contains invalid model metrics.");
  }
  if (
    data.evaluation &&
    (!data.testMetrics ||
      !Number.isFinite(data.evaluation.primary_control_metrics?.roc_auc) ||
      !validInterval(data.evaluation.primary_delta_auc) ||
      !validInterval(data.evaluation.primary_delta_f1) ||
      !validCount(data.evaluation.test_rows) ||
      !validCount(data.evaluation.distinct_test_sessions))
  )
    throw new Error("The snapshot contains an invalid held-out evaluation.");
  if (
    data.uncertainty &&
    (!Array.isArray(data.uncertainty.results) ||
      data.uncertainty.results.some(
        (result) =>
          !validCount(result.block_sessions) ||
          !Array.isArray(result.comparisons) ||
          result.comparisons.some(
            (comparison) => !validInterval(comparison.delta_auc),
          ),
      ))
  )
    throw new Error("The snapshot contains invalid uncertainty intervals.");
  return data;
}
