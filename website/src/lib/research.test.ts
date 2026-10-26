import test from "node:test";
import assert from "node:assert/strict";
import {
  filterMetrics,
  metricsCsv,
  metricsFor,
  validateSnapshot,
  type Metric,
  type ResearchData,
} from "./research.ts";

const metric = (id: string, auc: number): Metric => ({
  model_id: id,
  family: "logistic_regression",
  feature_set: "price_only",
  roc_auc: auc,
  f1: 0.6,
  accuracy: 0.5,
  precision: 0.5,
  recall: 0.8,
  rows: 755,
  positive_prevalence: 0.5,
  confusion_matrix: [
    [1, 2],
    [3, 4],
  ],
});
test("held-out and validation rows remain separate, including CSV exports", () => {
  const snapshot = {
    metrics: [metric("validation-model", 0.7)],
    testMetrics: [metric("frozen-model", 0.49)],
  } as ResearchData;
  assert.equal(metricsFor(snapshot, "test")[0].model_id, "frozen-model");
  const csv = metricsCsv(metricsFor(snapshot, "validation"), "validation");
  assert.ok(csv.includes('"validation","validation-model"'));
  assert.ok(!csv.includes("frozen-model"));
  assert.deepEqual(
    metricsFor({ metrics: snapshot.metrics } as ResearchData, "test"),
    [],
  );
});
test("baseline filtering retains the preselected seed, not every random experiment", () => {
  const rows = [
    metric("logistic_regression:price_only", 0.49),
    ...[1, 42].map((seed) => ({
      ...metric(`baseline:coin_seed_${seed}`, 0.5),
      family: "baseline",
      feature_set: "matched_sample",
    })),
  ];
  assert.equal(filterMetrics(rows).length, 1);
  assert.equal(filterMetrics(rows, "baseline").length, 1);
  assert.equal(filterMetrics(rows, "all", "all", true).length, 2);
  assert.equal(filterMetrics(rows, "all", "all", false, "forest").length, 0);
});
test("CSV quotes fields and neutralizes spreadsheet formulas from imported snapshots", () => {
  const csv = metricsCsv([metric('=HYPERLINK("bad")', 0.49)], "test");
  assert.ok(csv.includes('"\'=HYPERLINK(""bad"")"'));
});
test("import rejects malformed summaries and price timelines before rendering", () => {
  const snapshot = {
    asOf: "2026-10-23",
    freezeId: "frozen",
    prices: {
      AAPL: [
        { date: "2026-10-22", close: 1, volume: 5 },
        { date: "2026-10-23", close: 2, volume: 6 },
      ],
    },
    metrics: [metric("lr", 0.49)],
    milestones: [],
    summary: { trained_models: 1, holdout_rows_reserved: 755 },
    features: {
      retained_rows: 755,
      dropped_rows: 0,
      sentiment_covered_rows: {},
      partition_rows: {},
    },
  };
  assert.equal(validateSnapshot(snapshot).asOf, "2026-10-23");
  assert.throws(
    () => validateSnapshot({ ...snapshot, summary: {} }),
    /summary/,
  );
  assert.throws(
    () =>
      validateSnapshot({
        ...snapshot,
        prices: { AAPL: [...snapshot.prices.AAPL].reverse() },
      }),
    /unordered/,
  );
  assert.throws(
    () => validateSnapshot({ ...snapshot, testMetrics: {} }),
    /summary/,
  );
});
