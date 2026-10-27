import { useEffect, useRef, useState } from "react";
import {
  ArrowUpRight,
  ChartNoAxesCombined,
  CircleHelp,
  LayoutDashboard,
  ListChecks,
  Moon,
  Newspaper,
  Sun,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { DataWalkthrough, NewsFeed, NewsStatus } from "@/components/live-news";
import { PriceComparison } from "@/components/price-comparison";
import { MarketChart } from "@/components/market-chart";
import { useLiveFeed } from "@/hooks/use-live-feed";
import { useTheme } from "@/hooks/use-theme";
import { useDashboardMotion } from "@/hooks/use-motion";
import { activeTickers, dateLabel } from "@/lib/research";
import { cn } from "@/lib/utils";

const pages = [
  {
    id: "overview",
    label: "Overview",
    icon: LayoutDashboard,
    title: "Three stocks. A clearer picture.",
    description:
      "Apple, NVIDIA and Amazon. From financial news to checked price predictions.",
  },
  {
    id: "news",
    label: "News & sentiment",
    icon: Newspaper,
    title: "What the headlines are saying.",
    description:
      "Financial news, read by FinBERT. A simple view of positive, neutral and negative language.",
  },
  {
    id: "tracker",
    label: "Price comparison",
    icon: ListChecks,
    title: "Predictions, checked against reality.",
    description:
      "Compare the predicted closing price with the actual close. Smaller error is better.",
  },
  {
    id: "market",
    label: "Market data",
    icon: ChartNoAxesCombined,
    title: "The price behind the prediction.",
    description: "Daily closing prices for Apple, NVIDIA and Amazon.",
  },
] as const;
type Page = (typeof pages)[number]["id"];
function initialPage(): Page {
  return (
    pages.find((page) => page.id === location.hash.slice(1))?.id ?? "overview"
  );
}

export default function App() {
  const { feed, error, reload } = useLiveFeed();
  const { theme, toggleTheme } = useTheme();
  const [view, setView] = useState<Page>(initialPage);
  const [help, setHelp] = useState(false);
  const motionRoot = useRef<HTMLElement>(null);
  useDashboardMotion(motionRoot, view, Boolean(feed));
  const page = pages.find((item) => item.id === view)!;
  useEffect(() => {
    const change = () => setView(initialPage());
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  useEffect(() => {
    document.title = `${page.label} · Sentiment Lab`;
  }, [page.label]);
  const navigate = (id: Page) => {
    setView(id);
    location.hash = id;
    window.scrollTo({ top: 0, behavior: "instant" });
  };
  const result = feed?.price_evaluation;
  const prices = Object.fromEntries(
    activeTickers.flatMap((ticker) => {
      const rows = feed?.market?.prices[ticker];
      return rows?.length ? [[ticker, rows]] : [];
    }),
  );
  const nav = (mobile = false) => (
    <nav
      className={mobile ? "mobile-pages" : "simple-navigation"}
      aria-label={mobile ? "Mobile navigation" : "Main navigation"}
    >
      {pages.map((item) => (
        <button
          key={item.id}
          className={cn("nav-link", view === item.id && "is-active")}
          aria-current={view === item.id ? "page" : undefined}
          onClick={() => navigate(item.id)}
        >
          <item.icon size={17} strokeWidth={1.7} />
          <span>{item.label}</span>
          {!mobile && view === item.id && <span className="nav-active-dot" />}
        </button>
      ))}
    </nav>
  );
  return (
    <TooltipProvider>
      <div className="app-shell">
        <aside className="desktop-sidebar">
          <div className="sidebar-inner simple-sidebar">
            <a
              href="#overview"
              className="brand"
              onClick={() => navigate("overview")}
            >
              <img src="logo.svg" width="38" height="38" alt="" />
              <span>
                Sentiment<span className="brand-light">Lab</span>
              </span>
            </a>
            <div className="sidebar-section-label">YOUR WORKSPACE</div>
            {nav()}
            <div className="sidebar-study">
              <ChartNoAxesCombined size={17} />
              <div>
                <strong>Only three stocks</strong>
                <span>AAPL · NVDA · AMZN</span>
              </div>
            </div>
            <div className="simple-sidebar-bottom">
              <p>
                Prices + headlines.
                <br />
                Predictions checked against facts.
              </p>
              <button onClick={() => setHelp(true)}>
                <CircleHelp size={14} />
                How this works
              </button>
            </div>
          </div>
        </aside>
        <div className="page-shell">
          <header className="topbar">
            <div className="breadcrumb">
              <img
                className="mobile-brand"
                src="logo.svg"
                width="26"
                height="26"
                alt="Sentiment Lab"
              />
              <span>Sentiment Lab</span>
              <span className="breadcrumb-divider">/</span>
              <strong>{page.label}</strong>
            </div>
            <div className="topbar-actions">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label={
                      theme === "dark"
                        ? "Switch to light mode"
                        : "Switch to dark mode"
                    }
                    onClick={toggleTheme}
                  >
                    {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {theme === "dark" ? "Light mode" : "Dark mode"}
                </TooltipContent>
              </Tooltip>
              <Button
                variant="ghost"
                size="icon"
                aria-label="How this works"
                onClick={() => setHelp(true)}
              >
                <CircleHelp size={17} />
              </Button>
            </div>
          </header>
          {nav(true)}
          <main ref={motionRoot} className="main-content simple-main">
            <div className="page-heading">
              <div>
                <div className="page-eyebrow">STOCK SENTIMENT PROJECT</div>
                <h1>{page.title}</h1>
                <p>{page.description}</p>
              </div>
            </div>
            <NewsStatus feed={feed} error={error} reload={reload} />
            <div className="view-content" key={view}>
              {view === "overview" ? (
                <>
                  <section
                    className="metric-cards"
                    aria-label="Project summary"
                  >
                    {[
                      {
                        label: "Stocks followed",
                        value: "3",
                        caption: "Apple · NVIDIA · Amazon",
                      },
                      {
                        label: "News scored for training",
                        value: result?.news_records.toLocaleString() ?? "—",
                        caption: "Real headlines · FinBERT",
                      },
                      {
                        label: "Training examples",
                        value: result?.training_rows.toLocaleString() ?? "—",
                        caption: result
                          ? `${result.training_sessions} earlier trading sessions`
                          : "Preparing the comparison",
                      },
                      {
                        label: "Comparison day",
                        value: result
                          ? dateLabel(result.target_session, true)
                          : "—",
                        caption: "Predicted close vs. actual close",
                      },
                    ].map((item) => (
                      <Card className="metric-card" key={item.label}>
                        <div className="simple-stat">
                          <span>{item.label}</span>
                          <strong>{item.value}</strong>
                          <small>{item.caption}</small>
                        </div>
                      </Card>
                    ))}
                  </section>
                  <PriceComparison feed={feed} />
                  <DataWalkthrough feed={feed} />
                  <div className="overview-actions">
                    <Button variant="outline" onClick={() => navigate("news")}>
                      Explore headlines <ArrowUpRight size={14} />
                    </Button>
                    <Button variant="outline" onClick={() => setHelp(true)}>
                      How this works <CircleHelp size={14} />
                    </Button>
                  </div>
                </>
              ) : view === "news" ? (
                <NewsFeed feed={feed} error={error} reload={reload} />
              ) : view === "tracker" ? (
                <>
                  <PriceComparison feed={feed} />
                  <DataWalkthrough feed={feed} />
                </>
              ) : Object.keys(prices).length ? (
                <MarketChart data={{ prices, asOf: feed!.market!.as_of }} />
              ) : (
                <Card className="news-empty">
                  <p>{error || "Daily price history is loading…"}</p>
                </Card>
              )}
            </div>
            <footer>
              <div>
                <img src="logo.svg" alt="" width="18" height="18" />
                <span>
                  Sentiment Lab<span className="footer-divider">/</span>
                  <span className="footer-tagline">
                    A simple price and news experiment
                  </span>
                </span>
              </div>
              <button className="footer-help" onClick={() => setHelp(true)}>
                How this works <CircleHelp size={12} />
              </button>
            </footer>
          </main>
        </div>
      </div>
      <Dialog open={help} onOpenChange={setHelp}>
        <DialogContent className="help-dialog">
          <DialogHeader>
            <DialogTitle>How this works</DialogTitle>
            <DialogDescription>
              Three stocks, two data sources, one checked prediction.
            </DialogDescription>
          </DialogHeader>
          <div className="simple-help">
            <p>
              <strong>Data:</strong> yfinance daily price history and financial
              headlines from NewsAPI. The news feed refreshes every six hours
              and has a 24-hour provider delay.
            </p>
            <p>
              <strong>FinBERT:</strong> a pretrained transformer reads headlines
              and scores positive, neutral and negative language. We average
              eligible scores for each stock over a three-day window.
            </p>
            <p>
              <strong>Prediction:</strong> Ridge regression combines that
              sentiment and headline count with price momentum, volatility and
              RSI. It learns daily returns from earlier examples and converts
              the predicted return into a closing price.
            </p>
            <p>
              <strong>Timing:</strong> we run the comparison now using
              historical data. The target day's prices are hidden from training,
              then revealed for evaluation. We assume headlines become available
              24 hours after publication and daily closes one minute after
              market close. These were not predictions actually issued on those
              past dates, and the historical delivery times and later article or
              price revisions cannot be verified.
            </p>
            <p>
              <strong>Your example:</strong> for Monday, 26 October 2026, the
              last prior close is Friday the 23rd. The cutoff is 09:25 New York
              time on the 26th. Only headlines whose 24-hour delay ended before
              that cutoff are eligible, so later Sunday news is excluded.
            </p>
            <p>
              <strong>Checking:</strong> dollar error is |predicted − actual|;
              percentage error divides that by the actual close. We compare the
              combined model with a price-only model and simply reusing the
              preceding close. A short sample can demonstrate the method, but
              cannot prove forecasting accuracy.
            </p>
          </div>
        </DialogContent>
      </Dialog>
    </TooltipProvider>
  );
}
