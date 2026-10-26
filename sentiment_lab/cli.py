import argparse
from datetime import datetime, timezone
import hashlib
from importlib.resources import files, as_file
import json
from pathlib import Path
import platform
import sys
import tomllib
from contextlib import ExitStack

from . import __version__
from .config import Config
from .contracts import ContractError
from .io import file_hash, read_inputs, write_json, write_jsonl
from .pipeline import assemble


def run(config_path: Path, sessions_path: Path, prices_path: Path, texts_path: Path,
        output: Path, mode: str) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ContractError("Output directory must be empty; use a new directory to preserve previous runs")
    config = Config.load(config_path)
    paths = {"config": config_path, "sessions": sessions_path, "prices": prices_path, "texts": texts_path}
    hashes = {name: file_hash(path) for name, path in paths.items()}
    sessions, prices, texts = read_inputs(sessions_path, prices_path, texts_path)
    rows, report = assemble(config, sessions, prices, texts)
    # Keep targets physically separate from feature inputs to avoid accidental training leakage.
    label_fields = {"label_up", "target_return", "label_available_at"}
    features = [{k: v for k, v in row.items() if k not in label_fields} for row in rows]
    labels = [{k: row[k] for k in ("ticker", "target_session", "partition", *sorted(label_fields))} for row in rows]
    code_hashes = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    identity = {"inputs": hashes, "config": config.to_dict(), "code": code_hashes,
                "mode": mode, "python_version": platform.python_version()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "windows.jsonl", features)
    write_jsonl(output / "labels.jsonl", labels)
    write_json(output / "validation.json", report)
    manifest = {"schema_version": 1, "run_fingerprint": fingerprint, "package_version": __version__,
                "created_at": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version(),
                "mode": mode, "synthetic_data": True if mode == "synthetic_demo" else None,
                "config": config.to_dict(),
                "input_sha256": hashes, "code_sha256": code_hashes,
                "output_sha256": {name: file_hash(output / name) for name in ("windows.jsonl", "labels.jsonl", "validation.json")},
                "limitations": ["Validation does not collect, score text, compute features or train models.",
                    "Use verify-calendar and inspect the acquisition data card to verify calendar/provenance.",
                    "Timestamp values are validated, but provider point-in-time correctness is not independently proven.",
                    "Raw close direction excludes split target sessions; dividends remain part of price movement.",
                    "Features include metadata and text, not a numeric model matrix."]}
    write_json(output / "manifest.json", manifest)
    return {"output": str(output.resolve()), "rows": len(rows), "run_fingerprint": fingerprint,
            "mode": mode, "message": "Input validation completed. No models were trained."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stock sentiment research foundation")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Validate configuration without printing secrets")
    doctor.add_argument("--config", type=Path, default=Path("configs/project.toml"))
    demo = sub.add_parser("demo", help="Run synthetic alignment fixtures without network access")
    demo.add_argument("--out", type=Path, default=Path("artifacts/m1-demo"))
    validate = sub.add_parser("validate", help="Validate local normalized data and build audited windows")
    validate.add_argument("--config", required=True, type=Path)
    validate.add_argument("--sessions", required=True, type=Path)
    validate.add_argument("--prices", required=True, type=Path)
    validate.add_argument("--texts", required=True, type=Path)
    validate.add_argument("--out", required=True, type=Path)
    calendar = sub.add_parser('calendar', help='Generate a verified XNYS calendar with the preceding session')
    calendar.add_argument('--start', required=True)
    calendar.add_argument('--end', required=True)
    calendar.add_argument('--out', required=True, type=Path)
    verify = sub.add_parser('verify-calendar', help='Verify supplied sessions against the pinned XNYS library')
    verify.add_argument('--start', required=True)
    verify.add_argument('--end', required=True)
    verify.add_argument('--sessions', required=True, type=Path)
    acquire = sub.add_parser('collect', help='Acquire/cached-export M2 data and write the feasibility gate')
    acquire.add_argument('--config', type=Path, default=Path('configs/project.toml'))
    acquire.add_argument('--acquisition', type=Path, default=Path('configs/acquisition.toml'))
    acquire.add_argument('--cache', type=Path, default=Path('data/cache/acquisition.sqlite'))
    acquire.add_argument('--out', required=True, type=Path)
    acquire.add_argument('--refresh-prices', action='store_true')
    acquire.add_argument('--refresh-news', action='store_true')
    acquire.add_argument('--offline', action='store_true', help='Export only cached data without provider calls')
    acquire.add_argument('--require-feasible', action='store_true', help='Return exit code 3 if the full-study data gate fails')
    imported = sub.add_parser('import-text', help='Import approved normalized historical text with provenance')
    imported.add_argument('--config', type=Path, default=Path('configs/project.toml'))
    imported.add_argument('--input', required=True, type=Path)
    imported.add_argument('--provenance', required=True, type=Path)
    imported.add_argument('--cache', type=Path, default=Path('data/cache/acquisition.sqlite'))
    purge = sub.add_parser('purge-text', help='Delete all cached versions of a source origin ID')
    purge.add_argument('--source', required=True, choices=['reddit', 'newsapi'])
    purge.add_argument('--origin-id', required=True)
    purge.add_argument('--cache', type=Path, default=Path('data/cache/acquisition.sqlite'))
    m2_demo = sub.add_parser('m2-demo', help='Exercise M2 cache, verified calendar, dedup and feasibility without network')
    m2_demo.add_argument('--out', type=Path, required=True)
    m2_demo.add_argument('--cache', type=Path, default=Path('data/cache/m2-demo.sqlite'))
    prepare = sub.add_parser('prepare-finbert', help='Download and hash the exact pinned FinBERT/tokenizer artifacts')
    prepare.add_argument('--sentiment', type=Path, default=Path('configs/sentiment.toml'))
    prepare.add_argument('--model-root', type=Path, default=Path('data/cache/models/finbert'))
    prepare.add_argument('--offline', action='store_true', help='Verify existing local artifacts without downloading')
    score = sub.add_parser('score-sentiment', help='Score eligible M2 text and aggregate by ticker/session/source')
    score.add_argument('--dataset', required=True, type=Path)
    score.add_argument('--config', type=Path, default=Path('configs/project.toml'))
    score.add_argument('--sentiment', type=Path, default=Path('configs/sentiment.toml'))
    score.add_argument('--model-root', type=Path, default=Path('data/cache/models/finbert'))
    score.add_argument('--cache', type=Path, default=Path('data/cache/sentiment.sqlite'))
    score.add_argument('--out', required=True, type=Path)
    score.add_argument('--download-model', action='store_true', help='Allow first-time download; default is local artifacts only')
    smoke = sub.add_parser('sentiment-smoke', help='Run real FinBERT inference on clearly synthetic financial examples')
    smoke.add_argument('--sentiment', type=Path, default=Path('configs/sentiment.toml'))
    smoke.add_argument('--model-root', type=Path, default=Path('data/cache/models/finbert'))
    smoke.add_argument('--cache', type=Path, default=Path('data/cache/sentiment-smoke.sqlite'))
    smoke.add_argument('--out', required=True, type=Path)
    smoke.add_argument('--download-model', action='store_true')
    features = sub.add_parser('build-features', help='Build M4 matched technical/sentiment matrices and leakage audit')
    features.add_argument('--dataset', required=True, type=Path)
    features.add_argument('--sentiment', required=True, type=Path, help='M3 ticker/session sentiment output directory')
    features.add_argument('--config', type=Path, default=Path('configs/project.toml'))
    features.add_argument('--features', type=Path, default=Path('configs/features.toml'))
    features.add_argument('--out', required=True, type=Path)
    train = sub.add_parser('train-models', help='Fit M5 models/baselines on development data and freeze choices')
    train.add_argument('--features', required=True, type=Path)
    train.add_argument('--modeling', type=Path, default=Path('configs/modeling.toml'))
    train.add_argument('--out', required=True, type=Path)
    holdout = sub.add_parser('predict-holdout', help='Post-freeze prediction only; never opens test label files')
    holdout.add_argument('--features', required=True, type=Path)
    holdout.add_argument('--experiment', required=True, type=Path)
    holdout.add_argument('--out', required=True, type=Path)
    evaluate = sub.add_parser('evaluate', help='M6: verify and evaluate frozen test predictions with paired date-block intervals')
    evaluate.add_argument('--features', required=True, type=Path)
    evaluate.add_argument('--experiment', required=True, type=Path)
    evaluate.add_argument('--predictions', required=True, type=Path)
    evaluate.add_argument('--out', required=True, type=Path)
    release = sub.add_parser('finalize', help='M7: build the report, cards, figures and reproducible local handoff')
    release.add_argument('--features', required=True, type=Path)
    release.add_argument('--experiment', required=True, type=Path)
    release.add_argument('--predictions', required=True, type=Path)
    release.add_argument('--evaluation', required=True, type=Path)
    release.add_argument('--out', required=True, type=Path)
    purge.add_argument('--sentiment-cache', type=Path, default=Path('data/cache/sentiment.sqlite'),
                       help='Clear derived score cache when purging a source record')
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            config = Config.load(args.config)
            result = {"status": "ok", "version": __version__, "config": config.to_dict(),
                      "network_required": False, "python": platform.python_version()}
        elif args.command == "demo":
            with ExitStack() as stack:
                fixture = files("sentiment_lab").joinpath("fixtures")
                paths = [stack.enter_context(as_file(fixture.joinpath(name))) for name in
                         ("demo.toml", "sessions.csv", "prices.csv", "texts.jsonl")]
                result = run(*paths, args.out, "synthetic_demo")
        elif args.command == 'calendar':
            from .calendar import generate_calendar
            from .contracts import session_date
            from .acquisition import write_csv
            if args.out.exists() and any(args.out.iterdir()):
                raise ContractError('Calendar output directory must be empty')
            sessions, metadata = generate_calendar(session_date(args.start), session_date(args.end))
            args.out.mkdir(parents=True, exist_ok=True)
            write_csv(args.out / 'sessions.csv', [{'session': str(s.date), 'open_at': s.open_at.isoformat(), 'close_at': s.close_at.isoformat()} for s in sessions], ['session', 'open_at', 'close_at'])
            write_json(args.out / 'calendar.json', metadata)
            result = {'output': str(args.out.resolve()), **metadata}
        elif args.command == 'verify-calendar':
            from .calendar import verify_calendar
            from .contracts import session_date, Session
            from .io import read_csv
            result = verify_calendar(read_csv(args.sessions, Session), session_date(args.start), session_date(args.end))
        elif args.command == 'collect':
            from .acquisition import collect
            result = collect(args.config, args.acquisition, args.cache, args.out,
                             args.refresh_prices, args.refresh_news, args.offline)
        elif args.command == 'import-text':
            from .normalization import import_text
            from .store import Store
            config = Config.load(args.config)
            with Store(args.cache) as store:
                store.bind(config.to_dict())
                result = import_text(args.input, args.provenance, config, store)
        elif args.command == 'purge-text':
            from .store import Store
            with Store(args.cache) as store:
                result = {'deleted_versions': store.purge(args.source, args.origin_id),
                          'exports_require_removal_or_regeneration': True}
            if args.sentiment_cache.exists():
                from .sentiment_cache import SentimentCache
                with SentimentCache(args.sentiment_cache, synthetic=False) as cache:
                    result['cleared_sentiment_scores'] = cache.clear()
        elif args.command == 'm2-demo':
            from .acquisition_demo import demo
            result = demo(args.out, args.cache)
        elif args.command == 'prepare-finbert':
            from .finbert import SentimentConfig, prepare_model
            result = prepare_model(SentimentConfig.load(args.sentiment), args.model_root, offline=args.offline)
        elif args.command == 'score-sentiment':
            from .sentiment import run_sentiment
            result = run_sentiment(args.dataset, args.config, args.sentiment, args.model_root,
                                   args.cache, args.out, offline_model=not args.download_model)
        elif args.command == 'sentiment-smoke':
            from .sentiment import run_smoke
            result = run_smoke(args.sentiment, args.model_root, args.cache, args.out,
                               offline_model=not args.download_model)
        elif args.command == 'build-features':
            from .features import run_features
            result = run_features(args.dataset, args.sentiment, args.config, args.features, args.out)
        elif args.command == 'train-models':
            from .modeling import train_models
            result = train_models(args.features, args.modeling, args.out)
        elif args.command == 'predict-holdout':
            from .modeling import predict_holdout
            result = predict_holdout(args.features, args.experiment, args.out)
        elif args.command == 'evaluate':
            from .evaluation import evaluate
            result = evaluate(args.features, args.experiment, args.predictions, args.out)
        elif args.command == 'finalize':
            from .release import finalize
            result = finalize(args.features, args.experiment, args.predictions, args.evaluation, args.out)
        else:
            result = run(args.config, args.sessions, args.prices, args.texts, args.out, "local_import_unverified")
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.command == 'collect' and args.require_feasible and result['feasibility'] != 'full_study_feasible':
            return 3
        return 0
    except (ContractError, OSError, tomllib.TOMLDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
