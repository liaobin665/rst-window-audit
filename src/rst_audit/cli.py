"""Command-line entry point.

Examples
--------
    rst-audit reproduce                 # figures + checks from shipped tables
    rst-audit checks                    # recompute the paper's headline numbers
    rst-audit figures                   # redraw the five manuscript figures
    rst-audit pipeline --skip-models    # data -> labels -> features -> audit
    CITES_DB=/path/cites.duckdb rst-audit pipeline
"""
from __future__ import annotations

import argparse
import sys
import time

from . import config as C
from .io_utils import configure_stdout


def _panel(args) -> None:
    from .panel import build_panel

    build_panel(force=args.force)


def _labels(args) -> None:
    from .labels import build_labels

    build_labels(force=args.force)


def _features(args) -> None:
    from .features import build_features

    build_features(force=args.force)


def _models(args) -> None:
    from .model import run_models

    run_models(seeds=args.seeds, force=args.force)


def _audit(args) -> None:
    from .audit import run_audit

    run_audit(force=args.force)


def _sensitivity(args) -> None:
    import pandas as pd

    from .audit import build_audit_pool
    from .sensitivity import run_sensitivity

    panel = pd.read_parquet(C.PANEL)
    labels = pd.read_parquet(C.LABELS)
    pool = build_audit_pool(labels, panel, seed=0)
    run_sensitivity(panel, labels, pool, skip_models=args.skip_models)


def _figures(args) -> None:
    from .figures import build_all

    print(", ".join(build_all()))


def _checks(args) -> None:
    from .checks import run_checks

    run_checks(strict=not args.no_strict, freeze=args.freeze)


def _reproduce(args) -> None:
    from .checks import run_checks
    from .figures import build_all

    names = build_all()
    print(f"figures: {', '.join(names)}")
    run_checks(strict=not args.no_strict)


def _pipeline(args) -> None:
    import pandas as pd

    from .audit import build_audit_pool, run_audit
    from .features import build_features
    from .labels import build_labels
    from .model import run_models
    from .panel import build_panel
    from .sensitivity import run_sensitivity

    t0 = time.time()
    build_panel(force=args.force)
    build_labels(force=args.force)
    build_features(force=args.force)
    if not args.skip_models:
        run_models(seeds=args.seeds, force=args.force)
    run_audit(force=args.force)
    if not args.skip_sensitivity:
        panel = pd.read_parquet(C.PANEL)
        labels = pd.read_parquet(C.LABELS)
        pool = build_audit_pool(labels, panel, seed=0)
        run_sensitivity(panel, labels, pool, skip_models=args.skip_models)
    print(f"pipeline done in {time.time() - t0:.1f}s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rst-audit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("panel", help="build the exporter x taxon panel from CITES DuckDB")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_panel)

    p = sub.add_parser("labels", help="build rule states and audit labels")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_labels)

    p = sub.add_parser("features", help="build the model matrix")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_features)

    p = sub.add_parser("models", help="fit rolling-origin LightGBM scores")
    p.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_models)

    p = sub.add_parser("audit", help="fixed-budget audit and case verification")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_audit)

    p = sub.add_parser("sensitivity", help="threshold, model-family and field checks")
    p.add_argument("--skip-models", action="store_true")
    p.set_defaults(func=_sensitivity)

    p = sub.add_parser("figures", help="redraw figures from shipped tables")
    p.set_defaults(func=_figures)

    p = sub.add_parser("checks", help="recompute headline numbers against frozen expectations")
    p.add_argument("--no-strict", action="store_true")
    p.add_argument("--freeze", action="store_true")
    p.set_defaults(func=_checks)

    p = sub.add_parser("reproduce", help="figures + checks, no data download")
    p.add_argument("--no-strict", action="store_true")
    p.set_defaults(func=_reproduce)

    p = sub.add_parser("pipeline", help="full chain from the official CITES export")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-models", action="store_true")
    p.add_argument("--skip-sensitivity", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    p.set_defaults(func=_pipeline)

    args = ap.parse_args(argv)
    configure_stdout()
    C.ensure_dirs()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
