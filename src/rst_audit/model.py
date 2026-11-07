"""Rolling-origin ranking model used to generate an independent candidate list.

The model never sees a rule-state feature (ratio to the five-year mean, lagged
SI flags, or the mean level).  It is therefore a diagnostic device: it asks
which units a model would rank highly without being able to copy the rule.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C
from .features import FEATURE_SETS, load_master
from .io_utils import Logger, configure_stdout, write_csv


def ndcg_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    order = np.argsort(-score)[:k]
    rel = y[order].astype(float)
    discount = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    ideal = np.sort(y.astype(float))[::-1][:k]
    idcg = float((ideal * discount[:len(ideal)]).sum())
    return float((rel * discount).sum() / idcg) if idcg > 0 else 0.0


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score)
    yy = y[order]
    if yy.sum() == 0:
        return 0.0
    tp = np.cumsum(yy)
    precision = tp / np.arange(1, len(yy) + 1)
    return float((precision * yy).sum() / yy.sum())


def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score)
    out = {"n_test": int(len(y)), "n_pos": int(y.sum())}
    for budget in (50, 100, 200):
        k = min(budget, len(y))
        out[f"hit{budget}"] = int(y[order[:k]].sum())
        out[f"p{budget}"] = out[f"hit{budget}"] / k
    out["r100"] = out["hit100"] / out["n_pos"] if out["n_pos"] else 0.0
    out["ndcg100"] = ndcg_at_k(y, score, 100)
    out["ap"] = average_precision(y, score)
    return out


def random_expectation(y: np.ndarray, budget: int = 100, draws: int = 2000, seed: int = 0):
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    hits = np.empty(draws)
    for i in range(draws):
        idx = rng.choice(len(y), size=min(budget, len(y)), replace=False)
        hits[i] = y[idx].sum()
    return float(hits.mean()), float(np.quantile(hits, 0.025)), float(np.quantile(hits, 0.975))


def lgb_params(pos: int, n: int, seed: int) -> dict:
    params = dict(C.LGB_PARAMS)
    params.update(
        num_threads=C.N_THREADS,
        seed=seed,
        bagging_seed=seed + 1,
        feature_fraction_seed=seed + 2,
        scale_pos_weight=(n - max(pos, 1)) / max(pos, 1),
    )
    return params


def fit_predict_lgb(X_train, y_train, X_test, seed: int):
    import lightgbm as lgb

    model = lgb.train(
        lgb_params(int(y_train.sum()), len(y_train), seed),
        lgb.Dataset(X_train, label=y_train),
        num_boost_round=C.N_ROUNDS,
    )
    return model, model.predict(X_test)


def run_models(years: list[int] | None = None, seeds: list[int] | None = None,
               feature_set: str = C.FEATURE_SET, label: str = C.LABEL,
               force: bool = False) -> pd.DataFrame:
    configure_stdout()
    C.ensure_dirs()
    log = Logger(C.LOG_DIR / "run_models.log")
    t0 = time.time()
    years = years or C.TEST_YEARS
    seeds = seeds or C.SEEDS
    master = load_master()
    cols = FEATURE_SETS[feature_set]
    rows = []
    for T in years:
        train = master[master["year"] < T]
        test = master[master["year"] == T]
        if test.empty:
            raise RuntimeError(f"empty test frame for {T}")
        y_train = train[f"y_{label}"].values.astype(int)
        y_test = test[f"y_{label}"].values.astype(int)
        X_train = np.ascontiguousarray(train[cols].to_numpy(dtype=np.float32))
        X_test = np.ascontiguousarray(test[cols].to_numpy(dtype=np.float32))
        for seed in seeds:
            path = C.score_path(label, feature_set, T, seed)
            if path.exists() and not force:
                log(f"reuse {path.name}")
                score = pd.read_parquet(path)["score"].values
            else:
                model, score = fit_predict_lgb(X_train, y_train, X_test, seed)
                pd.DataFrame({
                    "exporter": test["exporter"].values,
                    "taxon": test["taxon"].values,
                    "score": score,
                    "y": y_test,
                    "y_B3": test["y_B3"].values.astype(int),
                }).to_parquet(path, index=False)
                log(f"fit T={T} seed={seed} -> {path.name}")
            mt = metrics(y_test, score)
            rand_mean, rand_lo, rand_hi = random_expectation(y_test, 100, seed=seed)
            rows.append(dict(
                label=label, feature_set=feature_set, test_year=T, seed=seed,
                **mt, random_mean=rand_mean, random_lo=rand_lo, random_hi=rand_hi,
                lift100=mt["hit100"] / rand_mean if rand_mean > 0 else np.nan,
            ))
    results = pd.DataFrame(rows)
    write_csv(results, C.RES_DIR / "model_results.csv")
    log(f"models done in {time.time() - t0:.1f}s")
    log.close()
    return results
