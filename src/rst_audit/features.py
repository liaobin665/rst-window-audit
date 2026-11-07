"""Features known at the end of year t (used to predict label year T = t + 1).

Four blocks are produced: trade history, network position, trade form, and the
explicit rule-state block.  The paper's primary model deliberately drops the
last block (plus the two five-year-mean levels and the EN flag), leaving 60
features that do not reconstruct the sharp-increase rule.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C
from .io_utils import Logger, configure_stdout, write_json
from .labels import roll_sum

EPS = 1e-9
YEARS = list(range(C.YEAR_MIN, C.YEAR_MAX + 1))
FEAT_YEARS = list(range(2004, 2024))

VOL = [
    "v_log1p", "v_lag2_log1p", "m5_log1p", "log5sum", "active5", "recency", "cv5",
    "slope5", "max_over_mean5", "zero_years5", "log_records", "qty_per_record",
    "last_share5", "v_allunit_ratio", "v_lag3_log1p", "v_sum_lag3_log1p", "v_std_log",
]
GRAPH = [
    "deg_exp", "deg_sp", "share_global", "hhi_sp", "peer_vol_log", "peer_growth",
    "exp_total_log", "exp_entropy", "exp_growth", "jac_max", "jac_mean", "aa_peer",
    "genus_countries", "genus_vol_share", "partner_country_n", "growth_rank_sp",
    "deg_exp_chg", "sp_growth",
]
FORM = [
    "term_entropy", "live_share", "top_term_share", "terms_n", "importers_n",
    "top_importer_share", "origin_n", "src_W", "src_C", "src_F", "src_R", "src_D",
    "src_A", "src_U", "v_wild_share", "p_T", "p_P", "p_H", "reexport_share",
    "appendix_code", "class_code", "order_code", "family_code", "genus_code",
    "is_spp", "is_hybrid", "taxon_n_tokens",
]
RULE = ["r1", "r_prev", "max_r3", "si_lag1", "si_lag2", "hv200_lag"]
STATIC = [
    "appendix_code", "class_code", "order_code", "family_code", "genus_code",
    "is_spp", "is_hybrid", "taxon_n_tokens",
]

GROUP_HVMEAN = ["m5_log1p", "log5sum"]
GROUP_VOL = [c for c in VOL if c not in GROUP_HVMEAN]
GROUP_FORM = [c for c in FORM if c not in STATIC]

FEATURE_GROUPS = {
    "vol": VOL,
    "graph": GRAPH,
    "form": FORM,
    "rule": RULE,
}

ALL_FEATURES = VOL + GRAPH + FORM + RULE

FEATURE_SETS = {
    "full": ["en_flag"] + RULE + GROUP_HVMEAN + GROUP_VOL + GRAPH + GROUP_FORM + STATIC,
    "no_leak_soft": RULE + GROUP_HVMEAN + GROUP_VOL + GRAPH + GROUP_FORM + STATIC,
    "no_leak_strict": GROUP_VOL + GRAPH + GROUP_FORM + STATIC,
    "no_leak_strict_plus": [
        c for c in GROUP_VOL + GRAPH + GROUP_FORM + STATIC
        if c not in ("family_code", "genus_code")
    ],
}
assert len(FEATURE_SETS["full"]) == 69
assert len(FEATURE_SETS["no_leak_strict"]) == 60


def _entropy_rows(weights: np.ndarray, group_index: np.ndarray, n_groups: int) -> np.ndarray:
    total = np.bincount(group_index, weights=weights, minlength=n_groups)
    p = weights / np.maximum(total[group_index], EPS)
    contrib = np.where(p > 0, p * np.log(np.maximum(p, EPS)), 0.0)
    return -np.bincount(group_index, weights=contrib, minlength=n_groups)


def _str_pivot(p: pd.DataFrame, col: str, n_units: int, n_years: int) -> np.ndarray:
    t = p.pivot_table(index="uid", columns="yi", values=col, aggfunc="first")
    t = t.reindex(index=range(n_units), columns=range(n_years))
    return t.ffill(axis=1).fillna("UNK").values  # forward fill only: no future leakage


def build_features(force: bool = False) -> None:
    configure_stdout()
    C.ensure_dirs()
    log = Logger(C.LOG_DIR / "build_features.log")
    t0 = time.time()
    needed = [C.DATA_DIR / f"feat_T{t}.parquet" for t in FEAT_YEARS]
    if all(p.exists() for p in needed) and not force:
        log("feature files exist, reuse (pass --force to rebuild)")
        return
    if not C.PANEL.exists():
        raise FileNotFoundError(f"missing panel: {C.PANEL}; run build_panel first")

    panel = pd.read_parquet(C.PANEL)
    units = panel[["exporter", "taxon"]].drop_duplicates().reset_index(drop=True)
    units["uid"] = np.arange(len(units))
    units["genus"] = units["taxon"].map(panel.groupby("taxon")["genus"].first()).fillna("UNK")
    p = units.merge(panel, on=["exporter", "taxon"], how="left")
    p["yi"] = p["year"] - YEARS[0]
    n_units, n_years = len(units), len(YEARS)

    def mat(col: str, fill: float = 0.0) -> np.ndarray:
        out = np.full((n_units, n_years), fill, dtype=np.float64)
        ok = p[col].notna().values
        out[p.loc[ok, "uid"].values, p.loc[ok, "yi"].values] = p.loc[ok, col].values
        return out

    V = mat("v_cnt")
    REC = mat("records")
    VA = mat("v_allunit")
    VWILD = mat("v_wild")
    VREX = mat("v_reexport")

    sp_ids, sp_inv = np.unique(units["taxon"].values, return_inverse=True)
    ex_ids, ex_inv = np.unique(units["exporter"].values, return_inverse=True)
    gen_ids, gen_inv = np.unique(units["genus"].values, return_inverse=True)
    n_sp, n_ex, n_gen = len(sp_ids), len(ex_ids), len(gen_ids)

    S5 = roll_sum(V, -4, 1)                  # V(t-4..t)
    M5 = S5 / 5.0
    M5p = roll_sum(V, -5, 0) / 5.0           # mean before the current year
    SD5 = np.sqrt(np.maximum(roll_sum(V ** 2, -4, 1) / 5.0 - M5 ** 2, 0))
    ACT5 = roll_sum((V > 0).astype(float), -4, 1)
    zero5 = 5 - ACT5

    positive = V > 0
    recency = np.zeros((n_units, n_years))
    last = np.full(n_units, -99)
    for j in range(n_years):
        last = np.where(positive[:, j], j, last)
        recency[:, j] = np.where(last >= 0, j - last, 25)

    x = np.arange(5) - 2.0
    slope = np.zeros((n_units, n_years))
    for j in range(n_years):
        a = max(0, j - 4)
        window = V[:, a:j + 1]
        if window.shape[1] == 5:
            slope[:, j] = (window * x).sum(axis=1) / 10.0
        else:
            k = window.shape[1]
            xx = np.arange(k) - (k - 1) / 2.0
            slope[:, j] = (window * xx).sum(axis=1) / max((xx ** 2).sum(), 1e-9)

    # Kept identical to the analysis that produced the published scores: the
    # denominator is one fifth of the five-year mean.  Only the within-species
    # percentile (growth_rank_sp) is derived from it, and that percentile is
    # scale-invariant except where the denominator hits the EPS floor.
    r1 = V / np.maximum(M5p / 5.0, EPS)
    r_prev = np.zeros((n_units, n_years))
    r_prev[:, 1:] = V[:, :-1] / np.maximum(M5p[:, :-1], EPS)
    r_prev2 = np.zeros((n_units, n_years))
    r_prev2[:, 2:] = V[:, :-2] / np.maximum(M5p[:, :-2], EPS)
    max_r3 = np.maximum(np.maximum(r1, r_prev), r_prev2)

    si1 = (V > C.SI_MULT * np.maximum(M5p, EPS)) & (M5p >= C.SI_MIN_MEAN)
    si_lag1 = np.zeros((n_units, n_years))
    si_lag2 = np.zeros((n_units, n_years))
    si_lag1[:, 1:] = si1[:, :-1]
    si_lag2[:, 2:] = si1[:, :-2]
    hv200_lag = np.zeros((n_units, n_years))
    hv200_lag[:, 1:] = M5p[:, :-1] > 200.0

    ex_sum = np.zeros((n_ex, n_years))
    sp_sum = np.zeros((n_sp, n_years))
    ex_S5 = np.zeros((n_ex, n_years))
    sp_S5 = np.zeros((n_sp, n_years))
    for j in range(n_years):
        ex_sum[:, j] = np.bincount(ex_inv, weights=V[:, j], minlength=n_ex)
        sp_sum[:, j] = np.bincount(sp_inv, weights=V[:, j], minlength=n_sp)
        ex_S5[:, j] = np.bincount(ex_inv, weights=S5[:, j], minlength=n_ex)
        sp_S5[:, j] = np.bincount(sp_inv, weights=S5[:, j], minlength=n_sp)

    appendix = _str_pivot(p, "appendix", n_units, n_years)
    app_map = {"I": 3, "II": 2, "III": 1, "N": 0, "UNK": -1}
    ap_code = np.vectorize(lambda s: app_map.get(s, -1))(appendix).astype(float)

    def codes(series: pd.Series) -> np.ndarray:
        values = series.astype(str).values
        order = {v: i for i, v in enumerate(pd.Series(values).value_counts().index)}
        return np.array([order[v] for v in values], dtype=float)

    cls_code = codes(units["taxon"].map(panel.groupby("taxon")["cls"].first()).fillna("UNK"))
    ord_code = codes(units["taxon"].map(panel.groupby("taxon")["ord"].first()).fillna("UNK"))
    fam_code = codes(units["taxon"].map(panel.groupby("taxon")["family"].first()).fillna("UNK"))
    gen_code = codes(units["genus"])

    tokens = units["taxon"].str.split()
    n_tok = tokens.str.len().astype(float).values
    is_spp = units["taxon"].str.contains(r"spp\.?", case=False, regex=True).astype(float).values
    is_hyb = units["taxon"].str.contains(r"\bx\b|hybrid|cultivar", case=False, regex=True).astype(float).values

    ctx = {
        c: mat(c) for c in [
            "term_entropy", "live_share", "top_term_share", "terms_n", "importers_n",
            "top_importer_share", "origin_n", "p_T", "p_P", "p_H",
        ]
    }
    for c in ["term_entropy", "live_share", "top_term_share", "top_importer_share"]:
        ctx[c] = np.nan_to_num(ctx[c], nan=-1.0)
    srcM = {s: mat(f"src_{s}") for s in C.SOURCES}

    write_json(FEATURE_GROUPS, C.DATA_DIR / "feature_groups.json")
    for t in FEAT_YEARS:
        j = t - YEARS[0]
        alive = V[:, j] > 0
        ex_deg = np.bincount(ex_inv[alive], minlength=n_ex).astype(float)
        sp_deg = np.bincount(sp_inv[alive], minlength=n_sp).astype(float)

        csets = [set() for _ in range(n_ex)]
        for uid in np.where(alive)[0]:
            csets[ex_inv[uid]].add(int(sp_inv[uid]))
        sp_members: dict[int, list[int]] = {}
        for uid in np.where(alive)[0]:
            sp_members.setdefault(int(sp_inv[uid]), []).append(int(ex_inv[uid]))

        jac_max = np.zeros(n_units)
        jac_mean = np.zeros(n_units)
        aa_peer = np.zeros(n_units)
        for s, members in sp_members.items():
            members = sorted(members)
            if len(members) < 2:
                continue
            for ii, a in enumerate(members):
                others = members[:ii] + members[ii + 1:]
                inter = [len(csets[a] & csets[b]) for b in others]
                union = [len(csets[a] | csets[b]) for b in others]
                jac = [i / u if u else 0.0 for i, u in zip(inter, union)]
                uids_a = [u for u in np.where(alive)[0] if ex_inv[u] == a and sp_inv[u] == s]
                if not uids_a:
                    continue
                ua = uids_a[0]
                jac_max[ua] = max(jac)
                jac_mean[ua] = float(np.mean(jac))
                aa = 0.0
                for b in others:
                    common = csets[a] & csets[b]
                    aa += sum(1.0 / np.log(1.0 + sp_deg[cs]) for cs in common if sp_deg[cs] > 0)
                aa_peer[ua] = aa / len(others)

        partner_n = np.zeros(n_ex)
        for a in range(n_ex):
            if csets[a]:
                partner_n[a] = sum(1 for b in range(n_ex) if b != a and csets[a] & csets[b])

        v = V[:, j]
        share = v / np.maximum(sp_sum[sp_inv, j], EPS)
        hhi = np.zeros(n_sp)
        np.add.at(hhi, sp_inv[alive], share[alive] ** 2)
        ent_ex = _entropy_rows(v, ex_inv, n_ex)

        dfr = pd.DataFrame({"sp": sp_inv, "v": v, "r": r1[:, j], "u": np.arange(n_units)})
        rank = dfr[dfr["v"] > 0].groupby("sp")["r"].rank(pct=True)
        growth_rank = np.full(n_units, np.nan)
        growth_rank[dfr[dfr["v"] > 0]["u"].values] = rank.values

        gen_sum = np.bincount(gen_inv, weights=v, minlength=n_gen)
        F = pd.DataFrame(index=np.arange(n_units))
        F["v_log1p"] = np.log1p(v)
        F["v_lag2_log1p"] = np.log1p(V[:, max(j - 1, 0)])
        F["m5_log1p"] = np.log1p(M5p[:, j])
        F["log5sum"] = np.log1p(S5[:, j])
        F["active5"] = ACT5[:, j]
        F["recency"] = recency[:, j]
        F["cv5"] = SD5[:, j] / np.maximum(M5[:, j], 1.0)
        F["slope5"] = slope[:, j] / np.maximum(M5[:, j], 1.0)
        F["max_over_mean5"] = np.max(V[:, max(0, j - 4):j + 1], axis=1) / np.maximum(M5[:, j], 1.0)
        F["zero_years5"] = zero5[:, j]
        F["log_records"] = np.log1p(REC[:, j])
        F["qty_per_record"] = v / np.maximum(REC[:, j], 1)
        F["last_share5"] = v / np.maximum(S5[:, j], 1)
        F["v_allunit_ratio"] = VA[:, j] / np.maximum(v, 1)
        F["v_lag3_log1p"] = np.log1p(V[:, max(j - 2, 0)])
        F["v_sum_lag3_log1p"] = np.log1p(roll_sum(V, -3, 0)[:, j])
        F["v_std_log"] = np.log1p(np.sqrt(np.maximum(
            roll_sum(V ** 2, -4, 1)[:, j] / 5 - M5[:, j] ** 2, 0)))

        F["deg_exp"] = ex_deg[ex_inv]
        F["deg_sp"] = sp_deg[sp_inv]
        F["share_global"] = share
        F["hhi_sp"] = hhi[sp_inv]
        F["peer_vol_log"] = np.log1p(np.maximum(sp_sum[sp_inv, j] - v, 0))
        F["peer_growth"] = (sp_sum[sp_inv, j] - v) / np.maximum(sp_S5[sp_inv, j] / 5.0 - v, 1.0)
        F["exp_total_log"] = np.log1p(ex_sum[ex_inv, j])
        F["exp_entropy"] = ent_ex[ex_inv]
        F["exp_growth"] = ex_sum[ex_inv, j] / np.maximum(ex_S5[ex_inv, j] / 5.0, 1.0)
        F["jac_max"] = jac_max
        F["jac_mean"] = jac_mean
        F["aa_peer"] = aa_peer
        F["genus_countries"] = np.bincount(gen_inv[alive], minlength=n_gen)[gen_inv].astype(float)
        F["genus_vol_share"] = v / np.maximum(gen_sum[gen_inv], EPS)
        F["partner_country_n"] = partner_n[ex_inv]
        F["growth_rank_sp"] = growth_rank
        F["deg_exp_chg"] = ex_deg[ex_inv] - np.bincount(
            ex_inv[V[:, max(j - 1, 0)] > 0], minlength=n_ex)[ex_inv]
        F["sp_growth"] = sp_sum[sp_inv, j] / np.maximum(sp_S5[sp_inv, j] / 5.0, 1.0)

        for c in ["term_entropy", "live_share", "top_term_share", "terms_n", "importers_n",
                  "top_importer_share", "origin_n", "p_T", "p_P", "p_H"]:
            F[c] = ctx[c][:, j]
        for s in C.SOURCES[:7]:
            F[f"src_{s}"] = srcM[s][:, j] / np.maximum(v, 1)
        F["v_wild_share"] = VWILD[:, j] / np.maximum(v, 1)
        F["reexport_share"] = VREX[:, j] / np.maximum(v + VREX[:, j], 1)
        F["appendix_code"] = ap_code[:, j]
        F["class_code"] = cls_code
        F["order_code"] = ord_code
        F["family_code"] = fam_code
        F["genus_code"] = gen_code
        F["is_spp"] = is_spp
        F["is_hybrid"] = is_hyb
        F["taxon_n_tokens"] = n_tok

        F["r1"] = r1[:, j]
        F["r_prev"] = r_prev[:, j]
        F["max_r3"] = max_r3[:, j]
        F["si_lag1"] = si_lag1[:, j]
        F["si_lag2"] = si_lag2[:, j]
        F["hv200_lag"] = hv200_lag[:, j]

        assert set(F.columns) == set(ALL_FEATURES)
        F = F[ALL_FEATURES].replace([np.inf, -np.inf], np.nan)
        out = pd.concat([units[["exporter", "taxon"]].reset_index(drop=True), F], axis=1)
        out["feat_year"] = t
        assert out["feat_year"].max() == t < C.YEAR_MAX
        out.to_parquet(C.DATA_DIR / f"feat_T{t}.parquet", index=False)
        if t % 5 == 0 or t == FEAT_YEARS[-1]:
            log(f"t={t} rows={len(out):,} nan_rate={float(F.isna().mean().mean()):.3f} "
                f"elapsed={time.time() - t0:.0f}s")
    log("features complete")


def load_master() -> pd.DataFrame:
    """One row per (exporter, taxon, label year) with 69 features and labels."""
    lab = pd.read_parquet(C.LABELS)
    keep = ["exporter", "taxon", "year", "y_B", "y_B3", "si_c", "si_g_u",
            "si_3x", "si3x_3y", "hv200", "source_shift", "report_disc_relax",
            "en", "m5", "v_cnt", "active5"]
    lab = lab[keep].rename(columns={"active5": "lab_active5"})
    parts = []
    for year in range(C.TRAIN_START, C.YEAR_MAX):
        feat = pd.read_parquet(C.DATA_DIR / f"feat_T{year - 1}.parquet")
        labels = lab[lab["year"] == year]
        merged = feat.merge(labels, on=["exporter", "taxon"], how="inner", validate="one_to_one")
        if len(merged) != len(labels):
            raise RuntimeError(f"merge loss at label year {year}: {len(merged)} vs {len(labels)}")
        assert (merged["feat_year"] == year - 1).all()
        parts.append(merged)
    master = pd.concat(parts, ignore_index=True)
    master["en_flag"] = master["en"].astype(np.float32)
    for col in FEATURE_SETS["full"]:
        master[col] = master[col].astype(np.float32)
    return master
