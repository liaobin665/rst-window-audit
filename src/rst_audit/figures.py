"""Publication figures, rebuilt from the shipped aggregate tables.

The figures read only files under ``results/``; they never touch the raw CITES
database.  That makes ``rst-audit figures`` runnable on a laptop with no data
download, and it keeps the figure layer auditable: every number on a panel is
either read from a shipped table or computed from one in this module.

Palette (colour-blind-safe blue/orange axis):
    navy #1F4E79, mid blue #2E75B6, light blue #9DC3E6, orange #D9730D,
    orange #F4B183, neutral greys.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from . import config as C
from .io_utils import configure_stdout, read_csv

W_DOUBLE = 6.54  # printed text width (inches) used by the manuscript
pt = 1.0

NAVY = "#1F4E79"
NAVY_MID = "#2E75B6"
NAVY_LT = "#9DC3E6"
WASH = "#EAF1F7"
ORANGE = "#D9730D"
ORANGE_LT = "#F4B183"
GREY = "#D9D9D9"
GREY_LT = "#F2F2F2"
GREY_M = "#A6A6A6"
GREY_D = "#595959"
INK = "#262626"

FAMILY = {
    "three_year_window": "Window blind spot",
    "outside_high_volume": "Outside the rule",
    "outside_source_shift": "Outside the rule",
    "outside_report_disc": "Outside the rule",
    "near_threshold_4x": "Other",
    "min_mean_gate": "Other",
    "new_flow": "Other",
    "relaxed_3y": "Other",
    "no_clear_reason": "Other",
}
FAM_ORDER = ["Window blind spot", "Outside the rule", "Other"]
FAM_COLOR = {"Window blind spot": NAVY, "Outside the rule": "#8C8C8C", "Other": GREY}


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def panel_tag(ax, letter: str, dx: float = -0.10, dy: float = 1.02) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, fontweight="bold", color=INK)


def _save(fig, name: str) -> None:
    """Refuse to save a figure whose smallest text is below the 7 pt floor."""
    sizes = []
    for ax in fig.axes:
        sizes += [t.get_fontsize() for t in ax.texts]
        sizes += [ax.title.get_fontsize(), ax.xaxis.label.get_fontsize(),
                  ax.yaxis.label.get_fontsize()]
        sizes += [t.get_fontsize() for t in ax.get_xticklabels()]
        sizes += [t.get_fontsize() for t in ax.get_yticklabels()]
        leg = ax.get_legend()
        if leg is not None:
            sizes += [t.get_fontsize() for t in leg.get_texts()]
            if leg.get_title() is not None:
                sizes.append(leg.get_title().get_fontsize())
    sizes += [t.get_fontsize() for t in fig.texts]
    lo = min(sizes)
    if lo < 7:
        raise RuntimeError(f"{name}: text below 7 pt ({lo})")
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(C.FIG_DIR / f"{name}.{ext}", dpi=600, bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)


def _table(name: str) -> pd.DataFrame:
    path = C.RES_DIR / "tables" / name
    if not path.exists():
        raise FileNotFoundError(f"missing shipped table: {path}")
    return read_csv(path)


def _figure_data(name: str) -> pd.DataFrame:
    path = C.RES_DIR / "figure_data" / name
    if not path.exists():
        raise FileNotFoundError(f"missing shipped figure data: {path}")
    return read_csv(path)


# --------------------------------------------------------------- figA0
def fig_flagship() -> None:
    cases = _table("table_cases.csv")
    agg = _table("table_window_comparison.csv")
    flags = _figure_data("case_flag_matrix.csv")
    series = _figure_data("case_series.csv")
    flags = flags.sort_values(["n_fire", "test_year", "exporter"],
                              ascending=[False, True, True]).reset_index(drop=True)
    n_cases = len(flags)

    fig = plt.figure(figsize=(W_DOUBLE, 5.55))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.00, 1.22, 0.90], hspace=0.62,
                          left=0.135, right=0.985, top=0.945, bottom=0.075)

    # A: one column per audited case, one row per lag year
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, n_cases + 8.5)
    ax.set_ylim(-0.10, 3.35)
    ax.axis("off")
    rows = [("T\u22122", "si_T2", 2), ("T\u22121", "si_T1", 1), ("T", "si_T", 0)]
    for j, (_, r) in enumerate(flags.iterrows()):
        for _, col, ry in rows:
            face = ORANGE if bool(r[col]) else "white"
            edge = "white" if bool(r[col]) else "#E3E3E3"
            ax.add_patch(Rectangle((j + 0.075, ry + 0.20), 0.85, 0.60,
                                   facecolor=face, edgecolor=edge, lw=0.5, zorder=3))
    counts = {col: int(flags[col].sum()) for _, col, _ in rows}
    for lab, col, ry in rows:
        ax.text(-0.45, ry + 0.50, lab, ha="right", va="center", fontsize=7.5, color=INK)
        n = counts[col]
        ax.text(n_cases + 0.45, ry + 0.50, f"{n}", ha="left", va="center", fontsize=8,
                fontweight="bold", color=ORANGE if n else GREY_M)

    n_dual = int((flags["n_fire"] == 2).sum())
    ax.plot([n_dual, n_dual], [0.06, 2.94], color=GREY_M, lw=0.7,
            ls=(0, (2, 2)), zorder=1)
    ax.text(n_dual / 2, 3.06, f"fires at T\u22121 and T\u22122  ({n_dual})",
            ha="center", va="bottom", fontsize=7, color=ORANGE)
    ax.text(n_dual + (n_cases - n_dual) / 2, 3.06,
            f"fires at T\u22121 only  ({n_cases - n_dual})",
            ha="center", va="bottom", fontsize=7, color=ORANGE)

    def bracket(x, y0, y1, colour, label):
        ax.plot([x - 0.55, x, x, x - 0.55], [y0, y0, y1, y1], color=colour, lw=0.9,
                clip_on=False)
        ax.text(x + 0.18, (y0 + y1) / 2, label, rotation=90, ha="left", va="center",
                fontsize=7, color=colour)

    bracket(n_cases + 3.2, 0.20, 0.80, NAVY, "one-year screen")
    bracket(n_cases + 5.6, 0.20, 2.80, NAVY, "three-year window, 2026")
    ax.set_title("Every audited case fires at T\u22121; not one fires at T",
                 fontsize=9, loc="left", x=0.005, pad=18, color=INK, fontweight="bold")
    ax.text(0, -0.02, f"one column = one of the {n_cases} audited cases; "
                      "filled = the 4\u00d7 test fires that year",
            ha="left", va="top", fontsize=7, color=GREY_D)
    panel_tag(ax, "A", dx=-0.075, dy=1.0)

    # B: one case in full
    ax = fig.add_subplot(gs[1])
    preferred = series[(series["exporter"] == "GT")
                       & (series["taxon"] == "Mammillaria plumosa")]
    if preferred.empty:
        key = series[["exporter", "taxon", "test_year"]].drop_duplicates().iloc[0]
        one = series[(series["exporter"] == key["exporter"])
                     & (series["taxon"] == key["taxon"])
                     & (series["test_year"] == key["test_year"])]
    else:
        one = preferred
        T = int(one["test_year"].max())
        one = one[one["test_year"] == T]
    one = one.sort_values("offset")
    offs = one["offset"].to_numpy(int)
    v = one["volume"].to_numpy(float)
    thr = one["threshold"].to_numpy(float)
    ax.axvspan(-2.52, 0.52, color=WASH, lw=0, zorder=0)
    cols = [ORANGE if o == -1 else (NAVY if o == 0 else GREY) for o in offs]
    ax.bar(offs, np.maximum(v, 1.0), width=0.66, color=cols, edgecolor="white",
           lw=0.5, zorder=3)
    xs, ys = [], []
    for o, t in zip(offs, thr):
        if t <= 0:
            continue
        xs += [o - 0.5, o + 0.5]
        ys += [t, t]
    ax.plot(xs, ys, color=GREY_D, lw=0.9, ls=(0, (3, 1.8)), zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(0.55, max(v.max() * 4.0, 1e3))
    ax.yaxis.set_minor_locator(mticker.LogLocator(subs=(2, 5)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks(list(offs))
    ax.set_xticklabels([f"T\u2212{abs(o)}" if o else "T" for o in offs])
    ax.set_xlim(-5.55, 0.55)
    ax.set_xlabel("Year relative to the focal year")
    ax.set_ylabel("Export volume (items)")
    for o, val in zip(offs, v):
        if val == 0:
            continue
        col = ORANGE if o == -1 else (NAVY if o == 0 else GREY_D)
        ax.text(o, val * 1.30, f"{val:,.0f}", ha="center", va="bottom", fontsize=7,
                color=col, fontweight="bold" if o in (-1, 0) else "normal")
    ax.text(-5.42, ax.get_ylim()[1] * 0.45,
            "dashed: 4\u00d7 the preceding five-year mean",
            ha="left", va="center", fontsize=7, color=GREY_D)
    row = cases[(cases["exporter"] == one["exporter"].iloc[0])
                & (cases["taxon"] == one["taxon"].iloc[0])
                & (cases["test_year"] == int(one["test_year"].iloc[0]))]
    trigger = str(row["trigger_lags"].iloc[0]).replace("|", " + ") if len(row) else ""
    ax.set_title(f"{one['exporter'].iloc[0]} \u00b7 {one['taxon'].iloc[0]}  "
                 f"(T = {int(one['test_year'].iloc[0])}; trigger {trigger})",
                 fontsize=8.5, loc="left", color=INK)
    panel_tag(ax, "B", dx=-0.105, dy=1.03)

    # C: same candidate list under each rule
    ax = fig.add_subplot(gs[2])
    r3 = agg[agg["arm"] == "3y_4x"].iloc[0]
    r2 = agg[agg["arm"] == "2y_4x"].iloc[0]
    r1 = agg[agg["arm"] == "1y_3x"].iloc[0]
    base = float(r3["n_base_flagged"])
    rows_c = [
        ("Three-year window", float(r3["n_additional_verified"]),
         float(r3["n_additional"]) - float(r3["n_additional_verified"]), NAVY, NAVY_LT),
        ("Two-year window", float(r2["n_additional_verified"]),
         float(r2["n_additional"]) - float(r2["n_additional_verified"]), NAVY_MID, NAVY_LT),
        ("3\u00d7 threshold", float(r1["n_additional_verified"]),
         float(r1["n_additional"]) - float(r1["n_additional_verified"]), ORANGE, ORANGE_LT),
    ]
    ypos = [2, 1, 0]
    for y in ypos:
        ax.barh(y, base, height=0.52, color="#C9C9C9", edgecolor="white", lw=0.5, zorder=3)
        ax.text(base / 2, y, f"{base:.1f}", ha="center", va="center", fontsize=8, color=INK)
    for y, (name, ver, unver, dark, light) in zip(ypos, rows_c):
        ax.barh(y, ver, left=base, height=0.52, color=dark, edgecolor="white", lw=0.5, zorder=3)
        ax.barh(y, unver, left=base + ver, height=0.52, color=light, edgecolor="white",
                lw=0.5, zorder=3)
        ax.text(base + ver / 2, y, f"{ver:.1f}", ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold")
        ax.text(base + ver + unver + 1.0, y, f"{base + ver + unver:.1f}", ha="left",
                va="center", fontsize=8, color=dark, fontweight="bold")
        ax.text(-0.8, y, name, ha="right", va="center", fontsize=7.5, color=INK)
    ratio = float(r3["n_additional_verified"]) / float(r1["n_additional_verified"])
    ax.text(base + 18.0, 0.0, f"{ratio:.1f}\u00d7 more\ninternally checked\nper year",
            ha="left", va="center", fontsize=7, color=GREY_D, fontweight="bold",
            linespacing=1.3)
    ax.set_xlim(0, 76)
    ax.set_ylim(-0.62, 2.55)
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("Triggered units inside the top-100 candidate list per year")
    ax.set_title("Same candidate list: the window adds internally checked cases",
                 fontsize=8.5, loc="left", color=INK)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_tag(ax, "C", dx=-0.13, dy=1.05)

    _save(fig, "figA0_flagship")


# --------------------------------------------------------------- figA2
def fig_composition() -> None:
    comp = _figure_data("divergence_composition.csv")
    cases = _table("table_cases.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 2.65),
                             gridspec_kw={"width_ratios": [1.30, 1]})
    ax = axes[0]
    labels = ["Pooled\nTop-50", "Per-year\nTop-10", "Unfiltered\nTop-50"]
    keys = ["pooled_top50", "per_year_top10", "unfiltered_top50"]
    data = []
    for key in keys:
        d = comp[comp["selection"] == key]
        g = d["business_code"].map(FAMILY).value_counts().reindex(FAM_ORDER).fillna(0)
        data.append(g.values)
    data = np.array(data)
    y = np.arange(len(keys))[::-1]
    left = np.zeros(len(keys))
    for name in FAM_ORDER:
        j = FAM_ORDER.index(name)
        vals = data[:, j]
        ax.barh(y, vals, left=left, height=0.56, color=FAM_COLOR[name],
                edgecolor="white", linewidth=0.6, label=name, zorder=3)
        for yy, l0, v in zip(y, left, vals):
            if v >= 4:
                ax.text(l0 + v / 2, yy, f"{int(v)}", ha="center", va="center", fontsize=7.5,
                        color="white" if name == "Window blind spot" else INK,
                        fontweight="bold" if name == "Window blind spot" else "normal")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 55)
    ax.set_ylim(-0.55, 2.75)
    ax.set_xlabel("Model-top combinations the one-year rule never flagged")
    ax.set_title("Window cases make up almost every defensible miss", fontsize=8.5,
                 loc="left", x=0.055, color=INK)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3,
              handlelength=1.1, columnspacing=1.0, frameon=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_tag(ax, "A", dx=-0.20)

    ax = axes[1]
    n_multi = int((cases["verdict"] == "confirmed_multi_base").sum())
    n_single = int((cases["verdict"] == "confirmed_single_base").sum())
    n_total = n_multi + n_single
    ax.set_xlim(-0.4, 14.4)
    ax.set_ylim(-0.15, 4.45)
    ax.axis("off")

    def unit_block(x0, n, face, edge, lw, label, sub):
        per_row = 6
        for i in range(n):
            r, c = divmod(i, per_row)
            ax.add_patch(Rectangle((x0 + c * 1.0, 2.55 - r * 1.0), 0.82, 0.82,
                                   facecolor=face, edgecolor=edge, lw=lw, zorder=3))
        ax.text(x0 + 0.05, 4.15, f"{n}", ha="left", va="center", fontsize=13,
                color=edge if face == "white" else face, fontweight="bold")
        ax.text(x0 + 0.05, 3.66, label, ha="left", va="center", fontsize=7.5, color=INK)
        ax.text(x0, 0.10, sub, ha="left", va="center", fontsize=7, color=GREY_D)

    unit_block(0.0, n_multi, NAVY, NAVY, 0.0, "multi-year", "confirmed")
    unit_block(7.6, n_single, "white", ORANGE, 1.2, "single-year", "watchlist")
    ax.set_title(f"{n_single} of the {n_total} rest on a single year", fontsize=8.5,
                 loc="left", x=0.055, color=INK)
    panel_tag(ax, "B", dx=-0.10)
    fig.tight_layout(w_pad=1.6)
    _save(fig, "figA2_divergence_structure")


# --------------------------------------------------------------- figA3
def fig_counterfactual() -> None:
    agg = _table("table_window_comparison.csv")
    by = _figure_data("window_by_year.csv")
    means = by.groupby("arm")["n_additional_verified"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.00),
                             gridspec_kw={"width_ratios": [1.30, 1]})

    ax = axes[0]
    r3 = agg[agg["arm"] == "3y_4x"].iloc[0]
    base = float(r3["n_base_flagged"])
    ver = float(r3["n_additional_verified"])
    multi_unver = float(r3["n_additional_multi_base"]) - ver
    single = float(r3["n_additional_single_base"])
    total = base + ver + multi_unver + single
    x = np.arange(5)
    bottoms = [0, base, base + ver, base + ver + multi_unver, 0]
    heights = [base, ver, multi_unver, single, total]
    cols = ["#C9C9C9", NAVY, NAVY_MID, NAVY_LT, NAVY]
    for i, (b, h, c) in enumerate(zip(bottoms, heights, cols)):
        ax.bar(x[i], h, bottom=b, width=0.62, color=c, edgecolor="white", lw=0.6, zorder=3)
    for i in range(4):
        yy = bottoms[i] + heights[i]
        ax.plot([x[i] + 0.31, x[i + 1] - 0.31], [yy, yy], color=GREY_M, lw=0.8,
                ls=(0, (2, 1.4)), zorder=2)
    ax.text(x[0], base / 2, f"{base:.1f}", ha="center", va="center", fontsize=8,
            color=INK, fontweight="bold")
    ax.text(x[1], base + ver / 2, f"+{ver:.1f}", ha="center", va="center", fontsize=8,
            color="white", fontweight="bold")
    ax.annotate(f"+{multi_unver:.1f}", xy=(x[2], base + ver + 1.2),
                xytext=(x[2] + 0.10, total * 0.85),
                ha="center", va="bottom", fontsize=7, color=GREY_D, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=GREY_D, lw=0.6))
    ax.text(x[3], base + ver + multi_unver + single / 2, f"+{single:.1f}", ha="center",
            va="center", fontsize=8, color=INK, fontweight="bold")
    ax.text(x[4], total / 2, f"{total:.1f}", ha="center", va="center", fontsize=8,
            color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["One-year\nbaseline", "Multi-year\nsupport", "Multi-year\nnot passed",
                        "Single-year\nsupport", "Three-year\nwindow"], fontsize=7)
    ax.set_ylabel("Triggered units per year\nin the top-100 candidate list")
    ax.set_ylim(0, max(total, 71) * 1.17)
    ax.set_title(f"Where the window's extra {total - base:.1f} comes from",
                 fontsize=8.5, loc="left", color=INK)
    r1 = agg[agg["arm"] == "1y_3x"].iloc[0]
    r33 = agg[agg["arm"] == "3y_3x"].iloc[0]
    ax.text(0.02, 0.97,
            f"3\u00d7 threshold alone: {float(r1['n_after']):.1f} "
            f"(+{float(r1['n_additional_verified']):.1f} checked)\n"
            f"Window + 3\u00d7: {float(r33['n_after']):.1f}, "
            f"same {float(r33['n_additional_verified']):.1f} checked",
            transform=ax.transAxes, ha="left", va="top", fontsize=7, color=GREY_D,
            linespacing=1.35)
    panel_tag(ax, "A", dx=-0.105, dy=1.03)

    ax = axes[1]
    jit = np.array([-0.09, -0.045, 0.0, 0.045, 0.09])
    for k, (arm, col) in enumerate([("3y_4x", NAVY), ("1y_3x", ORANGE)]):
        d = by[by["arm"] == arm].sort_values("test_year")
        ax.plot(k - 0.13 + jit, d["n_additional_verified"].values, "o", ms=3.4, color=col,
                mec="white", mew=0.4, zorder=3)
        m = float(means[arm])
        ax.plot([k], [m], "D", ms=4.6, color=col, mec="white", mew=0.5, zorder=5)
        ax.text(k + 0.20, m, f"{m:.1f}", ha="left", va="center",
                fontsize=7.5, color=col, fontweight="bold")
    ratio = float(means["3y_4x"]) / float(means["1y_3x"])
    ax.annotate("", xy=(1.44, means["3y_4x"]), xytext=(1.44, means["1y_3x"]),
                arrowprops=dict(arrowstyle="<|-|>", color=GREY_D, lw=0.9, ls="--"))
    ax.text(1.47, (means["3y_4x"] + means["1y_3x"]) / 2, f"{ratio:.1f}\u00d7",
            rotation=90, ha="left", va="center", fontsize=8, color=GREY_D,
            fontweight="bold")
    ax.set_xlim(-0.45, 1.82)
    ax.set_ylim(-1.6, max(34.0, float(means.max()) * 1.6))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Three-year\nwindow", "3\u00d7\nthreshold"])
    ax.set_ylabel("Checked additions per year")
    ax.set_title(f"{ratio:.1f}\u00d7 more checked cases per year", fontsize=8.5,
                 loc="left", color=INK)
    ax.text(0.02, 0.98, "Dots: the five test years.\nDiamond: mean.",
            transform=ax.transAxes, ha="left", va="top", fontsize=7, color=GREY_D,
            linespacing=1.3)
    panel_tag(ax, "B", dx=-0.18, dy=1.03)
    fig.tight_layout(w_pad=1.0)
    _save(fig, "figA3_counterfactual")


# --------------------------------------------------------------- figA4
def fig_timelines() -> None:
    tl = _figure_data("case_timelines.csv")
    series = _figure_data("case_series.csv")
    cases = _table("table_cases.csv")
    idx = cases.set_index(["exporter", "taxon", "test_year"])
    key = tl[["exporter", "taxon", "test_year"]].drop_duplicates().values.tolist()
    n = min(6, len(key))
    fig, axes = plt.subplots(2, 3, figsize=(W_DOUBLE, 3.55), sharex=True)
    letters = ["A", "B", "C", "D", "E", "F"]
    for ax, (ex, tx, T), letter in zip(axes.ravel(), key[:n], letters):
        T = int(T)
        sub = series[(series["exporter"] == ex) & (series["taxon"] == tx)
                     & (series["test_year"] == T)].sort_values("offset")
        offs = sub["offset"].to_numpy(int)
        v = sub["volume"].to_numpy(float)
        si = sub["fires"].to_numpy(bool)
        ax.axvspan(-2.5, 0.5, color=WASH, lw=0, zorder=0)
        ax.bar(offs, np.maximum(v, 1.0), width=0.68,
               color=[NAVY if o == 0 else (ORANGE if s else GREY)
                      for o, s in zip(offs, si)],
               edgecolor="white", linewidth=0.35, zorder=3)
        xs, ys = [], []
        for o, t in zip(offs, sub["threshold"].to_numpy(float)):
            if t <= 0:
                continue
            xs += [o - 0.5, o + 0.5]
            ys += [t, t]
        ax.plot(xs, ys, color=GREY_D, lw=0.8, ls=(0, (3, 1.8)), zorder=4)
        ax.set_yscale("log")
        ax.set_ylim(0.55, max(v.max() * 3.2, 80))
        row = idx.loc[(ex, tx, T)]
        lag = str(row["trigger_lags"]).replace("|", " + ")
        ax.set_title(f"{ex} \u00b7 {tx}\nT = {T}  |  trigger {lag}", fontsize=7,
                     loc="left", color=INK, linespacing=1.25)
        ax.set_xticks(list(offs))
        ax.set_xticklabels(["T" if o == 0 else f"T\u2212{abs(o)}" for o in offs], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.text(1.0, 1.015, letter, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, fontweight="bold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Export volume (log)")
        ax.yaxis.set_minor_locator(mticker.LogLocator(subs=(2, 5)))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    for ax in axes[-1, :]:
        ax.set_xlabel("Years relative to T")
    fig.text(0.5, 0.012,
             "Blue: focal year; orange: years in which the 4\u00d7 criterion fires; "
             "grey: other years. Zeros are plotted at 1 on the log scale.\n"
             "Dashed step: 4\u00d7 the preceding five-year mean in each year \u2014 "
             "the level each bar has to clear.",
             ha="center", va="top", fontsize=7, color="#444444", linespacing=1.35)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.165, hspace=0.62, wspace=0.30)
    _save(fig, "figA4_case_timelines")


# --------------------------------------------------------------- figA5
def fig_country_coverage() -> None:
    conc = _table("table_concentration.csv")
    gt = _table("table_gt_sensitivity.csv")
    names = {"all_candidates": "Candidate\nuniverse",
             "three_year_only": "Three-year-only\ncases",
             "B3_positive": "All three-year\npositives"}
    vals = [float(conc[conc["stratum"] == k]["gt_share"].iloc[0]) * 100 for k in names]
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 2.85),
                             gridspec_kw={"width_ratios": [1, 1.30]})
    ax = axes[0]
    y = np.arange(len(names))[::-1]
    ax.set_xscale("log")
    ax.set_xlim(0.25, max(11.0, max(vals) * 1.8))
    ax.set_ylim(-0.62, 2.62)
    ax.xaxis.set_major_locator(mticker.FixedLocator([0.3, 0.5, 1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(axis="x", color="#E6E6E6", lw=0.5, zorder=0)
    for yy, v in zip(y, vals):
        ax.plot([0.25, v], [yy, yy], color=NAVY_LT, lw=1.6, solid_capstyle="butt", zorder=2)
        ax.plot([v], [yy], "o", ms=7.0, color=NAVY, mec="white", mew=0.8, zorder=4)
        ax.text(v * 1.18, yy, f"{v:.2f}%", ha="left", va="center", fontsize=7.5,
                color=NAVY, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(list(names.values()), fontsize=7)
    ax.set_xlabel("Guatemala share (%, log scale)")
    ax.set_title(f"\u00d7{vals[1] / vals[0]:.0f}\u2013{vals[2] / vals[0]:.0f} "
                 "enrichment in the blind spot",
                 fontsize=8.5, loc="left", x=0.055, color=INK)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_tag(ax, "A", dx=-0.24)

    ax = axes[1]
    rows = [("Full pool", int(gt[gt["selection"] == "all"]["n_defensible"].iloc[0])),
            ("Excluding Guatemala", int(gt[gt["selection"] == "exclude_GT"]["n_defensible"].iloc[0])),
            ("Excluding reporting-break\nexporters",
             int(gt[gt["selection"] == "exclude_reporting_break"]["n_defensible"].iloc[0]))]
    yy = [3, 2, 1]
    for y_, (lab, v) in zip(yy, rows):
        ax.plot([0, v], [y_, y_], color=NAVY_LT, lw=2.4, solid_capstyle="butt", zorder=3)
        ax.plot([v], [y_], "o", ms=6.5, color=NAVY, mec="white", mew=0.7, zorder=4)
        ax.text(v + 0.9, y_, f"{v}", ha="left", va="center", fontsize=8, color=NAVY,
                fontweight="bold")
    ax.set_yticks(yy)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_ylim(0.4, 3.6)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.2)
    ax.set_xlabel("Defensible cases in the pooled Top-50")
    ax.set_title("The audit survives both exclusions", fontsize=8.5, loc="left", color=INK)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_tag(ax, "B", dx=-0.42, dy=1.03)
    fig.tight_layout(w_pad=1.4)
    _save(fig, "figA5_country_coverage")


def build_all() -> list[str]:
    """Rebuild every figure used in the manuscript; return their names."""
    configure_stdout()
    C.ensure_dirs()
    style()
    fig_flagship()
    fig_composition()
    fig_counterfactual()
    fig_timelines()
    fig_country_coverage()
    return ["figA0_flagship", "figA2_divergence_structure",
            "figA3_counterfactual", "figA4_case_timelines",
            "figA5_country_coverage"]
