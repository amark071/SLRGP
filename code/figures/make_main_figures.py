#!/usr/bin/env python3
"""Generate publication-ready empirical main-text Figures 3 and 4.

The script reads only frozen confirmatory artifacts and exports vector PDF/SVG,
600-dpi TIFF/PNG, and a JSON provenance manifest. It is intentionally
deterministic: no inferential statistic is recomputed except the S1a ROC curve
and descriptive system means that are not stored in the frozen summaries.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"

PATHS = {
    "s1a_annotations": ROOT / "data/exp1_structural_recovery/confirmatory/s1a_annotations_blind_pass1_dedup.jsonl",
    "s1a_summary": ROOT / "data/exp1_structural_recovery/confirmatory/s1a_eval_summary_blind.json",
    "s2b": ROOT / "data/exp2_operator_learnability/location/s2b_diag.json",
    "s2d": ROOT / "data/exp2_operator_learnability/ranking/tight_s2d_eval.json",
    "s2d_specter": ROOT / "data/exp2_operator_learnability/ranking/tight_s2d_specter2.json",
    "s2e": ROOT / "data/exp2_operator_learnability/organization/confirmatory/s2e_learning_report.json",
    "s4a": ROOT / "data/exp3_interface_substitution/analysis/S4A_BLIND_STATS.json",
    "s4b": ROOT / "data/exp3_interface_substitution/s4b_qwen4_20260713_1745/S4B_ROBUSTNESS_STATS.json",
    "s5": ROOT / "data/exp4_structural_recursion/s5a_structural_quant_20260714/S5A_STRUCTURAL_STATS.json",
    "s3_audit": ROOT / "data/exp5_native_comparison/s3_native_confirmatory_20260716/audit/deterministic_audit.json",
    "s3_cost": ROOT / "data/exp5_native_comparison/s3_native_confirmatory_20260716/analysis/cost_efficiency.json",
    "s3_pref": ROOT / "data/exp5_native_comparison/s3_native_confirmatory_20260716/analysis/preference_stats.json",
    "s3_pref_window": ROOT / "data/exp5_native_comparison/s3_native_confirmatory_20260716/common_window/analysis/preference_stats.json",
    "s3_claim": ROOT / "data/exp5_native_comparison/s3_native_confirmatory_20260716/analysis/claim_support_stats.json",
}


# Okabe-Ito inspired, colour-blind-safe palette.
# SLRGP / positive-result identity colour is a single navy, held constant across
# Figures 3 and 4 so reviewers recognise SLRGP data at a glance.
BLUE = "#1E5F8E"
SKY = "#56B4E9"
NULL_GREY = "#B4B8BD"
ORANGE = "#D55E00"
AMBER = "#E69F00"
GREEN = "#009E73"
PURPLE = "#7B61A8"
BLACK = "#202124"
DARK_GREY = "#5F6368"
MID_GREY = "#9AA0A6"
LIGHT_GREY = "#E8EAED"
PALE_GREY = "#F5F6F7"
WHITE = "#FFFFFF"

SYSTEMS = ["slrgp", "autosurvey", "surveyforge", "surveygen"]
SYSTEM_LABEL = {
    "slrgp": "SLRGP",
    "autosurvey": "AutoSurvey",
    "surveyforge": "SURVEYFORGE",
    "surveygen": "SurveyGen",
}
SYSTEM_COLOR = {
    "slrgp": BLUE,
    "autosurvey": ORANGE,
    "surveyforge": AMBER,
    "surveygen": GREEN,
}
TOPIC_LABEL = {
    "mature_neural_machine_translation": "NMT",
    "mature_deep_reinforcement_learning": "DRL",
    "fast_text_to_image_diffusion": "T2I",
    "fast_vision_language_pretraining": "VLP",
    "method_self_supervised_learning": "SSL",
    "method_continual_learning": "CL",
    "cross_medical_image_analysis": "MIA",
    "cross_ml_drug_discovery": "DD",
}


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 8.1,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
        }
    )


def clean_axis(ax, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(DARK_GREY)
    ax.spines["bottom"].set_color(DARK_GREY)
    ax.tick_params(colors=BLACK, pad=2)
    if grid:
        ax.grid(axis=grid, color=LIGHT_GREY, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
        color=BLACK,
    )


def panel_title(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold", pad=5, color=BLACK)


def export_figure(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in [
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 600}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ]:
        fig.savefig(
            OUT / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.04,
            **kwargs,
        )


def get_overall(items, contrast: str, eligibility: str | None = None):
    for row in items:
        if (
            row.get("contrast") == contrast
            and row.get("endpoint") == "overall"
            and (eligibility is None or row.get("eligibility") == eligibility)
        ):
            return row
    raise KeyError((contrast, eligibility))


def rate_ci_from_margin(pooled: dict) -> tuple[float, float, float]:
    point = pooled["mean_preference_rate"]
    lo, hi = pooled["paired_bootstrap_95ci_margin"]
    return point, lo + 0.5, hi + 0.5


def make_figure3(data: dict) -> dict:
    fig = plt.figure(figsize=(7.15, 7.0))
    outer = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.985,
        bottom=0.07,
        top=0.965,
        wspace=0.30,
        hspace=0.42,
    )

    # a — paired score distributions + ROC.
    gs_a = outer[0, 0].subgridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.36)
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[0, 1])
    panel_label(ax_a1, "a", x=-0.22)
    panel_title(ax_a1, "Expert organization is detectably structured")

    rows = data["s1a_annotations"]
    authentic = {}
    synthetic = {}
    for row in rows:
        pair_id = row["pair_id"]
        if row["condition"] == "authentic":
            authentic[pair_id] = float(row["expressibility_score"])
        elif row.get("tier") == "tier2" and row["condition"] == "synthetic":
            synthetic[pair_id] = float(row["expressibility_score"])
    common = sorted(authentic.keys() & synthetic.keys())
    auth = np.asarray([authentic[k] for k in common])
    hard = np.asarray([synthetic[k] for k in common])
    diff = auth - hard

    parts = ax_a1.violinplot(
        [hard, auth],
        positions=[0, 1],
        widths=0.72,
        showextrema=False,
        showmeans=False,
        showmedians=False,
    )
    for body, color in zip(parts["bodies"], [MID_GREY, BLUE]):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.72)
    rng = np.random.default_rng(20260718)
    for i, vals in enumerate([hard, auth]):
        sample = rng.choice(vals, size=min(360, len(vals)), replace=False)
        jitter = rng.normal(0, 0.065, len(sample))
        ax_a1.scatter(
            np.full_like(sample, i) + jitter,
            sample,
            s=3.5,
            c=[DARK_GREY if i == 0 else BLUE],
            alpha=0.18,
            linewidth=0,
            rasterized=True,
            zorder=2,
        )
        mean = float(np.mean(vals))
        ax_a1.plot([i - 0.22, i + 0.22], [mean, mean], color=BLACK, lw=1.2, zorder=4)
    ax_a1.set_xticks([0, 1], ["Hard\ncounterfactual", "Authentic"])
    ax_a1.set_ylabel("Expressibility score")
    ax_a1.set_ylim(-0.03, 1.05)
    clean_axis(ax_a1)
    tier2 = data["s1a_summary"]["annotation_evaluation"][0]["tiers"]["tier2"]
    ax_a1.text(
        0.03,
        0.97,
        f"$n$ = {len(common):,} pairs\n$\\Delta$ = {tier2['mean_expressibility_diff_auth_minus_synth']:.3f}\n"
        f"95% CI [{tier2['diff_bootstrap_ci'][0]:.3f}, {tier2['diff_bootstrap_ci'][1]:.3f}]",
        transform=ax_a1.transAxes,
        va="top",
        ha="left",
        fontsize=6.2,
        color=BLACK,
    )

    y_true = np.r_[np.zeros_like(hard), np.ones_like(auth)]
    scores = np.r_[hard, auth]
    fpr, tpr, _ = roc_curve(y_true, scores)
    ax_a2.plot(fpr, tpr, color=BLUE, lw=1.6)
    ax_a2.plot([0, 1], [0, 1], color=MID_GREY, lw=0.8, ls="--")
    ax_a2.fill_between(fpr, tpr, fpr, color=SKY, alpha=0.13)
    ax_a2.set_xlim(0, 1)
    ax_a2.set_ylim(0, 1)
    ax_a2.set_aspect("equal", adjustable="box")
    ax_a2.set_xlabel("False-positive rate")
    ax_a2.set_ylabel("True-positive rate")
    ax_a2.set_title(f"ROC-AUC = {tier2['roc_auc_auth_vs_synth']:.3f}", fontsize=7.2, pad=4)
    clean_axis(ax_a2, grid=None)

    # b — three compact learnability views.
    gs_b = outer[0, 1].subgridspec(1, 3, width_ratios=[1.25, 1.0, 1.2], wspace=0.58)
    ax_b1 = fig.add_subplot(gs_b[0, 0])
    ax_b2 = fig.add_subplot(gs_b[0, 1])
    ax_b3 = fig.add_subplot(gs_b[0, 2])
    panel_label(ax_b1, "b", x=-0.28)
    panel_title(ax_b1, "Frontend operators are selectively learnable")

    r_eval = data["s2d"]
    r_spec = data["s2d_specter"]
    effect_rows = [
        ("BM25", r_eval["paired_stats"]["learned_vs_bm25_only_ndcg10"]["ci"]),
        ("Dense", r_eval["paired_stats"]["learned_vs_dense_only_ndcg10"]["ci"]),
        ("RRF", r_eval["paired_stats"]["learned_vs_rrf_ndcg10"]["ci"]),
        ("SPECTER2", r_spec["learned_vs_specter2_ndcg10"]["ci"]),
        ("Linear", r_eval["paired_stats"]["learned_vs_linear_pairwise_ndcg10"]["ci"]),
    ]
    ypos = np.arange(len(effect_rows))[::-1]
    for y, (label, ci) in zip(ypos, effect_rows):
        point, lo, hi = ci["point_estimate"], ci["ci_low"], ci["ci_high"]
        ax_b1.errorbar(
            point,
            y,
            xerr=[[point - lo], [hi - point]],
            fmt="o",
            ms=4.0,
            color=BLUE,
            ecolor=BLUE,
            elinewidth=1.0,
            capsize=2,
            zorder=3,
        )
    ax_b1.axvline(0, color=DARK_GREY, lw=0.8)
    ax_b1.set_yticks(ypos, [r[0] for r in effect_rows])
    ax_b1.set_xlabel("$\\Delta$nDCG@10\nlearned $R$ − comparator")
    ax_b1.set_xlim(-0.01, 0.245)
    clean_axis(ax_b1, grid="x")
    ax_b1.text(0.02, 0.96, "Ranking $R$", transform=ax_b1.transAxes, fontweight="bold", fontsize=6.8, va="top")

    o = data["s2e"]["axes"]
    labels = ["Facet", "Relation"]
    base = [o["facet"]["metrics"]["depth_majority"]["accuracy"], o["relation"]["metrics"]["depth_majority"]["accuracy"]]
    learned = [o["facet"]["metrics"]["tfidf_logreg"]["accuracy"], o["relation"]["metrics"]["tfidf_logreg"]["accuracy"]]
    x = np.arange(2)
    width = 0.34
    ax_b2.bar(x - width / 2, base, width, color=LIGHT_GREY, edgecolor=DARK_GREY, linewidth=0.6, label="Structure-only")
    ax_b2.bar(x + width / 2, learned, width, color=GREEN, edgecolor=GREEN, linewidth=0.6, label="Learned")
    for i, axis_name in enumerate(["facet", "relation"]):
        ci = o[axis_name]["tfidf_logreg_minus_best_baseline_accuracy_ci"]
        ax_b2.text(
            i,
            learned[i] + 0.025,
            f"+{ci['mean_diff']:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.0,
            color=BLACK,
        )
    ax_b2.set_xticks(x, labels, rotation=25, ha="right")
    ax_b2.set_ylabel("Held-out accuracy")
    ax_b2.set_ylim(0, 0.64)
    ax_b2.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.23), ncol=1, handlelength=1.2)
    clean_axis(ax_b2)
    ax_b2.text(0.02, 0.96, "Organization $O$", transform=ax_b2.transAxes, fontweight="bold", fontsize=6.8, va="top")

    loc = data["s2b"]["by_split"]["test"]
    depths = np.asarray(data["s2b"]["depths"])
    for name, color, ls in [
        ("BM25", DARK_GREY, ":"),
        ("Dense", PURPLE, "--"),
        ("RRF / union", BLUE, "-"),
    ]:
        key = {"BM25": "bm25", "Dense": "dense", "RRF / union": "rrf"}[name]
        vals = [loc["baseline_micro_curves"][key][str(d)] for d in depths]
        ax_b3.plot(depths, vals, color=color, lw=1.25, ls=ls, marker="o", ms=2.5, label=name)
    ax_b3.axvline(300, color=MID_GREY, lw=0.7, ls="--")
    ax_b3.axhspan(loc["retrieval_ceiling_micro"], 1.0, color=ORANGE, alpha=0.07, zorder=0)
    ax_b3.text(
        0.98,
        0.93,
        f"~{loc['gap_retrieval_micro']:.0%} unreachable\nat depth 3,000",
        transform=ax_b3.transAxes,
        ha="right",
        va="top",
        fontsize=6.1,
        color=ORANGE,
    )
    ax_b3.set_xscale("log")
    ax_b3.set_xticks([100, 300, 1000, 3000], ["100", "300", "1k", "3k"])
    ax_b3.set_ylim(0, 1.0)
    ax_b3.set_xlabel("Retrieval depth")
    ax_b3.set_ylabel("True-citation coverage")
    ax_b3.legend(frameon=False, loc="lower right", fontsize=5.7, handlelength=1.5)
    clean_axis(ax_b3)
    ax_b3.text(0.02, 0.96, "Location $L$", transform=ax_b3.transAxes, fontweight="bold", fontsize=6.8, va="top")

    # c — S4 matched substitution forest.
    ax_c = fig.add_subplot(outer[1, 0])
    panel_label(ax_c, "c", x=-0.14)
    panel_title(ax_c, "Matched substitutions isolate operator effects")
    s4a = data["s4a"]["contrasts"]
    s4b = data["s4b"]["quality"]
    c_rows = [
        ("Semantic $O$ vs rank slabs", get_overall(s4a, "O semantic vs rank slab"), "Claude Sonnet 4.6", BLUE, "o"),
        ("Semantic $O$ vs rank slabs", get_overall(s4b, "O semantic vs rank slab", "primary"), "Qwen3-32B", SKY, "s"),
        ("Guarded $V$ vs unguarded", get_overall(s4a, "Guarded V vs unguarded stress"), "Claude Sonnet 4.6", BLUE, "o"),
        ("Guarded $V$ vs unguarded", get_overall(s4b, "Guarded V vs unguarded stress", "primary"), "Qwen3-32B", SKY, "s"),
        ("Short recursion vs flat", get_overall(s4a, "Recursion vs flat"), "Claude Sonnet 4.6", NULL_GREY, "o"),
    ]
    y_positions = [4.2, 3.55, 2.35, 1.7, 0.5]
    for y, (label, row, backbone, color, marker) in zip(y_positions, c_rows):
        point = row["mean_diff"]
        lo, hi = row["ci95"]
        ax_c.errorbar(
            point,
            y,
            xerr=[[point - lo], [hi - point]],
            fmt=marker,
            ms=4.4,
            mfc=WHITE if backbone == "Qwen3-32B" else color,
            mec=color,
            mew=1.0,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.2,
            zorder=4,
        )
        ax_c.text(1.98, y, f"$n$={row['n']}", ha="right", va="center", fontsize=6.0, color=DARK_GREY)
    ax_c.axvline(0, color=BLACK, lw=0.8)
    ax_c.axhline(3.0, color=LIGHT_GREY, lw=0.7)
    ax_c.axhline(1.1, color=LIGHT_GREY, lw=0.7)
    ax_c.set_yticks([3.88, 2.03, 0.5], ["Semantic $O$", "Guarded $V$", "Short recursion"])
    ax_c.set_ylim(-0.1, 4.75)
    ax_c.set_xlim(-0.35, 2.05)
    ax_c.set_xlabel("Paired difference in panel-overall quality (95% CI)")
    clean_axis(ax_c, grid="x")
    ax_c.legend(
        handles=[
            Line2D([0], [0], marker="o", color=BLUE, lw=0, markerfacecolor=BLUE, label="Claude Sonnet 4.6"),
            Line2D([0], [0], marker="s", color=SKY, lw=0, markerfacecolor=WHITE, label="Qwen3-32B (directional)"),
        ],
        loc="upper left",
        frameon=False,
        ncol=1,
        handletextpad=0.4,
    )

    # d — S5 structural recursion forest with topic-level points.
    ax_d = fig.add_subplot(outer[1, 1])
    panel_label(ax_d, "d", x=-0.14)
    panel_title(ax_d, "Structural recursion improves long-form synthesis")
    d_rows = [
        ("Naive chunking", get_overall(data["s5"]["contrasts"], "structural vs naive chunking")),
        ("Fixed outline", get_overall(data["s5"]["contrasts"], "structural vs fixed outline")),
        ("Flat generation", get_overall(data["s5"]["contrasts"], "structural vs flat")),
    ]
    y = np.arange(3)[::-1]
    for yi, (label, row) in zip(y, d_rows):
        topic_diff = np.asarray([t["diff"] for t in row["topics"]])
        jitter = np.linspace(-0.09, 0.09, len(topic_diff))
        ax_d.scatter(topic_diff, yi + jitter, s=9, color=SKY, alpha=0.72, edgecolor=WHITE, linewidth=0.35, zorder=2)
        point = row["mean_diff"]
        lo, hi = row["ci95"]
        ax_d.errorbar(
            point,
            yi,
            xerr=[[point - lo], [hi - point]],
            fmt="D",
            ms=4.2,
            color=BLUE,
            mfc=BLUE,
            ecolor=BLUE,
            elinewidth=1.3,
            capsize=2.4,
            zorder=4,
        )
        ax_d.text(2.36, yi, "6/6 wins", ha="right", va="center", fontsize=6.0, color=DARK_GREY)
    ax_d.axvline(0, color=BLACK, lw=0.8)
    ax_d.set_yticks(y, [r[0] for r in d_rows])
    ax_d.set_xlim(-0.15, 2.42)
    ax_d.set_xlabel("Panel-overall quality difference\n(LOTCF-LR − control)")
    clean_axis(ax_d, grid="x")
    ax_d.legend(
        handles=[
            Line2D([0], [0], marker="D", color=BLUE, lw=0, label="Mean"),
            Line2D([0], [0], marker="o", color=SKY, lw=0, label="Topic"),
        ],
        frameon=False,
        loc="lower right",
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.7,
    )

    export_figure(fig, "Figure3_internal_evidence")
    plt.close(fig)
    return {
        "s1a_n_pairs": len(common),
        "s1a_auc_recomputed": float(np.trapezoid(tpr, fpr)),
        "s1a_mean_paired_difference_recomputed": float(np.mean(diff)),
        "figure": "Figure3_internal_evidence",
    }


def make_figure4(data: dict) -> dict:
    fig = plt.figure(figsize=(7.15, 6.45))
    outer = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.985,
        bottom=0.085,
        top=0.965,
        wspace=0.32,
        hspace=0.40,
    )
    claim = data["s3_claim"]
    audit_rows = data["s3_audit"]["rows"]
    cost = data["s3_cost"]

    score_map = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}
    support_values = defaultdict(list)
    support_topic = defaultdict(list)
    for row in claim["finalized_claims"]:
        if row["label"] in score_map:
            value = score_map[row["label"]]
            support_values[row["system"]].append(value)
            support_topic[(row["system"], row["topic_id"])].append(value)
    support_mean = {system: float(np.mean(values)) for system, values in support_values.items()}

    compliance = {}
    for system in SYSTEMS:
        subset = [r for r in audit_rows if r["system"] == system]
        compliance[system] = (
            sum(r["length_compliant"] for r in subset),
            sum(r["reference_compliant"] for r in subset),
        )

    # a — cost-quality-compliance landscape.
    ax_a = fig.add_subplot(outer[0, 0])
    panel_label(ax_a, "a", x=-0.16)
    panel_title(ax_a, "Cost–quality–compliance profile")
    for system in SYSTEMS:
        x = cost["systems"][system]["usd_per_output"]
        y = support_mean[system]
        length_ok, refs_ok = compliance[system]
        marker = "o" if (length_ok == 8 and refs_ok == 8) else ("D" if refs_ok == 8 else "s")
        face = SYSTEM_COLOR[system] if length_ok == 8 else WHITE
        ax_a.scatter(
            x,
            y,
            s=74,
            marker=marker,
            facecolor=face,
            edgecolor=SYSTEM_COLOR[system],
            linewidth=1.5,
            zorder=4,
        )
        offsets = {
            "slrgp": (6, 7),
            "autosurvey": (-6, -13),
            "surveyforge": (-6, 8),
            "surveygen": (6, -12),
        }
        dx, dy = offsets[system]
        ax_a.annotate(
            f"{SYSTEM_LABEL[system]}\n\\${x:,.2f}",
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="center",
            fontsize=6.3,
            fontweight="bold" if system == "slrgp" else "normal",
            color=SYSTEM_COLOR[system],
            linespacing=1.15,
        )
    ax_a.set_xscale("log")
    ax_a.set_xlim(0.45, 16.5)
    ax_a.set_ylim(0.46, 0.90)
    ax_a.set_xlabel("Generation cost per output (US$, log scale)")
    ax_a.set_ylabel("Mean claim–source support")
    clean_axis(ax_a)
    ax_a.legend(
        handles=[
            Line2D([0], [0], marker="o", color=BLACK, markerfacecolor=BLACK, lw=0, label="Both gates: 8/8"),
            Line2D([0], [0], marker="D", color=BLACK, markerfacecolor=WHITE, lw=0, label="Reference gate only"),
            Line2D([0], [0], marker="s", color=BLACK, markerfacecolor=WHITE, lw=0, label="Neither gate"),
        ],
        frameon=False,
        loc="lower left",
        handletextpad=0.4,
    )

    # b — per-topic word counts and target window.
    ax_b = fig.add_subplot(outer[0, 1])
    panel_label(ax_b, "b", x=-0.15)
    panel_title(ax_b, "Native output length by topic")
    topic_order = list(TOPIC_LABEL)
    x = np.arange(len(topic_order))
    ax_b.axhspan(3200, 4800, color=BLUE, alpha=0.10, label="Registered target")
    for system in SYSTEMS:
        row_map = {r["topic_id"]: r["word_count"] for r in audit_rows if r["system"] == system}
        vals = [row_map[t] for t in topic_order]
        ax_b.plot(
            x,
            vals,
            color=SYSTEM_COLOR[system],
            lw=1.15,
            marker="o",
            ms=3.1,
            label=SYSTEM_LABEL[system],
        )
    ax_b.set_yscale("log")
    ax_b.set_xticks(x, [TOPIC_LABEL[t] for t in topic_order])
    ax_b.set_yticks([3200, 4800, 10000, 30000, 80000], ["3.2k", "4.8k", "10k", "30k", "80k"])
    ax_b.set_ylabel("Output length (words, log scale)")
    ax_b.set_xlabel("Frozen topic (abbreviations in caption)")
    clean_axis(ax_b)
    ax_b.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    # c — full-native versus common-window preference.
    ax_c = fig.add_subplot(outer[1, 0])
    panel_label(ax_c, "c", x=-0.16)
    panel_title(ax_c, "Blind preference depends on reading budget")
    baseline_order = ["autosurvey", "surveyforge", "surveygen"]
    y_centers = np.arange(3)[::-1] * 1.35
    for y0, baseline in zip(y_centers, baseline_order):
        for offset, source, color, marker, label in [
            (0.20, data["s3_pref"], DARK_GREY, "o", "Full native"),
            (-0.20, data["s3_pref_window"], BLUE, "s", "4,000-word window"),
        ]:
            pooled = source["comparisons"][baseline]["overall_preference"]["pooled"]
            point, lo, hi = rate_ci_from_margin(pooled)
            topic_scores = np.asarray(list(pooled["topic_scores"].values()))
            jitter = np.linspace(-0.055, 0.055, len(topic_scores))
            ax_c.scatter(
                topic_scores,
                np.full_like(topic_scores, y0 + offset) + jitter,
                s=6,
                color=color,
                alpha=0.26,
                linewidth=0,
                zorder=1,
            )
            ax_c.errorbar(
                point,
                y0 + offset,
                xerr=[[point - lo], [hi - point]],
                fmt=marker,
                ms=4.2,
                mfc=color if marker == "s" else WHITE,
                mec=color,
                mew=1.0,
                color=color,
                ecolor=color,
                elinewidth=1.15,
                capsize=2.3,
                zorder=4,
            )
    ax_c.axvline(0.5, color=BLACK, lw=0.8, ls="--")
    ax_c.set_yticks(y_centers, [SYSTEM_LABEL[b] for b in baseline_order])
    ax_c.set_xlim(-0.02, 1.02)
    ax_c.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_c.set_xlabel("Probability panel prefers SLRGP (95% paired bootstrap CI)")
    clean_axis(ax_c, grid="x")
    ax_c.legend(
        handles=[
            Line2D([0], [0], marker="o", color=DARK_GREY, markerfacecolor=WHITE, lw=0, label="Full native"),
            Line2D([0], [0], marker="s", color=BLUE, markerfacecolor=BLUE, lw=0, label="4,000-word window"),
        ],
        frameon=False,
        loc="center left",
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.3,
    )

    # d — paired claim-support differences.
    ax_d = fig.add_subplot(outer[1, 1])
    panel_label(ax_d, "d", x=-0.15)
    panel_title(ax_d, "SLRGP improves sampled claim–source support")
    y = np.arange(3)[::-1]
    for yi, baseline in zip(y, baseline_order):
        row = claim["comparisons"][baseline]
        point = row["mean_topic_paired_difference"]
        lo, hi = row["hierarchical_bootstrap_95ci"]
        topic_diffs = np.asarray(list(row["topic_differences"].values()))
        jitter = np.linspace(-0.085, 0.085, len(topic_diffs))
        ax_d.scatter(topic_diffs, yi + jitter, s=11, color=SKY, alpha=0.78, edgecolor=WHITE, linewidth=0.35, zorder=2)
        significant = row["bh_fdr_q_across_six_primary_tests"] < 0.05
        ax_d.errorbar(
            point,
            yi,
            xerr=[[point - lo], [hi - point]],
            fmt="D",
            ms=4.4,
            mfc=BLUE if significant else WHITE,
            mec=BLUE,
            mew=1.0,
            color=BLUE,
            ecolor=BLUE,
            elinewidth=1.25,
            capsize=2.4,
            zorder=4,
        )
        ax_d.text(
            0.46,
            yi,
            f"$q$={row['bh_fdr_q_across_six_primary_tests']:.3f}",
            ha="right",
            va="center",
            fontsize=6.0,
            color=DARK_GREY,
        )
    ax_d.axvline(0, color=BLACK, lw=0.8)
    ax_d.set_yticks(y, [SYSTEM_LABEL[b] for b in baseline_order])
    ax_d.set_xlim(-0.18, 0.48)
    ax_d.set_xlabel("Topic-paired support difference\nSLRGP − comparator (95% hierarchical-bootstrap CI)")
    clean_axis(ax_d, grid="x")
    ax_d.legend(
        handles=[
            Line2D([0], [0], marker="D", color=BLUE, markerfacecolor=BLUE, lw=0, label="$q<0.05$"),
            Line2D([0], [0], marker="D", color=BLUE, markerfacecolor=WHITE, lw=0, label="not significant"),
            Line2D([0], [0], marker="o", color=SKY, lw=0, label="Topic"),
        ],
        frameon=False,
        loc="upper left",
        ncol=1,
        handletextpad=0.35,
    )

    export_figure(fig, "Figure4_native_comparison")
    plt.close(fig)
    return {
        "support_means_recomputed": support_mean,
        "compliance_counts": {k: {"length": v[0], "references": v[1]} for k, v in compliance.items()},
        "figure": "Figure4_native_comparison",
    }


def main() -> None:
    configure_matplotlib()
    data = {name: load_json(path) for name, path in PATHS.items() if name != "s1a_annotations"}
    data["s1a_annotations"] = load_jsonl(PATHS["s1a_annotations"])
    audit = {
        "sources": {name: str(path.relative_to(ROOT)) for name, path in PATHS.items()},
        "figure3": make_figure3(data),
        "figure4": make_figure4(data),
        "notes": [
            "S1a ROC is recomputed from opaque-ID blind tier-2 scores; summary statistics are read from the frozen evaluation.",
            "Preference confidence intervals are stored as margins from tie and converted to probability intervals by adding 0.5.",
            "Claim-support system means exclude inaccessible/insufficient source packets, matching the primary manuscript estimand.",
            "All other intervals and p/q values are read directly from frozen confirmatory summaries.",
        ],
    }
    with (OUT / "main_figures_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
