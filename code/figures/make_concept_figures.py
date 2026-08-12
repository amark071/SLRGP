#!/usr/bin/env python3
"""Generate publication-ready conceptual Figures 1 and 2.

Deterministic vector schematics summarizing the operator architecture (Fig. 1)
and the experimental design (Fig. 2). No new empirical quantities are introduced.
The visual language is deliberately restrained for a Nature-family aesthetic:
a single navy identity colour, two muted accents used only where they carry
meaning, hairline strokes, pale fills and generous whitespace. Panel titles are
short noun phrases; all interpretation lives in the captions.

Exports: vector PDF/SVG plus 600-dpi PNG/TIFF and a JSON provenance manifest.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"

# --- Restrained, colour-blind-safe palette -------------------------------------
INK = "#1F2329"        # primary text
SUB = "#5C646C"        # secondary text
HAIR = "#C2C8CE"       # neutral hairline strokes
FAINT = "#F4F6F8"      # neutral pale fill

NAVY = "#28607F"       # primary accent = SLRGP identity
NAVY_FILL = "#ECF2F6"
OCHRE = "#A9772C"      # structure / control (used only in Fig. 1)
OCHRE_FILL = "#F6F0E5"
TEAL = "#2E7161"       # grounded synthesis (used only in Fig. 1)
TEAL_FILL = "#E9F1EE"
WHITE = "#FFFFFF"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
        }
    )


# --- Drawing primitives --------------------------------------------------------
def new_axis(fig, gs) -> "plt.Axes":
    ax = fig.add_subplot(gs)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def panel_header(ax, letter: str, title: str) -> None:
    ax.text(-0.045, 1.075, letter, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", ha="left", va="top", color=INK)
    ax.text(0.045, 1.068, title, transform=ax.transAxes, fontsize=7.4,
            fontweight="bold", ha="left", va="top", color=INK)


def rbox(ax, x, y, w, h, *, face=WHITE, edge=HAIR, lw=0.8, radius=0.02, z=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def labelled_box(ax, x, y, w, h, title, sub="", *, face=WHITE, edge=HAIR,
                 title_color=INK, title_size=6.6, sub_size=5.6, lw=0.8,
                 align="center"):
    rbox(ax, x, y, w, h, face=face, edge=edge, lw=lw)
    tx = x + w / 2 if align == "center" else x + 0.02
    ha = "center" if align == "center" else "left"
    ty = y + h * (0.63 if sub else 0.5)
    ax.text(tx, ty, title, ha=ha, va="center", fontsize=title_size,
            fontweight="bold", color=title_color, zorder=4)
    if sub:
        ax.text(tx, y + h * 0.29, sub, ha=ha, va="center", fontsize=sub_size,
                color=SUB, linespacing=1.15, zorder=4)


def arrow(ax, start, end, *, color=SUB, lw=0.85, rad=0.0, z=3):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=7, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
        zorder=z,
    ))


def connector(ax, start, end, *, color=HAIR, lw=0.8, z=1):
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=lw,
            solid_capstyle="round", zorder=z)


def phase_label(ax, x, y, text, color):
    ax.text(x, y, text.upper(), fontsize=5.7, fontweight="bold", color=color,
            ha="left", va="center", zorder=5)


def export_figure(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kw in [
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 600}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ]:
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight",
                    pad_inches=0.05, **kw)


# --- Figure 1: operator architecture -------------------------------------------
def operator_chip(ax, x, y, w, h, symbol, name, edge):
    rbox(ax, x, y, w, h, face=WHITE, edge=edge, lw=0.9, radius=0.014)
    ax.text(x + w / 2, y + h * 0.64, symbol, ha="center", va="center",
            fontsize=10.0, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.28, name, ha="center", va="center",
            fontsize=5.6, color=SUB)


def make_figure1() -> None:
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[0.90, 1.10], width_ratios=[1.0, 1.0],
        hspace=0.42, wspace=0.16, left=0.045, right=0.98, top=0.90, bottom=0.05,
    )
    ax_a = new_axis(fig, gs[0, :])
    ax_b = new_axis(fig, gs[1, 0])
    ax_c = new_axis(fig, gs[1, 1])

    # a — operator pipeline over the shared state.
    panel_header(ax_a, "a", "Typed operators over a shared review state")
    ax_a.text(0.0, 0.92, r"$s=\langle q,\ D,\ \Gamma,\ \kappa,\ x\rangle$",
              fontsize=8.0, color=INK, va="center")
    ax_a.text(0.30, 0.92,
              "intention · evidence · organization tree · validation · synthesis",
              fontsize=5.7, color=SUB, va="center")

    bands = [
        (0.010, 0.405, "Intention & evidence", NAVY, NAVY_FILL),
        (0.430, 0.205, "Structure & control", OCHRE, OCHRE_FILL),
        (0.655, 0.335, "Grounded synthesis", TEAL, TEAL_FILL),
    ]
    band_y, band_h = 0.13, 0.50
    for x, w, label, edge, fill in bands:
        rbox(ax_a, x, band_y, w, band_h, face=fill, edge=edge, lw=0.7,
             radius=0.02, z=0)
        phase_label(ax_a, x + 0.015, band_y + band_h - 0.055, label, edge)

    ops = [
        (0.030, "E", "Expand", NAVY), (0.126, "L", "Locate", NAVY),
        (0.222, "F", "Filter", NAVY), (0.318, "R", "Rank", NAVY),
        (0.452, "O", "Organize", OCHRE), (0.550, "V", "Validate", OCHRE),
        (0.678, "P", "Prepare", TEAL), (0.776, "W", "Write", TEAL),
        (0.874, "C", "Reconcile", TEAL),
    ]
    op_w, op_h, op_y = 0.078, 0.245, 0.20
    for x, symbol, name, edge in ops:
        operator_chip(ax_a, x, op_y, op_w, op_h, symbol, name, edge)
    for (x1, *_), (x2, *_) in zip(ops[:-1], ops[1:]):
        arrow(ax_a, (x1 + op_w + 0.007, op_y + op_h / 2),
              (x2 - 0.007, op_y + op_h / 2), color=HAIR, lw=0.8)

    # Guarded backtracking arc, routed above the band with clear label space.
    arrow(ax_a, (0.589, band_y + band_h), (0.069, band_y + band_h),
          color=OCHRE, lw=0.95, rad=0.32)
    ax_a.text(0.33, 0.80, "guarded backtracking", fontsize=5.7,
              fontweight="bold", color=OCHRE, ha="center", va="center")

    # b — organization output.
    panel_header(ax_b, "b", "Organization output: a labelled ordered tree")
    labelled_box(ax_b, 0.05, 0.82, 0.90, 0.12,
                 r"$O(q,D)\ \rightarrow\ \Gamma$",
                 "partition facet × ordering relation",
                 face=OCHRE_FILL, edge=OCHRE, title_color=OCHRE,
                 title_size=7.6, sub_size=5.5)
    nodes = {
        "root": (0.34, 0.60, 0.32, 0.10, "Review scope", NAVY_FILL, NAVY),
        "a": (0.02, 0.375, 0.29, 0.10, "Foundations", WHITE, OCHRE),
        "b": (0.355, 0.375, 0.29, 0.10, "Methods", WHITE, OCHRE),
        "c": (0.69, 0.375, 0.29, 0.10, "Applications", WHITE, OCHRE),
        "a1": (0.005, 0.145, 0.185, 0.095, "Theory A", WHITE, HAIR),
        "a2": (0.205, 0.145, 0.185, 0.095, "Theory B", WHITE, HAIR),
        "b1": (0.405, 0.145, 0.19, 0.095, "Method class", WHITE, HAIR),
        "c1": (0.665, 0.145, 0.15, 0.095, "Domain 1", WHITE, HAIR),
        "c2": (0.83, 0.145, 0.15, 0.095, "Domain 2", WHITE, HAIR),
    }
    for parent, children in [("root", ["a", "b", "c"]), ("a", ["a1", "a2"]),
                             ("b", ["b1"]), ("c", ["c1", "c2"])]:
        px, py, pw, _, *_ = nodes[parent]
        for ch in children:
            x, y, w, h, *_ = nodes[ch]
            connector(ax_b, (px + pw / 2, py), (x + w / 2, y + h), lw=0.8)
    for key, (x, y, w, h, title, face, edge) in nodes.items():
        tcol = edge if edge in {NAVY, OCHRE} else INK
        labelled_box(ax_b, x, y, w, h, title, face=face, edge=edge,
                     title_color=tcol, title_size=5.9, lw=0.8)
    ax_b.text(0.5, 0.315, "ordered branches, section-level citations",
              fontsize=5.4, color=SUB, ha="center", va="center")
    for x in [0.045, 0.085, 0.125, 0.245, 0.285, 0.475, 0.515, 0.71, 0.75,
              0.87, 0.91]:
        ax_b.add_patch(Circle((x, 0.075), 0.007, facecolor=NAVY_FILL,
                              edgecolor=NAVY, lw=0.5, zorder=3))

    # c — recursive descent.
    panel_header(ax_c, "c", "Recursive descent over the tree")
    labelled_box(ax_c, 0.02, 0.80, 0.27, 0.135, r"Internal node $i$",
                 r"local state $s_i$", face=NAVY_FILL, edge=NAVY,
                 title_color=NAVY, title_size=6.4, sub_size=5.4)
    labelled_box(ax_c, 0.365, 0.80, 0.27, 0.135, r"$O\rightarrow$ Descend",
                 "refine intention", face=OCHRE_FILL, edge=OCHRE,
                 title_color=OCHRE, title_size=6.4, sub_size=5.4)
    labelled_box(ax_c, 0.71, 0.80, 0.27, 0.135, "Child states",
                 r"$s_{i1},s_{i2},\ldots$", face=NAVY_FILL, edge=NAVY,
                 title_color=NAVY, title_size=6.4, sub_size=5.4)
    arrow(ax_c, (0.29, 0.8675), (0.365, 0.8675), color=OCHRE)
    arrow(ax_c, (0.635, 0.8675), (0.71, 0.8675), color=OCHRE)
    for idx, x in enumerate([0.745, 0.83, 0.915], start=1):
        labelled_box(ax_c, x - 0.033, 0.615, 0.066, 0.085, rf"$i{idx}$",
                     face=WHITE, edge=NAVY, title_color=NAVY, title_size=5.8)
        connector(ax_c, (0.845, 0.80), (x, 0.70), color=HAIR, lw=0.8)

    labelled_box(ax_c, 0.365, 0.485, 0.27, 0.135, r"$P\rightarrow W$",
                 "cards → bounded leaf", face=TEAL_FILL, edge=TEAL,
                 title_color=TEAL, title_size=6.4, sub_size=5.4)
    labelled_box(ax_c, 0.02, 0.485, 0.27, 0.135, r"$V$: guard",
                 "validate / backtrack", face=OCHRE_FILL, edge=OCHRE,
                 title_color=OCHRE, title_size=6.4, sub_size=5.4)
    arrow(ax_c, (0.71, 0.635), (0.635, 0.575), color=TEAL)
    arrow(ax_c, (0.29, 0.5525), (0.365, 0.5525), color=OCHRE)
    arrow(ax_c, (0.145, 0.62), (0.145, 0.80), color=OCHRE, rad=-0.35)

    labelled_box(ax_c, 0.365, 0.255, 0.27, 0.135, r"$C$: upward merge",
                 "citation closure", face=TEAL_FILL, edge=TEAL,
                 title_color=TEAL, title_size=6.4, sub_size=5.4)
    labelled_box(ax_c, 0.02, 0.255, 0.27, 0.135, "Parent synthesis",
                 r"$x_i$", face=TEAL_FILL, edge=TEAL, title_color=TEAL,
                 title_size=6.4, sub_size=5.4)
    arrow(ax_c, (0.5, 0.485), (0.5, 0.39), color=TEAL)
    arrow(ax_c, (0.365, 0.3225), (0.29, 0.3225), color=TEAL)

    ax_c.text(0.735, 0.15, "stop:", fontsize=5.6, fontweight="bold",
              color=SUB, ha="left")
    ax_c.text(0.735, 0.095, r"depth $\geq D_{\max}$ · papers $\leq \theta_{\rm leaf}$",
              fontsize=5.3, color=SUB, ha="left")
    ax_c.text(0.735, 0.045, r"words $\leq \theta_{\rm words}$",
              fontsize=5.3, color=SUB, ha="left")

    export_figure(fig, "Figure1_operator_architecture")
    plt.close(fig)


# --- Figure 2: experimental design ---------------------------------------------
def make_figure2() -> None:
    fig = plt.figure(figsize=(7.2, 5.5))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[0.96, 1.04], width_ratios=[1.0, 1.0],
        hspace=0.40, wspace=0.20, left=0.045, right=0.98, top=0.91, bottom=0.05,
    )
    ax_a = new_axis(fig, gs[0, 0])
    ax_b = new_axis(fig, gs[0, 1])
    ax_c = new_axis(fig, gs[1, 0])
    ax_d = new_axis(fig, gs[1, 1])

    # a — corpus construction (analytic population only; sampling frame in SI).
    panel_header(ax_a, "a", "Corpus construction")
    stages = [
        (0.02, "2,221", "analytic reviews"),
        (0.27, "LaTeX", "ordered section trees"),
        (0.52, "14,332", "organization nodes"),
        (0.77, "2,881", "held-out nodes"),
    ]
    for x, title, sub in stages:
        labelled_box(ax_a, x, 0.74, 0.21, 0.15, title, sub, face=NAVY_FILL,
                     edge=NAVY, title_color=NAVY, title_size=7.0, sub_size=5.3)
    for x1, x2 in [(0.23, 0.27), (0.48, 0.52), (0.73, 0.77)]:
        arrow(ax_a, (x1, 0.815), (x2 - 0.006, 0.815), color=HAIR, lw=0.8)

    labelled_box(ax_a, 0.14, 0.48, 0.30, 0.14, "Train / validation",
                 "operator learning only", face=FAINT, edge=HAIR,
                 title_color=SUB, title_size=6.0, sub_size=5.1)
    labelled_box(ax_a, 0.56, 0.48, 0.31, 0.14, "Held-out evaluation",
                 "authentic + matched controls", face=NAVY_FILL, edge=NAVY,
                 title_color=NAVY, title_size=6.0, sub_size=5.1)
    ax_a.text(0.5, 0.68, "review-level SHA-256 split", fontsize=5.3,
              color=SUB, ha="center")
    arrow(ax_a, (0.62, 0.735), (0.29, 0.625), color=HAIR, lw=0.75, rad=0.10)
    arrow(ax_a, (0.86, 0.735), (0.72, 0.625), color=NAVY, lw=0.8, rad=-0.06)

    ax_a.text(0.5, 0.34, "section hierarchy · citation assignment · source identifiers",
              fontsize=5.4, color=NAVY, ha="center", va="center")
    ax_a.text(0.5, 0.28, "every derived item stays linked to its review and node",
              fontsize=5.4, color=SUB, ha="center", va="center")

    # b — progressive experiment chain (single clean column).
    panel_header(ax_b, "b", "Progressive experiment chain")
    chain = [
        ("1", "Structure", "expert trees vs hard negatives", "representational contact"),
        ("2", "Learnability", "held-out typed interfaces", "selective expert signal"),
        ("3", "Contribution", "matched operator substitutions", "interface-local effects"),
        ("4", "Recursion", "structural vs length controls", "compositional advantage"),
        ("5", "Package", "native workflow comparison", "deployment profile"),
    ]
    ys = [0.80, 0.645, 0.49, 0.335, 0.18]
    badge_x = 0.075
    connector(ax_b, (badge_x, ys[-1]), (badge_x, ys[0]), color=HAIR, lw=1.0)
    for (num, title, desc, establishes), y in zip(chain, ys):
        ax_b.add_patch(Circle((badge_x, y), 0.033, facecolor=NAVY,
                              edgecolor="none", zorder=4))
        ax_b.text(badge_x, y, num, ha="center", va="center", fontsize=6.6,
                  fontweight="bold", color=WHITE, zorder=5)
        ax_b.text(0.15, y + 0.028, title, ha="left", va="center", fontsize=6.8,
                  fontweight="bold", color=INK)
        ax_b.text(0.15, y - 0.028, desc, ha="left", va="center", fontsize=5.5,
                  color=SUB)
        ax_b.text(0.985, y, establishes, ha="right", va="center", fontsize=5.4,
                  color=NAVY, style="italic")

    # c — matched controls.
    panel_header(ax_c, "c", "Matched controls")
    rows = [
        (0.80, "Structural signal", "authentic node", "hard synthetic",
         "matched: discipline · depth · child count"),
        (0.585, "Operator effect", r"$O$ / guarded $V$", "compatible substitute",
         "matched: inputs · backbone · budget"),
        (0.37, "Recursive effect", "structural recursion", "length controls",
         "matched: leaf evidence · topics"),
        (0.155, "System package", "SLRGP native", "external native flow",
         "matched: backbone · cutoff · scope"),
    ]
    for y, level, test, ctrl, matched in rows:
        ax_c.text(0.015, y + 0.045, level, fontsize=5.8, fontweight="bold",
                  color=INK, ha="left")
        labelled_box(ax_c, 0.30, y, 0.27, 0.09, test, face=NAVY_FILL,
                     edge=NAVY, title_color=NAVY, title_size=5.8)
        ax_c.text(0.595, y + 0.045, "vs", fontsize=5.6, color=SUB,
                  ha="center", va="center")
        labelled_box(ax_c, 0.62, y, 0.31, 0.09, ctrl, face=WHITE, edge=HAIR,
                     title_color=SUB, title_size=5.7)
        ax_c.text(0.30, y - 0.035, matched, fontsize=5.0, color=SUB,
                  ha="left", va="center", style="italic")

    # d — blinding, outcomes and inference.
    panel_header(ax_d, "d", "Blinding, outcomes and inference")
    top = [
        (0.02, "Separation of roles", "annotate ≠ generate ≠ judge"),
        (0.35, "Blind comparison", "opaque IDs · paired units"),
        (0.68, "Frozen protocols", "held-out · registered"),
    ]
    for x, title, sub in top:
        labelled_box(ax_d, x, 0.72, 0.30, 0.155, title, sub, face=NAVY_FILL,
                     edge=NAVY, title_color=NAVY, title_size=6.1, sub_size=5.1)

    ax_d.text(0.015, 0.63, "OUTCOME FAMILIES", fontsize=5.4, fontweight="bold",
              color=SUB, ha="left")
    families = [
        (0.02, "Structure", "expressibility"),
        (0.265, "Learning", "coverage · nDCG"),
        (0.51, "Quality", "coherence · citations"),
        (0.755, "Deployment", "gates · cost"),
    ]
    for x, title, sub in families:
        labelled_box(ax_d, x, 0.44, 0.225, 0.135, title, sub, face=WHITE,
                     edge=HAIR, title_color=INK, title_size=5.9, sub_size=5.0)

    ax_d.text(0.015, 0.355, "INFERENCE", fontsize=5.4, fontweight="bold",
              color=SUB, ha="left")
    inference = ["95% bootstrap CI", "exact / paired tests", "BH–FDR families",
                 "hierarchical bootstrap"]
    for x, title in zip([0.02, 0.265, 0.51, 0.755], inference):
        labelled_box(ax_d, x, 0.20, 0.225, 0.10, title, face=FAINT, edge=HAIR,
                     title_color=INK, title_size=5.1)
    ax_d.text(0.5, 0.085, "paired at the review, topic or claim level",
              fontsize=5.4, color=SUB, ha="center", va="center")

    export_figure(fig, "Figure2_experimental_design")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    make_figure1()
    make_figure2()
    provenance = {
        "figure1": {
            "file": "Figure1_operator_architecture",
            "content": "Typed operators, LOTCF-LR output, guarded validation and recursion.",
            "source": ["v2.tex: State representation and operator contracts",
                       "v2.tex: Structural recursion"],
        },
        "figure2": {
            "file": "Figure2_experimental_design",
            "content": "Corpus provenance, five-experiment chain, matched controls, evaluation.",
            "frozen_counts": {"analytic_reviews": 2221,
                              "organization_nodes": 14332, "heldout_nodes": 2881},
            "sampling_frame_detail": "SI Note 1 (not shown in Figure 2a)",
            "source": ["v2.tex: Experimental design", "v2_SI.tex: Supplementary Notes 1-2"],
        },
        "style": "single navy identity colour; muted ochre/teal only where meaningful; "
                 "hairline strokes; pale fills; short noun-phrase panel titles; "
                 "vector PDF/SVG plus 600-dpi PNG/TIFF; colour-blind-safe.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "concept_figures_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(provenance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
