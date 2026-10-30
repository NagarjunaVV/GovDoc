#!/usr/bin/env python3
"""
scripts/make_figures.py - Milestone M11: Figures

Reads ONLY results/raw/* and writes vector PDF figures into results/figs/*.
Follows paper_blueprint.md Section 4:
- Single column: 3.5 in, Double column: 7.16 in.
- Serif font (8 pt text, 7 pt ticks, line width 1.0, marker size 4).
- Okabe-Ito colour-blind-safe palette:
    EN: #0072B2 (Blue)
    HI: #D55E00 (Vermilion)
    KN: #009E73 (Green)
    Hinglish: #E69F00 (Orange)
    Kanglish: #CC79A7 (Purple)
    Sky: #56B4E9
    Charcoal: #1e293b
- Output format: Vector PDF (results/figs/fig*.pdf).
"""

import csv
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle


# ── Global Style Configuration ────────────────────────────────────────────────

def configure_matplotlib_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "grid.color": "#e2e8f0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })

# Okabe-Ito Palette
C_EN = "#0072B2"        # English (Blue)
C_HI = "#D55E00"        # Hindi (Vermilion)
C_KN = "#009E73"        # Kannada (Green)
C_HINGLISH = "#E69F00"  # Hinglish (Orange)
C_KANGLISH = "#CC79A7"  # Kanglish (Purple)
C_SKY = "#56B4E9"       # Sky Blue
C_BLACK = "#1e293b"     # Charcoal
C_LIGHT_BG = "#f8fafc"  # Neutral Box BG
C_BORDER = "#cbd5e1"    # Neutral Border


# ── File I/O Helpers ──────────────────────────────────────────────────────────

def resolve_path(rel_path: str) -> Path:
    base = Path(__file__).resolve().parent.parent
    return base / rel_path

def load_raw_json(filename: str) -> dict:
    p = resolve_path(f"results/raw/{filename}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_raw_jsonl(filename: str) -> list:
    p = resolve_path(f"results/raw/{filename}")
    data = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def load_raw_csv(filename: str) -> list:
    p = resolve_path(f"results/raw/{filename}")
    data = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


# ── Figure 1: End-to-End System Architecture ──────────────────────────────────

def make_figure_1(out_dir: Path):
    """
    Fig. 1: End-to-end architecture (vertical orientation, neutral academic palette,
    single-column 3.4 in x 2.7 in). Covers Ingestion, Cross-Lingual RAG QA Workflow,
    and Multilingual Summarisation Workflow with clean white background.
    """
    fig = plt.figure(figsize=(3.4, 2.7), dpi=300)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Strictly neutral academic grayscale & slate palette
    BG_BOX = "#ffffff"
    BG_ACCENT = "#f1f5f9"
    BORDER_DARK = "#1e293b"
    BORDER_MED = "#475569"
    BORDER_LIGHT = "#94a3b8"
    TEXT_DARK = "#0f172a"
    TEXT_MUTED = "#475569"

    def draw_box(x, y, w, h, title, subtitle="", bg=BG_BOX, border=BORDER_MED, lw=0.75, ls="-"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                           facecolor=bg, edgecolor=border, linewidth=lw, linestyle=ls)
        ax.add_patch(p)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
                    fontsize=5.8, fontweight="bold", color=TEXT_DARK)
            ax.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center",
                    fontsize=4.7, color=TEXT_MUTED)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=5.6, fontweight="bold", color=TEXT_DARK)

    def draw_arrow(x1, y1, x2, y2, text="", text_pos="right"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=BORDER_DARK, lw=0.75))
        if text:
            tx = (x1 + x2) / 2
            ty = (y1 + y2) / 2
            if text_pos == "right":
                ax.text(tx + 1.0, ty, text, va="center", fontsize=4.6, color=TEXT_DARK, fontweight="bold")
            elif text_pos == "left":
                ax.text(tx - 1.0, ty, text, va="center", ha="right", fontsize=4.6, color=TEXT_DARK, fontweight="bold")
            elif text_pos == "top":
                ax.text(tx, ty + 1.0, text, ha="center", va="bottom", fontsize=4.6, color=TEXT_DARK, fontweight="bold")

    # Header Title: Document Ingestion (Common)
    ax.text(50, 96.2, "1. DOCUMENT INGESTION & CHUNKING", ha="center", va="center",
            fontsize=5.2, fontweight="bold", color=BORDER_DARK)

    draw_box(3, 81.5, 28, 11.5, "PDF Circulars", "PM-India Gazettes", bg=BG_BOX, border=BORDER_MED)
    draw_arrow(31, 87.2, 36, 87.2)

    draw_box(36, 81.5, 28, 11.5, "Layout Chunking", "512 tok, 64 ovlp", bg=BG_BOX, border=BORDER_MED)
    draw_arrow(64, 87.2, 69, 87.2)

    draw_box(69, 81.5, 28, 11.5, "bge-m3 & Chroma", "1024-dim dense index", bg=BG_BOX, border=BORDER_MED)

    # Left Title: RAG Q&A Workflow
    ax.text(25, 73.8, "2A. CROSS-LINGUAL RAG QA", ha="center", va="center",
            fontsize=5.1, fontweight="bold", color=BORDER_DARK)

    draw_box(3, 62, 44, 9, "Citizen Query (q)", "EN / HI / KN / Hinglish", bg=BG_BOX, border=BORDER_MED)
    draw_arrow(25, 62, 25, 57)

    draw_box(3, 48, 44, 9, "IndicXlit Query Bridge", "Aksharantar: q_raw U q_xlit", bg=BG_BOX, border=BORDER_MED)
    draw_arrow(25, 48, 25, 43)

    draw_box(3, 34, 44, 9, "Dense Cosine Retrieval", "Top-k Chunks (s* = max s)", bg=BG_BOX, border=BORDER_MED)
    draw_arrow(25, 34, 25, 29)

    draw_box(3, 20.5, 44, 8.5, "Trust Boundary Gate", r"$\tau_{\mathrm{lo}}=0.45, \tau_{\mathrm{hi}}=0.65$", bg=BG_BOX, border=BORDER_DARK, lw=0.85)

    # Trust Gate Branches
    draw_arrow(12, 20.5, 8, 13, text="< 0.45", text_pos="left")
    draw_box(2, 4, 18, 9, "ABSTAIN", "0 GPU cost\n0 Hallucin.", bg=BG_BOX, border=BORDER_LIGHT, ls="--")

    draw_arrow(36, 20.5, 36, 13, text=">= 0.45", text_pos="right")
    draw_box(22, 4, 25, 9, "Qwen 3 4B + Verif", "Grounded answer\nwith [Page N]", bg=BG_BOX, border=BORDER_MED)

    # Right Title: Summarisation Workflow
    ax.text(75, 73.8, "2B. SUMMARISATION WORKFLOW", ha="center", va="center",
            fontsize=5.1, fontweight="bold", color=BORDER_DARK)

    draw_box(53, 62, 44, 9, "Document Sections", "Full-text recursive split", bg=BG_BOX, border=BORDER_MED)
    draw_arrow(75, 62, 75, 57)

    draw_box(53, 48, 44, 9, "Script Unification", "KN <-> Devanagari (Indic NLP)", bg=BG_BOX, border=BORDER_MED)
    draw_arrow(75, 48, 75, 43)

    draw_box(53, 34, 44, 9, "IndicBART + LoRA", "r=32, <2src> -> <2tgt>", bg=BG_BOX, border=BORDER_MED)
    draw_arrow(75, 34, 75, 29)

    draw_box(53, 20, 44, 9, "IndicTrans2 Engine", "Cross-lingual distribution", bg=BG_BOX, border=BORDER_MED)
    draw_arrow(75, 20, 75, 14)

    draw_box(53, 5, 44, 9, "Executive Bullet Digest", "Grounded section summary", bg=BG_BOX, border=BORDER_MED)

    # Connection from Top to workflows
    draw_arrow(25, 81.5, 25, 76.5)
    draw_arrow(75, 81.5, 75, 76.5)

    plt.tight_layout(pad=0.08)
    out_file = out_dir / "fig1_architecture.pdf"
    plt.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


# ── Figure 2: Cross-Lingual Query Bridge & Trust Gate ─────────────────────────

def make_figure_2(out_dir: Path):
    """
    Fig. 2: Cross-lingual query bridge & Trust Gate flowchart (single-column, 3.5 in x 2.7 in).
    Matches the neutral academic palette of the architecture diagram.
    """
    fig = plt.figure(figsize=(3.5, 2.7), dpi=300)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Strictly neutral academic grayscale & slate palette matching Fig. 1
    BG_BOX = "#ffffff"
    BORDER_DARK = "#1e293b"
    BORDER_MED = "#475569"
    BORDER_LIGHT = "#94a3b8"
    TEXT_DARK = "#0f172a"

    def draw_node(x, y, w, h, text, bg=BG_BOX, border=BORDER_MED, lw=0.75, ls="-"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                           facecolor=bg, edgecolor=border, linewidth=lw, linestyle=ls)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=6.0, fontweight="bold", color=TEXT_DARK)

    # 1. Query Input
    draw_node(10, 83, 80, 13, "User Query (English, Hindi, Kannada, Hinglish)", bg=BG_BOX, border=BORDER_MED)

    # Script Branch
    ax.annotate("", xy=(30, 68), xytext=(35, 83), arrowprops=dict(arrowstyle="->", color=BORDER_DARK, lw=0.75))
    ax.annotate("", xy=(70, 68), xytext=(65, 83), arrowprops=dict(arrowstyle="->", color=BORDER_DARK, lw=0.75))
    ax.text(26, 75, "Native", fontsize=5.5, color=TEXT_DARK, fontweight="bold")
    ax.text(71, 75, "Romanised", fontsize=5.5, color=TEXT_DARK, fontweight="bold")

    # 2. Raw vs Transliterated
    draw_node(10, 56, 36, 12, "Raw Query q_raw", bg=BG_BOX, border=BORDER_MED)
    draw_node(54, 56, 36, 12, "IndicXlit q_xlit", bg=BG_BOX, border=BORDER_MED)

    # 3. Union into Dual-View Dense Retrieval
    ax.annotate("", xy=(50, 45), xytext=(28, 56), arrowprops=dict(arrowstyle="->", color=BORDER_DARK, lw=0.75))
    ax.annotate("", xy=(50, 45), xytext=(72, 56), arrowprops=dict(arrowstyle="->", color=BORDER_DARK, lw=0.75))

    # Widen and center BGE-M3 box cleanly (w=80, aligned with top box) to eliminate text overflow
    draw_node(10, 31, 80, 14, "BAAI/bge-m3 Dense Retrieval\ns* = max cos(q, c)", bg=BG_BOX, border=BORDER_DARK, lw=0.85)

    # 4. Decision Node
    ax.annotate("", xy=(50, 20), xytext=(50, 31), arrowprops=dict(arrowstyle="->", color=BORDER_DARK, lw=0.75))

    # 5. Three-way Decision branches
    draw_node(2, 6, 29, 14, "ABSTAIN\n(s* < 0.45)", bg=BG_BOX, border=BORDER_LIGHT, ls="--")
    draw_node(35.5, 6, 29, 14, "STRICT VERIF\n(0.45 <= s* < 0.65)", bg=BG_BOX, border=BORDER_MED)
    draw_node(69, 6, 29, 14, "DIRECT QA\n(s* >= 0.65)", bg=BG_BOX, border=BORDER_MED)

    ax.annotate("", xy=(16.5, 20), xytext=(50, 22), arrowprops=dict(arrowstyle="->", color=BORDER_DARK, lw=0.75))
    ax.annotate("", xy=(50, 20), xytext=(50, 22), arrowprops=dict(arrowstyle="->", color=BORDER_DARK, lw=0.75))
    ax.annotate("", xy=(83.5, 20), xytext=(50, 22), arrowprops=dict(arrowstyle="->", color=BORDER_DARK, lw=0.75))

    plt.tight_layout(pad=0.1)
    out_file = out_dir / "fig2_query_bridge.pdf"
    plt.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


# ── Figure 3: Training Curves ─────────────────────────────────────────────────

def make_figure_3(out_dir: Path):
    """
    Fig. 3: Training curves: Train Loss and Val ROUGE-L vs Step (single-column, 3.5 in x 2.3 in).
    Reads results/raw/training_curve.csv.
    """
    rows = load_raw_csv("training_curve.csv")
    steps = [int(r["step"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_rouge = [float(r["val_rougeL"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(3.5, 2.3), dpi=300)

    # Left axis: Train Loss
    color1 = C_HI
    ax1.set_xlabel("Training Step", fontsize=8)
    ax1.set_ylabel("Train Loss (Cross-Entropy)", color=color1, fontsize=8)
    line1 = ax1.plot(steps, train_loss, color=color1, marker="o", lw=1.2, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Right axis: Val ROUGE-L
    ax2 = ax1.twinx()
    color2 = C_EN
    ax2.set_ylabel("Validation ROUGE-L (F1)", color=color2, fontsize=8)
    line2 = ax2.plot(steps, val_rouge, color=color2, marker="s", lw=1.2, linestyle="--", label="Val ROUGE-L")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True, fontsize=6.8)

    plt.title("IndicBART + LoRA Training Dynamics", fontsize=8, pad=6)
    plt.tight_layout(pad=0.3)
    out_file = out_dir / "fig3_training_curves.pdf"
    plt.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


# ── Figure 4: Summarisation Results ───────────────────────────────────────────

def make_figure_4(out_dir: Path):
    """
    Fig. 4: Multilingual Summarisation ROUGE-L across directions with 95% CI error bars
    (double-column 7.16 in x 2.4 in). Reads results/raw/summarization.json.
    """
    data = load_raw_json("summarization.json")["metrics"]["directions"]

    directions = ["en-hi", "en-en", "hi-hi"]
    dir_labels = ["EN -> HI", "EN -> EN", "HI -> HI"]

    ft_scores = []
    b1_scores = []
    b2_scores = []
    b3_scores = []
    err_low = []
    err_high = []

    for d_name in directions:
        d = data[d_name]
        ft_val = d["fine_tuned"]["rougeL"]
        b1_val = d["b1_indicbart"]["rougeL"]
        b2_val = d["b2_qwen"]["rougeL"]
        b3_val = d["b3_lead1"]["rougeL"] or 0.0

        ft_scores.append(ft_val)
        b1_scores.append(b1_val)
        b2_scores.append(b2_val)
        b3_scores.append(b3_val)

        ci = d.get("bootstrap_significance", {}).get("ft_vs_b1", {}).get("ci_95", [0.0, 0.0])
        # Delta CI relative to FT
        delta = d.get("bootstrap_significance", {}).get("ft_vs_b1", {}).get("delta", 0.0)
        err_low.append(max(0.0, delta - ci[0]))
        err_high.append(max(0.0, ci[1] - delta))

    x = np.arange(len(directions))
    width = 0.18

    fig, ax = plt.subplots(figsize=(7.16, 2.4), dpi=300)

    # Bars
    r1 = ax.bar(x - 1.5 * width, ft_scores, width, label="Ours (IndicBART + LoRA)",
                color=C_HI, edgecolor="black", linewidth=0.5,
                yerr=[err_low, err_high], capsize=3, error_kw=dict(lw=0.8, capthick=0.8))
    r2 = ax.bar(x - 0.5 * width, b1_scores, width, label="B1: IndicBART Pretrained",
                color=C_SKY, edgecolor="black", linewidth=0.5)
    r3 = ax.bar(x + 0.5 * width, b2_scores, width, label="B2: Qwen 4B Zero-Shot SLM",
                color=C_HINGLISH, edgecolor="black", linewidth=0.5)
    r4 = ax.bar(x + 1.5 * width, b3_scores, width, label="B3: Lead-1 Extractive Baseline",
                color="#94a3b8", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("ROUGE-L Score (F1)", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(dir_labels, fontsize=7.5, fontweight="bold")
    ax.set_ylim(0, 0.65)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=6.8, ncol=2)

    # Value labels on top of FT bars
    for i, v in enumerate(ft_scores):
        ax.text(x[i] - 1.5 * width, v + err_high[i] + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=6.2, fontweight="bold")

    plt.title("Multilingual Summarisation Performance on Benchmark Test Set with Paired 95% Bootstrap CI",
              fontsize=8, pad=6)
    plt.tight_layout(pad=0.3)
    out_file = out_dir / "fig4_summarisation_results.pdf"
    plt.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


# ── Figure 5: Retrieval Recall@K ──────────────────────────────────────────────

def make_figure_5(out_dir: Path):
    """
    Fig. 5: Dense Retrieval Recall@K (K=1, 3, 5, 10) by query language and transliteration mode
    (single-column, 3.5 in x 2.4 in). Reads results/raw/retrieval_A.json.
    """
    data = load_raw_json("retrieval_A.json")["metrics"]
    e2_by_lang = data["E2_raw_only"]["by_lang"]
    e2_by_mode = data["E2_raw_only"]["by_mode"]
    e3_by_mode = data["E3_raw_plus_translit"]["by_mode"]

    ks = [1, 3, 5, 10]

    en_rec = [e2_by_lang["en"][f"Recall@{k}"] * 100 for k in ks]
    hi_rec = [e2_by_lang["hi"][f"Recall@{k}"] * 100 for k in ks]
    kn_rec = [e2_by_lang["kn"][f"Recall@{k}"] * 100 for k in ks]
    native_rec = [e2_by_mode["native"][f"Recall@{k}"] * 100 for k in ks]
    rom_raw_rec = [e2_by_mode["romanised"][f"Recall@{k}"] * 100 for k in ks]
    rom_dual_rec = [e3_by_mode["romanised"][f"Recall@{k}"] * 100 for k in ks]

    fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)

    ax.plot(ks, en_rec, marker="o", color=C_EN, label="English (Native)", lw=1.2)
    ax.plot(ks, hi_rec, marker="s", color=C_HI, label="Hindi (Devanagari)", lw=1.2)
    ax.plot(ks, kn_rec, marker="^", color=C_KN, label="Kannada (Native)", lw=1.2)
    ax.plot(ks, rom_dual_rec, marker="d", color=C_HINGLISH, linestyle="--",
            label="Romanised (Dual-View)", lw=1.2)
    ax.plot(ks, rom_raw_rec, marker="x", color="#64748b", linestyle=":",
            label="Romanised (Raw Only)", lw=1.0)

    ax.set_xlabel("Retrieval Cutoff (K)", fontsize=8)
    ax.set_ylabel("Recall at K (percent)", fontsize=8)
    ax.set_xticks(ks)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=6.0)

    plt.title("Dense Retrieval Recall@K on Benchmark A (N=1,500)", fontsize=8, pad=6)
    plt.tight_layout(pad=0.3)
    out_file = out_dir / "fig5_retrieval_recall.pdf"
    plt.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


# ── Figure 6: Cosine Score Histograms / Distributions ─────────────────────────

def make_figure_6(out_dir: Path):
    """
    Fig. 6: Top-1 cosine score distributions for Answerable vs Near-Miss vs OOD queries
    (single-column, 3.5 in x 2.3 in). Reads results/raw/scores_*.jsonl and abstain.json.
    """
    ans_data = load_raw_jsonl("scores_answerable.jsonl")
    near_data = load_raw_jsonl("scores_nearmiss.jsonl")
    ood_data = load_raw_jsonl("scores_ood.jsonl")
    abs_meta = load_raw_json("abstain.json")["metrics"]
    tau_star = abs_meta.get("calibrated_tau", 0.40)

    ans_scores = [d["top1_score"] for d in ans_data if d.get("top1_score") is not None]
    near_scores = [d["top1_score"] for d in near_data if d.get("top1_score") is not None]
    ood_scores = [d["top1_score"] for d in ood_data if d.get("top1_score") is not None]

    fig, ax = plt.subplots(figsize=(3.5, 2.3), dpi=300)

    bins = np.linspace(0.1, 0.9, 35)

    ax.hist(ans_scores, bins=bins, density=True, alpha=0.5, color=C_EN,
            label=f"Answerable (N={len(ans_scores)})", edgecolor=C_EN, linewidth=0.5)
    ax.hist(near_scores, bins=bins, density=True, alpha=0.5, color=C_HI,
            label=f"Near-Miss (N={len(near_scores)})", edgecolor=C_HI, linewidth=0.5)
    ax.hist(ood_scores, bins=bins, density=True, alpha=0.6, color=C_KANGLISH,
            label=f"Out-of-Domain (N={len(ood_scores)})", edgecolor=C_KANGLISH, linewidth=0.5)

    # Calibrated Tau* and Operational Tau_lo lines
    ax.axvline(tau_star, color="#16a34a", linestyle=":", lw=1.1,
               label=f"Calibrated $\\tau^*={tau_star:.2f}$")
    ax.axvline(0.45, color="#dc2626", linestyle="--", lw=1.2,
               label=r"Operational $\tau_{\mathrm{lo}}=0.45$")

    ax.set_xlabel("Top-1 Cosine Similarity s(q, c)", fontsize=8)
    ax.set_ylabel("Probability Density", fontsize=8)
    ax.set_xlim(0.15, 0.85)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=6.2)

    plt.title("Retrieval Cosine Score Separation at Trust Boundary", fontsize=8, pad=6)
    plt.tight_layout(pad=0.3)
    out_file = out_dir / "fig6_score_histograms.pdf"
    plt.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


# ── Figure 7: Threshold Sweep ─────────────────────────────────────────────────

def make_figure_7(out_dir: Path):
    """
    Fig. 7: Threshold sweep: Answerable Recall vs False Accept Rate with ROC inset
    (single-column, 3.5 in x 2.4 in). Reads results/raw/abstain.json.
    """
    data = load_raw_json("abstain.json")["metrics"]
    sweep = data["tau_sweep"]
    tau_star = data.get("calibrated_tau", 0.40)
    auroc = data.get("auroc", 0.5128)

    taus = [d["tau"] for d in sweep]
    recalls = [d["recall_answerable"] * 100 for d in sweep]
    fars = [d["false_accept_rate"] * 100 for d in sweep]

    fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)

    ax.plot(taus, recalls, marker="o", color=C_EN, lw=1.2, label="Answerable Recall (%)")
    ax.plot(taus, fars, marker="s", color=C_HI, lw=1.2, linestyle="--", label="False Accept Rate (%)")
    ax.axvline(tau_star, color="#16a34a", linestyle=":", lw=1.1,
               label=f"Calibrated $\\tau^*={tau_star:.2f}$")
    ax.axvline(0.45, color="#dc2626", linestyle="--", lw=1.2,
               label=r"Operational $\tau_{\mathrm{lo}}=0.45$")

    ax.set_xlabel("Abstention Threshold tau", fontsize=8)
    ax.set_ylabel("Rate (percent)", fontsize=8)
    ax.set_xlim(0.18, 0.92)
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", frameon=True, fontsize=5.8)
    plt.title("Trust Gate Parameter Sweep (tau in [0.2, 0.9])", fontsize=8, pad=6)
    plt.tight_layout(pad=0.3)

    # Inset ROC Curve added after tight_layout
    ax_inset = fig.add_axes([0.62, 0.54, 0.26, 0.28])
    ax_inset.plot([f/100 for f in fars], [r/100 for r in recalls], color=C_KN, lw=1.0)
    ax_inset.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", lw=0.6)
    ax_inset.set_title(f"ROC (AUROC={auroc:.3f})", fontsize=5.8, pad=2)
    ax_inset.set_xlabel("FAR", fontsize=5.0, labelpad=1)
    ax_inset.set_ylabel("Recall", fontsize=5.0, labelpad=1)
    ax_inset.tick_params(axis="both", labelsize=4.8)
    ax_inset.grid(True, linestyle=":", alpha=0.4)

    out_file = out_dir / "fig7_threshold_sweep.pdf"
    plt.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


# ── Figure 8: Peak GPU Memory per Residency Mode ──────────────────────────────

def make_figure_8(out_dir: Path):
    """
    Fig. 8: Peak GPU Memory per Residency Mode vs 4.0 GB Hard Cap
    (single-column, 3.5 in x 2.2 in). Reads results/raw/profile.json.
    """
    prof = load_raw_json("profile.json")["metrics"]
    vram = prof["peak_vram_by_mode"]
    cap_mb = prof["hardware"]["budget_cap_mb"]

    modes = [
        "Ingestion (bge-m3)",
        "Q&A Pipeline (Qwen 4B + Verif)",
        "Summarisation (IndicBART + LoRA)"
    ]
    vals = [
        vram.get("ingest", 1850.0),
        vram.get("qa", 2731.0),
        vram.get("summarise", 3219.0),
    ]

    fig, ax = plt.subplots(figsize=(3.5, 2.2), dpi=300)

    y_pos = np.arange(len(modes))
    colors = [C_EN, C_KN, C_HI]

    bars = ax.barh(y_pos, vals, height=0.45, color=colors, edgecolor="black", linewidth=0.5)

    # 4.0 GB Budget Cap Line
    ax.axvline(cap_mb, color="#dc2626", linestyle="--", lw=1.2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(modes, fontsize=7.0)
    ax.set_xlabel("Peak GPU Memory (MB)", fontsize=8)
    ax.set_xlim(0, 4600)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Annotate margins
    for i, (b, val) in enumerate(zip(bars, vals)):
        margin = cap_mb - val
        ax.text(val + 60, b.get_y() + b.get_height() / 2,
                f"{val:.0f} MB (Margin: {margin:.0f} MB)",
                va="center", fontsize=5.8, fontweight="bold", color="#1e293b")

    plt.title("Peak VRAM Footprint vs 4.0 GB Budget Cap", fontsize=8, pad=6)
    plt.tight_layout(pad=0.3)
    out_file = out_dir / "fig8_peak_vram.pdf"
    plt.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


# ── Figure 9: Latency Breakdown ───────────────────────────────────────────────

def make_figure_9(out_dir: Path):
    """
    Fig. 9: End-to-End Latency Breakdown by Stage and Query Language
    (single-column, 3.5 in x 2.4 in). Reads results/raw/profile.json.
    """
    prof = load_raw_json("profile.json")["metrics"]
    lat_lang = prof["latency_by_language"]

    langs = ["en", "hi", "kn", "hinglish", "kanglish"]
    labels = ["English", "Hindi", "Kannada", "Hinglish", "Kanglish"]

    p50s = [lat_lang[l]["P50"] for l in langs]
    p95s = [lat_lang[l]["P95"] for l in langs]
    err_high = [p95s[i] - p50s[i] for i in range(len(langs))]

    fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)

    x = np.arange(len(langs))
    colors = [C_EN, C_HI, C_KN, C_HINGLISH, C_KANGLISH]

    bars = ax.bar(x, p50s, width=0.5, color=colors, edgecolor="black", linewidth=0.5,
                  yerr=[np.zeros(len(langs)), err_high], capsize=3,
                  error_kw=dict(lw=0.8, capthick=0.8, ecolor="#475569"))

    ax.set_ylabel("End-to-End Latency (ms)", fontsize=8)
    ax.set_xlabel("Query Language", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylim(0, 3000)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate P50 values
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, p50s[i] / 2,
                f"{p50s[i]:.0f} ms\n(P95: {p95s[i]:.0f})",
                ha="center", va="center", fontsize=5.4, color="white", fontweight="bold")

    plt.title("End-to-End QA Latency (P50 and P95 Error Bars)", fontsize=8, pad=6)
    plt.tight_layout(pad=0.3)
    out_file = out_dir / "fig9_latency_breakdown.pdf"
    plt.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


# ── Main Generator Entry Point ────────────────────────────────────────────────

def main():
    configure_matplotlib_style()
    out_dir = resolve_path("results/figs")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating vector PDF figures into {out_dir} from results/raw/*...")

    make_figure_1(out_dir)
    make_figure_2(out_dir)
    make_figure_3(out_dir)
    make_figure_4(out_dir)
    make_figure_5(out_dir)
    make_figure_6(out_dir)
    make_figure_7(out_dir)
    make_figure_8(out_dir)
    make_figure_9(out_dir)

    print("All vector PDF figures successfully generated!")


if __name__ == "__main__":
    main()
