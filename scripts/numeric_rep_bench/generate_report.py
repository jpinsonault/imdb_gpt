import argparse
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from .tracking import StageTracker
from .tasks import REGIMES_ORDERED


def _load_results(run_dir):
    tracker = StageTracker(run_dir)
    stages = tracker.all_results()
    results = [v["result"] for v in stages.values()]
    return results


def _family_of(rep_name):
    if rep_name.startswith("scalar_"):
        return "scalar"
    if rep_name.startswith("digits_b"):
        return "digits"
    return "other"


def _group_by_task_and_rep(results):
    by_tr = defaultdict(list)
    for r in results:
        key = (r["task"], r["rep"])
        by_tr[key].append(float(r["primary_metric"]))
    return by_tr


def _group_by_rep_and_regime(results):
    by_rep_reg = defaultdict(lambda: defaultdict(list))
    for r in results:
        rep = r["rep"]
        mreg = r.get("metrics_by_regime", {}) or {}
        for reg, val in mreg.items():
            by_rep_reg[rep][reg].append(float(val))
    return by_rep_reg


def _ensure_plots_dir(run_dir):
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def _plot_per_task_bars(run_dir, results):
    plots_dir = _ensure_plots_dir(run_dir)
    by_tr = _group_by_task_and_rep(results)

    tasks = sorted({r["task"] for r in results})
    for task in tasks:
        rep_to_vals = {}
        for (t, rep), vals in by_tr.items():
            if t == task and vals:
                rep_to_vals[rep] = float(np.mean(vals))
        if not rep_to_vals:
            continue

        reps = sorted(rep_to_vals.keys(), key=lambda r: rep_to_vals[r], reverse=True)
        vals = [rep_to_vals[r] for r in reps]

        fig, ax = plt.subplots()
        x = np.arange(len(reps))
        ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(reps, rotation=45, ha="right")
        ax.set_ylabel("primary metric")
        ax.set_title(f"{task}: performance per representation")

        fig.tight_layout()
        out = os.path.join(plots_dir, f"{task}_per_rep.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)


def _plot_matrix_summary(run_dir, results):
    plots_dir = _ensure_plots_dir(run_dir)
    by_tr = _group_by_task_and_rep(results)

    tasks = sorted({r["task"] for r in results})
    reps = sorted({r["rep"] for r in results})
    if not tasks or not reps:
        return

    mat = np.full((len(tasks), len(reps)), np.nan, dtype=float)
    for i, t in enumerate(tasks):
        for j, rep in enumerate(reps):
            vals = by_tr.get((t, rep))
            if vals:
                mat[i, j] = float(np.mean(vals))

    if np.all(np.isnan(mat)):
        return

    fig, ax = plt.subplots()
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(reps)))
    ax.set_xticklabels(reps, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks)

    for i in range(len(tasks)):
        for j in range(len(reps)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)

    ax.set_title("Average primary metric by task and representation")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("primary metric")

    fig.tight_layout()
    out = os.path.join(plots_dir, "matrix_summary.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_scalar_vs_digits_overall(run_dir, results):
    plots_dir = _ensure_plots_dir(run_dir)

    fam_vals = defaultdict(list)
    for r in results:
        fam = _family_of(r["rep"])
        if fam in ("scalar", "digits"):
            fam_vals[fam].append(float(r["primary_metric"]))

    if not fam_vals:
        return

    fams = sorted(fam_vals.keys())
    means = [float(np.mean(fam_vals[f])) for f in fams]
    stds = [float(np.std(fam_vals[f])) for f in fams]

    fig, ax = plt.subplots()
    x = np.arange(len(fams))
    ax.bar(x, means, yerr=stds)
    ax.set_xticks(x)
    ax.set_xticklabels(fams)
    ax.set_ylabel("primary metric")
    ax.set_title("Overall scalar vs digits performance")

    fig.tight_layout()
    out = os.path.join(plots_dir, "scalar_vs_digits_overall.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _plot_scalar_vs_digits_by_task(run_dir, results):
    plots_dir = _ensure_plots_dir(run_dir)
    by_tr = _group_by_task_and_rep(results)

    tasks = sorted({r["task"] for r in results})
    if not tasks:
        return

    fams = ["scalar", "digits"]

    fig, ax = plt.subplots()
    width = 0.35
    x = np.arange(len(tasks))

    fam_means = {f: [] for f in fams}
    for task in tasks:
        fam_raw = {f: [] for f in fams}
        for (t, rep), vals in by_tr.items():
            if t != task or not vals:
                continue
            fam = _family_of(rep)
            if fam in fam_raw:
                fam_raw[fam].append(float(np.mean(vals)))
        max_val = max([max(v) for v in fam_raw.values() if v] or [1.0])
        for f in fams:
            if fam_raw[f]:
                fam_means[f].append(float(np.mean(fam_raw[f]) / max_val))
            else:
                fam_means[f].append(0.0)

    offset = -width / 2
    for idx, f in enumerate(fams):
        vals = fam_means[f]
        xpos = x + offset + idx * width
        ax.bar(xpos, vals, width, label=f)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_ylabel("normalized metric (per task)")
    ax.set_title("Scalar vs digits by task (normalized)")
    ax.legend()

    fig.tight_layout()
    out = os.path.join(plots_dir, "scalar_vs_digits_by_task.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _plot_scalar_vs_digits_by_regime(run_dir, results):
    plots_dir = _ensure_plots_dir(run_dir)
    by_rep_reg = _group_by_rep_and_regime(results)

    fams = ["scalar", "digits"]
    regimes = [r for r in REGIMES_ORDERED]

    fam_reg_vals = {f: {reg: [] for reg in regimes} for f in fams}
    for rep, reg_map in by_rep_reg.items():
        fam = _family_of(rep)
        if fam not in fams:
            continue
        for reg in regimes:
            vals = reg_map.get(reg)
            if vals:
                fam_reg_vals[fam][reg].extend(vals)

    if not any(fam_reg_vals[f][reg] for f in fams for reg in regimes):
        return

    fig, ax = plt.subplots()
    width = 0.35
    x = np.arange(len(regimes))

    for idx, fam in enumerate(fams):
        means = []
        for reg in regimes:
            vals = fam_reg_vals[fam][reg]
            means.append(float(np.mean(vals)) if vals else 0.0)
        xpos = x + (idx - 0.5) * width
        ax.bar(xpos, means, width, label=fam)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=45, ha="right")
    ax.set_ylabel("primary metric")
    ax.set_title("Scalar vs digits by evaluation regime")
    ax.legend()

    fig.tight_layout()
    out = os.path.join(plots_dir, "scalar_vs_digits_by_regime.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _plot_per_rep_regime_heatmap(run_dir, results):
    plots_dir = _ensure_plots_dir(run_dir)
    by_rep_reg = _group_by_rep_and_regime(results)

    reps = sorted(by_rep_reg.keys())
    regimes = [r for r in REGIMES_ORDERED]
    if not reps or not regimes:
        return

    mat = np.full((len(reps), len(regimes)), np.nan, dtype=float)
    for i, rep in enumerate(reps):
        for j, reg in enumerate(regimes):
            vals = by_rep_reg[rep].get(reg)
            if vals:
                mat[i, j] = float(np.mean(vals))

    if np.all(np.isnan(mat)):
        return

    fig, ax = plt.subplots()
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(regimes)))
    ax.set_xticklabels(regimes, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(reps)))
    ax.set_yticklabels(reps)

    for i in range(len(reps)):
        for j in range(len(regimes)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5)

    ax.set_title("Representation performance across regimes")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("primary metric")

    fig.tight_layout()
    out = os.path.join(plots_dir, "per_rep_regime_heatmap.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_per_rep_regime_lines(run_dir, results, max_reps=6):
    plots_dir = _ensure_plots_dir(run_dir)
    by_rep_reg = _group_by_rep_and_regime(results)

    reps = sorted(by_rep_reg.keys())
    regimes = [r for r in REGIMES_ORDERED]
    if not reps or not regimes:
        return

    rep_scores = []
    for rep in reps:
        vals = []
        for reg_vals in by_rep_reg[rep].values():
            vals.extend(reg_vals)
        if vals:
            rep_scores.append((rep, float(np.mean(vals))))
    if not rep_scores:
        return

    rep_scores.sort(key=lambda x: x[1], reverse=True)

    picked = []
    seen_fam = {"scalar": 0, "digits": 0, "other": 0}
    for rep, _ in rep_scores:
        fam = _family_of(rep)
        if fam in ("scalar", "digits"):
            if seen_fam[fam] < max_reps:
                picked.append(rep)
                seen_fam[fam] += 1
        if len(picked) >= max_reps:
            break
    if not picked:
        picked = [rep_scores[0][0]]

    fig, ax = plt.subplots()
    x = np.arange(len(regimes))

    for rep in picked:
        ys = []
        for reg in regimes:
            vals = by_rep_reg[rep].get(reg)
            ys.append(float(np.mean(vals)) if vals else np.nan)
        ax.plot(x, ys, marker="o", label=rep)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=45, ha="right")
    ax.set_ylabel("primary metric")
    ax.set_title("Selected representations across regimes")
    ax.legend(fontsize=6)

    fig.tight_layout()
    out = os.path.join(plots_dir, "per_rep_regime_lines.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _build_primary_table_rows(results):
    by_tr = _group_by_task_and_rep(results)
    rows = []
    for (task, rep), vals in sorted(by_tr.items()):
        if not vals:
            continue
        metric = float(np.mean(vals))
        rows.append(
            f"{_tex_escape(task)} & {_tex_escape(rep)} & {metric:.4f} \\\\"
        )
    return rows

def _build_family_table_rows(results):
    fam_vals = defaultdict(list)
    for r in results:
        fam = _family_of(r["rep"])
        if fam in ("scalar", "digits"):
            fam_vals[fam].append(float(r["primary_metric"]))
    rows = []
    for fam in sorted(fam_vals.keys()):
        vals = fam_vals[fam]
        if not vals:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        rows.append(
            f"{_tex_escape(fam)} & {mean:.4f} & {std:.4f} \\\\"
        )
    return rows



def _tex_escape(s: str) -> str:
    """
    Escape LaTeX special chars in a lightweight way for table/text content.
    """
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(s))



def _write_tex(run_dir, results):
    plots_dir = os.path.join(run_dir, "plots")
    run_id = os.path.basename(os.path.normpath(run_dir))
    run_id_tex = _tex_escape(run_id)

    tasks = sorted({r["task"] for r in results})
    have_task = {t: os.path.exists(os.path.join(plots_dir, f"{t}_per_rep.png")) for t in tasks}

    primary_rows = _build_primary_table_rows(results)
    family_rows = _build_family_table_rows(results)

    lines = []

    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[margin=1.1in]{geometry}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{array}")
    lines.append(r"\usepackage{amsmath}")
    lines.append(r"\usepackage{amssymb}")
    lines.append(r"\usepackage{microtype}")
    lines.append(r"\usepackage{hyperref}")
    lines.append(r"\usepackage{caption}")
    lines.append(r"\usepackage{subcaption}")
    lines.append(r"\usepackage{enumitem}")
    lines.append(r"\usepackage{titlesec}")
    lines.append(r"\setlength{\parskip}{6pt}")
    lines.append(r"\setlength{\parindent}{0pt}")
    lines.append(r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=black,citecolor=black,pdfborder={0 0 0}}")
    lines.append(r"\titleformat{\section}{\large\bfseries}{\thesection}{0.75em}{}")
    lines.append(r"\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.5em}{}")
    lines.append(r"\begin{document}")
    lines.append(r"\begin{center}")
    lines.append(r"{\LARGE Numeric Representation Benchmark}\\[4pt]")
    lines.append(r"{\large Scalar vs Digit Encodings}\\[6pt]")
    lines.append(rf"{{\normalsize Run ID: {run_id_tex}}}")
    lines.append(r"\end{center}")
    lines.append(r"\vspace{1em}")

    lines.append(r"\section{Overview}")
    lines.append(
        "This report summarizes results from the numeric representation benchmark. "
        "The benchmark evaluates how different encodings of real-valued inputs affect "
        "downstream performance on simple supervised tasks under a fixed model class."
    )

    lines.append(r"\section{Benchmark Design}")
    lines.append(r"\subsection{Tasks}")
    lines.append(
        "Tasks operate on low-dimensional real inputs and deterministic targets:"
    )
    lines.append(r"\begin{itemize}[noitemsep,topsep=2pt]")
    if "add" in tasks:
        lines.append(r"\item Addition (\texttt{add}): $y = x_0 + x_1$.")
    if "mul" in tasks:
        lines.append(r"\item Multiplication (\texttt{mul}): $y = x_0 \cdot x_1$.")
    if "sin" in tasks:
        lines.append(r"\item Sine (\texttt{sin}): $y = \sin(x_0)$.")
    if "gt" in tasks:
        lines.append(r"\item Comparison (\texttt{gt}): binary label for $(x_0 > x_1)$.")
    if "parity" in tasks:
        lines.append(r"\item Integer parity (\texttt{parity}): parity of an integer input.")
    lines.append(r"\end{itemize}")

    lines.append(r"\subsection{Numeric Regimes}")
    lines.append(
        r"Inputs are sampled from predefined numeric regimes (small, medium, large, mixed, near\_zero). "
        "Unless stated otherwise, training uses the medium regime and evaluation is reported across all regimes."
    )

    lines.append(r"\subsection{Representations}")
    lines.append(
        "Each input coordinate is encoded either as a scalar value with a fixed transform, "
        "or as a digit-wise code with configurable base and fractional precision."
    )
    lines.append(
        "Scalar variants include raw, standardized, min-max scaled, and log-transformed. "
        r"Digit encodings follow the \texttt{digits\_bX\_fY} configurations and mirror the "
        "numeric digit field design used in the codebase."
    )

    lines.append(r"\section{Experimental Protocol}")
    lines.append(r"\subsection{Model}")
    lines.append(
        "For each task and representation, inputs are encoded, mapped to a fixed-size vector, "
        "and passed through a small MLP head. Architecture and capacity are shared across representations."
    )
    lines.append(r"\subsection{Training}")
    lines.append(r"\begin{itemize}[noitemsep,topsep=2pt]")
    lines.append(r"\item Shared optimizer and hyperparameters across runs.")
    lines.append(r"\item Fixed number of epochs and batch size.")
    lines.append(r"\item Multiple seeds per configuration; metrics are averaged.")
    lines.append(
        r"\item Regression tasks report R$^2$; classification tasks report accuracy."
    )
    lines.append(r"\end{itemize}")

    lines.append(r"\section{Results}")

    matrix_path = os.path.join(plots_dir, "matrix_summary.png")
    if os.path.exists(matrix_path):
        lines.append(r"\subsection{Global comparison}")
        lines.append(r"\begin{figure}[h]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.98\linewidth]{plots/matrix_summary.png}")
        lines.append(
            r"\caption{Average primary metric across tasks and representations. "
            r"Rows correspond to tasks; columns correspond to representations.}"
        )
        lines.append(r"\end{figure}")

    lines.append(r"\subsection{Per-task summaries}")
    for task in ["add", "mul", "sin", "gt", "parity"]:
        if have_task.get(task):
            lines.append(r"\begin{figure}[h]")
            lines.append(r"\centering")
            lines.append(rf"\includegraphics[width=0.9\linewidth]{{plots/{task}_per_rep.png}}")
            lines.append(rf"\caption{{{task}: performance per representation.}}")
            lines.append(r"\end{figure}")

    svd_overall = os.path.join(plots_dir, "scalar_vs_digits_overall.png")
    svd_task = os.path.join(plots_dir, "scalar_vs_digits_by_task.png")
    svd_reg = os.path.join(plots_dir, "scalar_vs_digits_by_regime.png")

    if os.path.exists(svd_overall) or os.path.exists(svd_task) or os.path.exists(svd_reg):
        lines.append(r"\subsection{Scalar vs digit encodings}")

    if os.path.exists(svd_overall):
        lines.append(r"\begin{figure}[h]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.9\linewidth]{plots/scalar_vs_digits_overall.png}")
        lines.append(
            r"\caption{Overall comparison of scalar and digit encoding families. "
            r"Bars show mean primary metric across tasks and seeds.}"
        )
        lines.append(r"\end{figure}")

    if os.path.exists(svd_task):
        lines.append(r"\begin{figure}[h]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.9\linewidth]{plots/scalar_vs_digits_by_task.png}")
        lines.append(
            r"\caption{Scalar vs digit encodings by task. "
            r"Values are normalized per task to highlight relative preference.}"
        )
        lines.append(r"\end{figure}")

    if os.path.exists(svd_reg):
        lines.append(r"\begin{figure}[h]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.9\linewidth]{plots/scalar_vs_digits_by_regime.png}")
        lines.append(
            r"\caption{Scalar vs digit encodings by evaluation regime. "
            r"Bars show average primary metric across tasks for each regime.}"
        )
        lines.append(r"\end{figure}")

    heatmap_path = os.path.join(plots_dir, "per_rep_regime_heatmap.png")
    lines_path = os.path.join(plots_dir, "per_rep_regime_lines.png")

    if os.path.exists(heatmap_path) or os.path.exists(lines_path):
        lines.append(r"\subsection{Regime generalization}")

    if os.path.exists(heatmap_path):
        lines.append(r"\begin{figure}[h]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.9\linewidth]{plots/per_rep_regime_heatmap.png}")
        lines.append(
            r"\caption{Heatmap of representation performance across evaluation regimes.}"
        )
        lines.append(r"\end{figure}")

    if os.path.exists(lines_path):
        lines.append(r"\begin{figure}[h]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.9\linewidth]{plots/per_rep_regime_lines.png}")
        lines.append(
            r"\caption{Selected representations across regimes. "
            r"Lines show primary metric for chosen scalar and digit encodings.}"
        )
        lines.append(r"\end{figure}")

    lines.append(r"\section{Summary tables}")
    lines.append(r"\subsection{Primary metrics}")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l l c}")
    lines.append(r"\toprule")
    lines.append(r"Task & Representation & Primary metric \\")
    lines.append(r"\midrule")
    for row in primary_rows:
        lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Primary metric per task and representation, averaged over seeds.}"
    )
    lines.append(r"\end{table}")

    lines.append(r"\subsection{Family-level aggregates}")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l c c}")
    lines.append(r"\toprule")
    lines.append(r"Family & Mean primary metric & Std across runs \\")
    lines.append(r"\midrule")
    for row in family_rows:
        lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Aggregated comparison of scalar and digit encoding families.}"
    )
    lines.append(r"\end{table}")

    lines.append(r"\section{Implementation notes}")
    lines.append(
        "Results are loaded from the benchmark status file and per-run artifacts. "
        "Plots are generated from the recorded metrics without manual adjustment."
    )
    lines.append(
        "Representations share the same downstream model class, optimizer, and training schedule."
    )

    lines.append(r"\section{Context}")
    lines.append(
        "The benchmark is motivated by numeric encoding choices in tabular and relational models. "
        "The design is standalone and can be applied independently of any specific dataset."
    )

    lines.append(r"\end{document}")

    out_path = os.path.join(run_dir, "numeric_rep_bench_report.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to a completed runs/numeric_rep_bench/<run_id> directory",
    )
    args = parser.parse_args()
    run_dir = args.run_dir

    if not os.path.isdir(run_dir):
        raise SystemExit(f"run-dir not found: {run_dir}")

    results = _load_results(run_dir)
    if not results:
        raise SystemExit("no results found in status.json / StageTracker")

    _plot_per_task_bars(run_dir, results)
    _plot_matrix_summary(run_dir, results)
    _plot_scalar_vs_digits_overall(run_dir, results)
    _plot_scalar_vs_digits_by_task(run_dir, results)
    _plot_scalar_vs_digits_by_regime(run_dir, results)
    _plot_per_rep_regime_heatmap(run_dir, results)
    _plot_per_rep_regime_lines(run_dir, results)

    _write_tex(run_dir, results)


if __name__ == "__main__":
    main()
