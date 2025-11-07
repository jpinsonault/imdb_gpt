import os
from typing import Dict, List

from .tracking import StageTracker


def _pretty_task_name(task: str) -> str:
    if task == "add":
        return "Addition (x + y)"
    if task == "mul":
        return "Multiplication (x × y)"
    if task == "sin":
        return "Sine (sin(x))"
    if task == "gt":
        return "Comparison (x > y)"
    if task == "parity":
        return "Integer parity"
    return task


def _pretty_rep_name(rep: str) -> str:
    if rep.startswith("scalar_"):
        suffix = rep[len("scalar_") :]
        if suffix == "none":
            return "Scalar: raw value"
        if suffix == "standardize":
            return "Scalar: z-score normalized"
        if suffix == "minmax":
            return "Scalar: min-max scaled"
        if suffix == "log":
            return "Scalar: log1p transformed"
        return f"Scalar: {suffix}"

    if rep.startswith("digits_b"):
        # digits_b{base}_f{frac}
        body = rep[len("digits_b") :]
        try:
            base_str, frac_str = body.split("_f")
            base = int(base_str)
            frac = int(frac_str)
        except Exception:
            return f"Digits: {rep}"

        if frac == 0:
            return f"Digits: base {base}, no fractional digits"
        if frac == 1:
            return f"Digits: base {base}, 1 fractional digit"
        return f"Digits: base {base}, {frac} fractional digits"

    return rep


def _pretty_metric_name(metric: str) -> str:
    if metric.lower() == "r2":
        return "R^2 (coefficient of determination)"
    if metric.lower() == "accuracy":
        return "Accuracy"
    return metric


def write_summary(run_dir: str, run_id: str, tracker: StageTracker):
    stages = tracker.all_results()
    out = os.path.join(run_dir, "summary.txt")

    if not stages:
        with open(out, "w") as f:
            f.write("No completed experiments.\n")
        return

    task_names = sorted({v["result"]["task"] for v in stages.values()})
    rep_names = sorted({v["result"]["rep"] for v in stages.values()})

    lines: List[str] = []
    lines.append(f"Numeric Representation Benchmark summary (run_id={run_id})")
    lines.append("")
    lines.append(
        "Each block reports the average performance over seeds for all representations "
        "on a single downstream task. Higher is better for all metrics shown."
    )
    lines.append("")

    for task in task_names:
        pretty_task = _pretty_task_name(task)
        lines.append(f"[Task: {pretty_task}]")
        lines.append(f"{'representation':>36} | {'metric':>10} | {'value':>10} | {'test_loss':>12}")
        lines.append("-" * 78)

        for rep in rep_names:
            keys = [k for k in stages.keys() if k.startswith(f"{task}__{rep}__s")]
            if not keys:
                continue

            vals = []
            metric_name = None
            for k in keys:
                r = stages[k]["result"]
                metric = r.get("primary_metric", 0.0)
                test_loss = r.get("test_loss", 0.0)
                vals.append((metric, test_loss))
                metric_name = r.get("metric_name", "metric")

            if not vals:
                continue

            avg_metric = sum(m for m, _ in vals) / len(vals)
            avg_loss = sum(l for _, l in vals) / len(vals)
            metric_label = _pretty_metric_name(metric_name or "metric")
            rep_label = _pretty_rep_name(rep)

            lines.append(
                f"{rep_label:>36} | {metric_label:>10} | {avg_metric:10.4f} | {avg_loss:12.5f}"
            )

        lines.append("")

    with open(out, "w") as f:
        f.write("\n".join(lines))


def write_plots(run_dir: str, tracker: StageTracker):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    stages = tracker.all_results()
    if not stages:
        return

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    by_task: Dict[str, List[Dict]] = {}
    for entry in stages.values():
        r = entry["result"]
        by_task.setdefault(r["task"], []).append(r)

    # Per-task bar plots: average primary metric per representation
    for task, rows in by_task.items():
        if not rows:
            continue

        rows = sorted(rows, key=lambda r: (r["rep"], r["seed"]))
        rep_to_metrics: Dict[str, List[float]] = {}
        metric_name_raw = rows[0].get("metric_name", "metric")

        for r in rows:
            rep = r["rep"]
            val = float(r.get("primary_metric", 0.0))
            rep_to_metrics.setdefault(rep, []).append(val)

        reps = sorted(rep_to_metrics.keys())
        if not reps:
            continue

        vals = [sum(rep_to_metrics[r]) / len(rep_to_metrics[r]) for r in reps]

        order = sorted(range(len(reps)), key=lambda i: vals[i], reverse=True)
        reps_ord = [reps[i] for i in order]
        vals_ord = [vals[i] for i in order]

        rep_labels = [_pretty_rep_name(r) for r in reps_ord]
        pretty_task = _pretty_task_name(task)
        metric_label = _pretty_metric_name(metric_name_raw)

        h = max(0.35 * len(rep_labels) + 1.8, 3.0)
        w = max(7.0, 0.40 * max(10, len(max(rep_labels, key=len))))

        fig, ax = plt.subplots(figsize=(w, h))
        y = np.arange(len(rep_labels))

        ax.barh(y, vals_ord)
        ax.set_yticks(y)
        ax.set_yticklabels(rep_labels)
        ax.invert_yaxis()

        ax.set_xlabel(metric_label)
        ax.set_title(f"{pretty_task}: downstream performance by numeric representation")

        for i, v in enumerate(vals_ord):
            ax.text(
                v,
                i,
                f"{v:.3f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8,
            )

        caption = (
            f"Metric: {metric_label}. Bars show the average over seeds for this task. "
            "Representations encode the same underlying numbers in different ways "
            "(scalar scaling vs digit-wise encodings)."
        )
        fig.text(0.01, 0.01, caption, fontsize=7, ha="left", va="bottom")

        fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
        fig.savefig(os.path.join(plots_dir, f"{task}_{metric_name_raw}_per_rep.png"), dpi=200)
        plt.close(fig)

    # Matrix summary: tasks x representations heatmap of average primary metric
    all_results = [v["result"] for v in stages.values()]
    if not all_results:
        return

    task_names = sorted({r["task"] for r in all_results})
    rep_names = sorted({r["rep"] for r in all_results})

    metric_by_tr: Dict[tuple, List[float]] = {}
    metric_name_raw = all_results[0].get("metric_name", "metric")

    for r in all_results:
        key = (r["task"], r["rep"])
        metric_by_tr.setdefault(key, []).append(float(r["primary_metric"]))

    if not task_names or not rep_names:
        return

    matrix = np.full((len(task_names), len(rep_names)), np.nan, dtype=float)
    for ti, t in enumerate(task_names):
        for ri, rep in enumerate(rep_names):
            vals = metric_by_tr.get((t, rep))
            if vals:
                matrix[ti, ri] = sum(vals) / len(vals)

    if np.all(np.isnan(matrix)):
        return

    pretty_tasks = [_pretty_task_name(t) for t in task_names]
    pretty_reps = [_pretty_rep_name(r) for r in rep_names]
    metric_label = _pretty_metric_name(metric_name_raw)

    fig_h = max(0.45 * len(pretty_tasks) + 2.8, 3.0)
    fig_w = max(0.40 * len(pretty_reps) + 3.5, 7.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(pretty_reps)))
    ax.set_xticklabels(pretty_reps, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pretty_tasks)))
    ax.set_yticklabels(pretty_tasks)

    for ti in range(len(pretty_tasks)):
        for ri in range(len(pretty_reps)):
            val = matrix[ti, ri]
            if not np.isnan(val):
                ax.text(
                    ri,
                    ti,
                    f"{val:.2f}",
                    va="center",
                    ha="center",
                    fontsize=6,
                )

    ax.set_title("Numeric representation benchmark: average downstream performance")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_label)

    caption = (
        "Rows: downstream tasks. Columns: numeric encodings. "
        "Entries: average primary metric across seeds when training on the medium regime. "
        "This figure summarizes how each encoding transfers across tasks."
    )
    fig.text(0.01, 0.01, caption, fontsize=7, ha="left", va="bottom")

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(os.path.join(plots_dir, "matrix_summary.png"), dpi=220)
    plt.close(fig)
