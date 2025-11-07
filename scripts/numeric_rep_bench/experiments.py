import os
import time
import json
import logging
from dataclasses import asdict
from typing import Iterable, List, Tuple, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from scripts.numeric_rep_bench.tasks import REGIME_SAMPLERS, REGIMES_ORDERED
from .config import TaskSpec, RepConfig, TrainConfig, get_tasks, get_reps
from .modeling import NumericRepModel
from .tracking import ExperimentResult, StageTracker
from .reporting import write_summary, write_plots


def create_logger(run_dir: str) -> logging.Logger:
    logger = logging.getLogger(f"numeric_rep_bench.{os.path.basename(run_dir)}")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)7s | %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    fh = logging.FileHandler(os.path.join(run_dir, "run.log"), mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


def sample_regime(task: TaskSpec, regime: str, n: int, device: torch.device):
    if task.name == "parity":
        x = task.sampler(n, device)
        y = task.target_fn(x)
        return x.to(device), y.to(device)

    if regime not in REGIME_SAMPLERS:
        raise ValueError(f"Unknown regime: {regime}")

    sampler = REGIME_SAMPLERS[regime]
    x = sampler(task.arity, n).to(device)
    y = task.target_fn(x)
    return x.to(device), y.to(device)


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool):
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def compute_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    total = 0.0
    n = 0
    model.eval()
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            out = model(bx)
            loss = criterion(out, by)
            bs = bx.size(0)
            total += float(loss.item()) * bs
            n += bs
    return total / max(n, 1)


def compute_metric(model: nn.Module, loader: DataLoader, task: TaskSpec, device: torch.device):
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            out = model(bx)
            ys.append(by.cpu())
            ps.append(out.cpu())
    y = torch.cat(ys, dim=0)
    p = torch.cat(ps, dim=0)

    if task.kind == "regression":
        y_mean = y.mean()
        ss_tot = ((y - y_mean) ** 2).sum()
        ss_res = ((y - p) ** 2).sum()
        r2 = 1.0 - ss_res / (ss_tot + 1e-9)
        return float(r2.item()), "r2"

    if p.size(-1) == 1:
        pred = (p.view(-1) > 0).long()
        y_flat = y.view(-1).long()
    else:
        pred = p.argmax(dim=-1)
        y_flat = y.view(-1).long()
    acc = (pred == y_flat).float().mean()
    return float(acc.item()), "accuracy"


def write_samples(
    run_dir: str,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    task: TaskSpec,
    rep: RepConfig,
    seed: int,
    train_regime: str,
    run_id: str,
    device: torch.device,
) -> str:
    model.eval()
    with torch.no_grad():
        preds = model(x.to(device)).cpu()

    if task.kind == "classification":
        if preds.size(-1) == 2:
            probs1 = preds.softmax(dim=-1)[:, 1]
            pred_cls = preds.argmax(dim=-1)
        else:
            probs1 = torch.sigmoid(preds.view(-1))
            pred_cls = (probs1 > 0.5).long()
    else:
        probs1 = None
        pred_cls = None

    head_cols = [f"x{i}" for i in range(task.arity)] + ["target", "prediction"]
    if task.kind == "classification":
        head_cols += ["pred_class", "p(class=1)"]

    header = " | ".join(f"{c:>12}" for c in head_cols)
    lines = [header, "-" * len(header)]

    for i in range(x.size(0)):
        row = []
        for j in range(task.arity):
            row.append(f"{float(x[i, j]):12.4f}")

        tgt_val = y[i].item()
        if isinstance(tgt_val, (int, bool)):
            row.append(f"{int(tgt_val):12d}")
        else:
            row.append(f"{float(tgt_val):12.4f}")

        pred_val = preds[i]
        if task.kind == "regression":
            row.append(f"{float(pred_val.squeeze()):12.4f}")
        else:
            if pred_val.numel() == 1:
                row.append(f"{float(pred_val.squeeze()):12.4f}")
            else:
                row.append(f"{float(pred_val[1]):12.4f}")

        if task.kind == "classification":
            row.append(f"{int(pred_cls[i]):12d}")
            row.append(f"{float(probs1[i]):12.4f}")

        lines.append(" | ".join(row))

    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    path = os.path.join(samples_dir, f"samples_{task.name}__{rep.name}__s{seed}.txt")

    with open(path, "w") as f:
        f.write(
            f"Task: {task.name} ({task.display_name})\n"
            f"Representation: {rep.name}\n"
            f"Seed: {seed}\n"
            f"Train regime: {train_regime}\n"
            f"Run id: {run_id}\n\n"
        )
        f.write("\n".join(lines))
    return path


def train_and_eval_single(
    run_id: str,
    run_dir: str,
    task: TaskSpec,
    rep: RepConfig,
    seed: int,
    cfg: TrainConfig,
    device: torch.device,
    train_regime: str,
    eval_regimes: Iterable[str],
    logger: logging.Logger,
) -> ExperimentResult:
    name = f"{task.name}__{rep.name}__s{seed}"
    torch.manual_seed(seed)

    train_x, train_y = sample_regime(task, train_regime, cfg.train_size, device)
    val_x, val_y = sample_regime(task, train_regime, cfg.val_size, device)
    test_x_main, test_y_main = sample_regime(task, train_regime, cfg.test_size, device)

    if task.kind == "classification":
        train_y = train_y.long()
        val_y = val_y.long()
        test_y_main = test_y_main.long()

    model = NumericRepModel(task, rep, rep_dim=cfg.rep_dim, device=device)
    model.init_from_train(train_x)
    model.to(device)

    criterion = nn.MSELoss() if task.kind == "regression" else nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = make_loader(train_x, train_y, cfg.batch_size, True)
    val_loader = make_loader(val_x, val_y, cfg.batch_size, False)
    test_loader_main = make_loader(test_x_main, test_y_main, cfg.batch_size, False)

    best_val = float("inf")
    best_state = None

    epoch_bar = tqdm(
        range(cfg.epochs),
        desc=name,
        leave=False,
        dynamic_ncols=True,
        unit="epoch",
    )
    for _ in epoch_bar:
        model.train()
        total = 0.0
        n = 0
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            opt.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            opt.step()
            bs = bx.size(0)
            total += float(loss.item()) * bs
            n += bs
        train_loss_epoch = total / max(n, 1)

        val_loss = compute_loss(model, val_loader, criterion, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        epoch_bar.set_postfix(
            train=f"{train_loss_epoch:.4f}",
            val=f"{val_loss:.4f}",
            best=f"{best_val:.4f}",
        )
    epoch_bar.close()

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    train_loss = compute_loss(model, train_loader, criterion, device)
    val_loss = compute_loss(model, val_loader, criterion, device)
    test_loss_main = compute_loss(model, test_loader_main, criterion, device)
    primary_metric_main, metric_name = compute_metric(model, test_loader_main, task, device)

    metrics_by_regime: Dict[str, float] = {}
    for regime in eval_regimes:
        x_te, y_te = sample_regime(task, regime, cfg.test_size, device)
        if task.kind == "classification":
            y_te = y_te.long()
        loader = make_loader(x_te, y_te, cfg.batch_size, False)
        m_val, _ = compute_metric(model, loader, task, device)
        metrics_by_regime[regime] = float(m_val)

    primary_metric = float(metrics_by_regime.get(train_regime, primary_metric_main))

    samples_path = write_samples(
        run_dir=run_dir,
        model=model,
        x=test_x_main[:32],
        y=test_y_main[:32],
        task=task,
        rep=rep,
        seed=seed,
        train_regime=train_regime,
        run_id=run_id,
        device=device,
    )

    logger.info(
        f"result {name}: train={train_loss:.5f} val={val_loss:.5f} "
        f"test={test_loss_main:.5f} {metric_name}@{train_regime}={primary_metric:.4f}"
    )

    return ExperimentResult(
        task=task.name,
        rep=rep.name,
        run_id=run_id,
        seed=seed,
        train_regime=train_regime,
        metric_name=metric_name,
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss_main,
        primary_metric=primary_metric,
        primary_metric_regime=train_regime,
        metrics_by_regime=metrics_by_regime,
        samples_path=samples_path,
    )


def run_all_experiments(
    run_id: str,
    full: bool,
    only: List[str],
    resume: bool,
    device_str: str,
    cfg: TrainConfig,
    root_dir: str = "runs/numeric_rep_bench",
):
    device = torch.device(device_str)
    os.makedirs(root_dir, exist_ok=True)
    run_dir = os.path.join(root_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    meta = {
        "run_id": run_id,
        "full": bool(full),
        "only": list(only or []),
        "resume": bool(resume),
        "device": device_str,
        "train_config": asdict(cfg),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger = create_logger(run_dir)
    tracker = StageTracker(run_dir)

    tasks = get_tasks(full)
    reps = get_reps(full)
    only_set = set(only or [])

    train_regime = "medium"
    eval_regimes = list(REGIMES_ORDERED)

    logger.info(
        f"run_id={run_id} device={device} full={full} resume={resume} "
        f"train_regime={train_regime} eval_regimes={','.join(eval_regimes)} "
        f"n_tasks={len(tasks)} n_reps={len(reps)} n_seeds={cfg.n_seeds}"
    )

    base_seed = int(cfg.seed)
    combos: List[Tuple[TaskSpec, RepConfig, int]] = []

    for task in tasks:
        for rep in reps:
            for s_idx in range(cfg.n_seeds):
                seed = base_seed + s_idx
                name = f"{task.name}__{rep.name}__s{seed}"

                if only_set and not (
                    task.name in only_set
                    or rep.name in only_set
                    or name in only_set
                ):
                    continue

                if resume and tracker.is_done(task.name, rep.name, seed):
                    continue

                combos.append((task, rep, seed))

    if not combos:
        logger.info("No experiments to run (all done or filtered); writing reports.")
        write_summary(run_dir, run_id, tracker)
        write_plots(run_dir, tracker)
        return

    bar = tqdm(combos, desc="experiments", dynamic_ncols=True, unit="exp")
    for task, rep, seed in bar:
        name = f"{task.name}__{rep.name}__s{seed}"
        bar.set_postfix_str(name)
        logger.info(f"[start] {name} ({task.display_name} / {rep.name} / seed={seed})")

        result = train_and_eval_single(
            run_id=run_id,
            run_dir=run_dir,
            task=task,
            rep=rep,
            seed=seed,
            cfg=cfg,
            device=device,
            train_regime=train_regime,
            eval_regimes=eval_regimes,
            logger=logger,
        )
        tracker.mark_done(result)

        logger.info(
            f"[done] {name} primary={result.metric_name}@{result.primary_metric_regime} "
            f"value={result.primary_metric:.4f} test_loss={result.test_loss:.5f}"
        )

    logger.info("All experiments complete; writing reports.")
    write_summary(run_dir, run_id, tracker)
    write_plots(run_dir, tracker)
    logger.info("Reports written.")
