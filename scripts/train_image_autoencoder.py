import os
import json
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import project_config, ensure_dirs
from scripts.image_encoding.dataset import ImageFolderDataset
from scripts.image_encoding.autoencoder import ConvAutoencoder
from scripts.image_encoding.callbacks import ReconstructionSaver


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_tensorboard_writer(cfg, run_dir):
    try:
        writer = SummaryWriter(log_dir=run_dir)
    except Exception:
        return None

    try:
        if hasattr(cfg, "__dataclass_fields__"):
            cfg_dict = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__.keys()}
        else:
            cfg_dict = dict(vars(cfg))
        text = json.dumps(cfg_dict, indent=2, sort_keys=True)
        writer.add_text("config/json", f"<pre>{text}</pre>", 0)
    except Exception:
        pass

    return writer


def gradient_loss(pred, target):
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    dx_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_x = F.l1_loss(dx_pred, dx_tgt)
    loss_y = F.l1_loss(dy_pred, dy_tgt)
    return loss_x + loss_y


def laplacian_loss(pred, target):
    k = torch.tensor(
        [[0.0, -1.0, 0.0],
         [-1.0, 4.0, -1.0],
         [0.0, -1.0, 0.0]],
        dtype=pred.dtype,
        device=pred.device,
    ).view(1, 1, 3, 3)
    c = pred.size(1)
    weight = k.expand(c, 1, 3, 3)
    pred_lap = F.conv2d(pred, weight, padding=1, groups=c)
    tgt_lap = F.conv2d(target, weight, padding=1, groups=c)
    return F.l1_loss(pred_lap, tgt_lap)


def total_variation(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def main():
    cfg = project_config
    ensure_dirs(cfg)

    device = get_device()

    dataset = ImageFolderDataset(
        root=cfg.image_ae_data_dir,
        image_size=(cfg.image_ae_image_size, cfg.image_ae_image_size),
    )

    if len(dataset) == 0:
        raise RuntimeError("image_ae_data_dir is empty or has no valid images")

    batch_size = max(1, int(cfg.image_ae_batch_size))
    train_drop_last = len(dataset) >= batch_size

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(getattr(cfg, "num_workers", 0) or 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=train_drop_last,
    )

    sample_batch_size = min(int(cfg.image_ae_max_recon_samples), len(dataset))
    sample_loader = DataLoader(
        dataset,
        batch_size=sample_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    sample_batch = next(iter(sample_loader))

    model = ConvAutoencoder(
        in_channels=cfg.image_ae_in_channels,
        base_channels=cfg.image_ae_base_channels,
        latent_dim=cfg.image_ae_latent_dim,
        image_size=cfg.image_ae_image_size,
    ).to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.image_ae_runs_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    writer = build_tensorboard_writer(cfg, run_dir)

    print(f"device: {device}")
    print(f"dataset root: {cfg.image_ae_data_dir}")
    print(f"dataset size: {len(dataset)} images")
    print(f"batch size: {batch_size}")
    print(f"steps per epoch: {len(loader)}")
    print(f"run dir: {run_dir}")
    print("model:")
    print(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.image_ae_learning_rate,
        weight_decay=0.0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    recon_saver = ReconstructionSaver(
        output_dir=run_dir,
        sample_batch=sample_batch,
        every_n_epochs=cfg.image_ae_recon_every,
        max_samples=cfg.image_ae_max_recon_samples,
    )

    epochs = int(cfg.image_ae_epochs)
    global_step = 0

    w_l1 = float(getattr(cfg, "image_ae_loss_w_l1", 1.0))
    w_grad = float(getattr(cfg, "image_ae_loss_w_grad", 0.0))
    w_tv = float(getattr(cfg, "image_ae_loss_w_tv", 0.0))
    w_lap = float(getattr(cfg, "image_ae_loss_w_laplace", 0.0))
    latent_reg_weight = float(getattr(cfg, "image_ae_latent_reg_weight", 0.0))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        pbar = tqdm(
            loader,
            desc=f"epoch {epoch}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )

        for batch in pbar:
            batch = batch.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(batch)
                recon = torch.sigmoid(logits)

                loss_l1 = F.l1_loss(recon, batch)

                if w_grad != 0.0:
                    loss_grad = gradient_loss(recon, batch)
                else:
                    loss_grad = recon.new_zeros(())

                if w_lap != 0.0:
                    loss_lap = laplacian_loss(recon, batch)
                else:
                    loss_lap = recon.new_zeros(())

                if w_tv != 0.0:
                    loss_tv = total_variation(recon)
                else:
                    loss_tv = recon.new_zeros(())

                loss = (
                    w_l1 * loss_l1
                    + w_grad * loss_grad
                    + w_lap * loss_lap
                    + w_tv * loss_tv
                )

                if latent_reg_weight > 0.0:
                    z = model.encode(batch)
                    loss = loss + latent_reg_weight * z.pow(2).mean()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_val = float(loss.detach().cpu().item())
            total_loss += loss_val
            total_batches += 1
            global_step += 1

            pbar.set_postfix(loss=f"{loss_val:.6f}")

            if writer is not None:
                writer.add_scalar("train/loss_step", loss_val, global_step)

        if total_batches == 0:
            raise RuntimeError(
                "No batches produced by DataLoader. "
                "Check image_ae_batch_size vs number of images."
            )

        avg_loss = total_loss / total_batches
        print(f"epoch {epoch} avg_loss {avg_loss:.6f}")

        if writer is not None:
            writer.add_scalar("train/loss_epoch", avg_loss, epoch)

        recon_saver.maybe_save(epoch, model, device)

        ckpt_path = os.path.join(run_dir, f"model_epoch.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg.__dict__,
            },
            ckpt_path,
        )

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
