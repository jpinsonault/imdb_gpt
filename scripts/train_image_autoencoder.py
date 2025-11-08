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

    latent_reg_weight = 0.0

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

                rec_loss = F.l1_loss(recon, batch)

                if latent_reg_weight > 0.0:
                    with torch.no_grad():
                        z = model.encode(batch)
                    z_loss = (z.pow(2).mean())
                    loss = rec_loss + latent_reg_weight * z_loss
                else:
                    loss = rec_loss

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

        ckpt_path = os.path.join(run_dir, f"model_epoch_{epoch:04d}.pt")
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
