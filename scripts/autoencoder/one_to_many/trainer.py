# scripts/autoencoder/one_to_many/trainer.py
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from config import ProjectConfig
from scripts.autoencoder.one_to_many.dataset import OneToManyDataset, collate_one_to_many
from scripts.autoencoder.one_to_many.model import OneToManyPredictor
from scripts.autoencoder.sequence_losses import _sequence_loss_and_breakdown, _info_nce_masked_rows
from scripts.autoencoder.prefetch import CudaPrefetcher
from scripts.autoencoder.print_model import print_model_layers_with_shapes
from scripts.autoencoder.run_logger import build_run_logger
from scripts.autoencoder.timing import _GPUEventTimer

def disable_inductor_autotune():
    from torch._inductor import config as _inductor_cfg
    _inductor_cfg.max_autotune = False

def build_loader(
    provider,
    source_fields,
    target_fields,
    config: ProjectConfig,
) -> DataLoader:
    ds = OneToManyDataset(
        provider=provider,
        source_fields=source_fields,
        target_fields=target_fields,
    )
    pin = bool(torch.cuda.is_available())
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        collate_fn=collate_one_to_many,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else 2,
        persistent_workers=True if config.num_workers > 0 else False,
        pin_memory=pin,
        drop_last=False,
    )

def train_one_to_many(
    config: ProjectConfig,
    provider,
    source_ae,
    target_ae,
    steps: int,
    save_every: int,
    seq_logger=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_dim = config.latent_dim
    seq_len = provider.seq_len
    lr = config.learning_rate
    wd = config.weight_decay
    temp = config.latent_temperature
    w_lat = config.latent_loss_weight
    w_rec = config.recon_loss_weight
    log_every = config.log_interval
    use_compile = config.compile_trunk

    disable_inductor_autotune()

    model = OneToManyPredictor(
        source_encoder=source_ae.encoder,
        target_decoder=target_ae.decoder,
        latent_dim=latent_dim,
        seq_len=seq_len,
    ).to(device)

    if use_compile and hasattr(torch, "compile"):
        try:
            model.trunk = torch.compile(model.trunk, mode="max-autotune")
        except Exception:
            pass

    loader = build_loader(
        provider=provider,
        source_fields=source_ae.fields,
        target_fields=target_ae.fields,
        config=config,
    )

    source_ae.encoder.to(device)
    target_ae.decoder.to(device)
    target_ae.encoder.to(device)

    opt = torch.optim.AdamW(model.trunk.parameters(), lr=lr, weight_decay=wd, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    run_logger = build_run_logger(config)

    it_preview = iter(loader)
    xm0, _, _ = next(it_preview)
    xm0 = [x.to(device) for x in xm0]
    print_model_layers_with_shapes(model, xm0)

    timer = _GPUEventTimer(print_every=log_every)

    step = 0
    prefetch = CudaPrefetcher(loader, device)
    model.train()

    infinite = steps is None or steps <= 0
    total_for_bar = None if infinite else steps

    try:
        from tqdm import tqdm
        with tqdm(total=total_for_bar, desc="one_to_many", dynamic_ncols=True, miniters=50) as pbar:
            while infinite or step < steps:
                with timer.cpu_range("data"):
                    batch = prefetch.next()
                    if batch is None:
                        prefetch = CudaPrefetcher(loader, device)
                        batch = prefetch.next()
                        if batch is None:
                            if infinite:
                                continue
                            break
                    xs, yt, m = batch

                timer.start_step()
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    with timer.gpu_range("mov_enc"):
                        z_src = source_ae.encoder(xs)
                    with timer.gpu_range("trunk"):
                        z_seq = model.trunk(z_src)
                    b = z_seq.size(0)
                    flat = z_seq.reshape(b * seq_len, latent_dim)
                    with timer.gpu_range("ppl_dec"):
                        outs = target_ae.decoder(flat)
                    preds = [y.view(b, seq_len, *y.shape[1:]) for y in outs]
                    with timer.gpu_range("rec"):
                        rec_loss, field_breakdown = _sequence_loss_and_breakdown(target_ae.fields, preds, yt, m)
                    with timer.gpu_range("tgt_enc"):
                        with torch.no_grad():
                            flat_targets = [y.view(b * seq_len, *y.shape[2:]) for y in yt]
                            z_tgt_flat = target_ae.encoder(flat_targets)
                            z_tgt_seq = z_tgt_flat.view(b, seq_len, latent_dim)
                        nce_loss = _info_nce_masked_rows(z_seq, z_tgt_seq, m, temp)
                    loss = w_lat * nce_loss + w_rec * rec_loss

                with timer.gpu_range("backward"):
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                with timer.gpu_range("opt"):
                    scaler.step(opt)
                    scaler.update()

                step += 1
                pbar.update(1)

                vals = timer.end_step_and_accumulate()
                if vals is not None:
                    keys, out = vals
                    run_logger.add_scalars(
                        float(loss.detach().cpu().item()),
                        float(rec_loss.detach().cpu().item()),
                        float(nce_loss.detach().cpu().item()),
                        out["total"] / 1000.0,
                        opt,
                    )
                    run_logger.add_field_losses("loss/sequence_target", field_breakdown)
                    run_logger.tick()

                if seq_logger is not None:
                    seq_logger.on_batch_end(step)

                if save_every and step % save_every == 0:
                    out = Path(config.model_dir)
                    out.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), out / f"OneToManyPredictor_step_{step}.pt")
    finally:
        out = Path(config.model_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "OneToManyPredictor_final.pt")
        run_logger.close()

    return model
