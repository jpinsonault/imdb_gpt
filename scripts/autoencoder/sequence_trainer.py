# scripts/autoencoder/sequence_trainer.py
from pathlib import Path
import sqlite3
from typing import Any, Dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts.autoencoder.print_model import print_model_layers_with_shapes
from scripts.autoencoder.run_logger import build_run_logger
from scripts.autoencoder.training_callbacks import SequenceReconstructionLogger
from scripts.autoencoder.ae_loader import _load_frozen_autoencoders
from scripts.autoencoder.imdb_sequence_decoders import MovieToPeopleSequencePredictor
from scripts.autoencoder.sequence_datasets import MoviesPeopleSequenceDataset, _collate
from scripts.autoencoder.prefetch import CudaPrefetcher
from scripts.autoencoder.sequence_losses import _sequence_loss_and_breakdown, _info_nce_masked_rows
from scripts.precompute_movie_people_seq import build_movie_people_seq
from config import ProjectConfig


def disable_inductor_autotune():
    from torch._inductor import config as _inductor_cfg
    _inductor_cfg.max_autotune = False

def train_sequence_predictor(
    config: ProjectConfig,
    steps: int,
    save_every: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mov, per = _load_frozen_autoencoders(config)

    latent_dim = config.latent_dim
    seq_len = config.people_sequence_length
    batch_size = config.batch_size
    lr = config.learning_rate
    wd = config.weight_decay
    temp = config.latent_temperature
    w_lat = config.latent_loss_weight
    w_rec = config.recon_loss_weight
    log_every = config.log_interval
    use_compile = config.compile_trunk
    use_cuda_graphs = config.use_cuda_graphs

    disable_inductor_autotune()
    _conn = sqlite3.connect(str(Path(config.db_path)))
    _has = _conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='movie_people_seq'").fetchone()
    _conn.close()
    if _has is None:
        build_movie_people_seq(str(Path(config.db_path)), seq_len)

    model = MovieToPeopleSequencePredictor(
        movie_encoder=mov.encoder,
        people_decoder=per.decoder,
        latent_dim=latent_dim,
        seq_len=seq_len,
    ).to(device)

    if use_compile and hasattr(torch, "compile"):
        try:
            model.trunk = torch.compile(model.trunk, mode="max-autotune")
        except Exception:
            pass

    ds = MoviesPeopleSequenceDataset(
        db_path=str(Path(config.db_path)),
        movie_fields=mov.fields,
        people_fields=per.fields,
        seq_len=seq_len,
    )

    num_workers = config.num_workers
    prefetch_factor = config.prefetch_factor
    pin = bool(torch.cuda.is_available())

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=_collate,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=pin,
        drop_last=False,
    )

    mov.encoder.to(device)
    per.decoder.to(device)
    per.encoder.to(device)

    opt = torch.optim.AdamW(model.trunk.parameters(), lr=lr, weight_decay=wd, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    run_logger = build_run_logger(config)
    seq_logger = SequenceReconstructionLogger(
        movie_ae=mov,
        people_ae=per,
        predictor=model,
        db_path=str(Path(config.db_path)),
        seq_len=seq_len,
        interval_steps=config.callback_interval,
        num_samples=2,
        table_width=38,
    )

    it_preview = iter(loader)
    xm0, _, _ = next(it_preview)
    xm0 = [x.to(device) for x in xm0]
    print_model_layers_with_shapes(model, xm0)

    from scripts.autoencoder.timing import _GPUEventTimer
    timer = _GPUEventTimer(print_every=log_every)

    step = 0
    prefetch = CudaPrefetcher(loader, device)
    model.train()

    with tqdm(total=steps, desc="sequence", dynamic_ncols=True, miniters=50) as pbar:
        while step < steps:
            with timer.cpu_range("data"):
                batch = prefetch.next()
                if batch is None:
                    prefetch = CudaPrefetcher(loader, device)
                    batch = prefetch.next()
                    if batch is None:
                        break
                xm, yp, m = batch

            timer.start_step()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                with timer.gpu_range("mov_enc"):
                    z_m = mov.encoder(xm)
                with timer.gpu_range("trunk"):
                    z_seq = model.trunk(z_m)
                b = z_seq.size(0)
                flat = z_seq.reshape(b * seq_len, latent_dim)
                with timer.gpu_range("ppl_dec"):
                    outs = per.decoder(flat)
                preds = [y.view(b, seq_len, *y.shape[1:]) for y in outs]
                with timer.gpu_range("rec"):
                    rec_loss, field_breakdown = _sequence_loss_and_breakdown(per.fields, preds, yp, m)
                with timer.gpu_range("tgt_enc"):
                    with torch.no_grad():
                        flat_targets = [y.view(b * seq_len, *y.shape[2:]) for y in yp]
                        z_tgt_flat = per.encoder(flat_targets)
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
                timer.print_line(keys, out, step)
                run_logger.add_scalars(
                    float(loss.detach().cpu().item()),
                    float(rec_loss.detach().cpu().item()),
                    float(nce_loss.detach().cpu().item()),
                    out["total"] / 1000.0,
                    opt,
                )
                run_logger.add_field_losses("loss/sequence_people", field_breakdown)
                run_logger.tick(
                    float(loss.detach().cpu().item()),
                    float(rec_loss.detach().cpu().item()),
                    float(nce_loss.detach().cpu().item()),
                )

            seq_logger.on_batch_end(step)

            if save_every and step % save_every == 0:
                out = Path(config.model_dir)
                out.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out / f"MovieToPeopleSequencePredictor_step_{step}.pt")

    out = Path(config.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "MovieToPeopleSequencePredictor_final.pt")
    run_logger.close()
