# scripts/autoencoder/many_to_many/trainer.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts.autoencoder.ae_loader import _load_frozen_autoencoders
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from scripts.autoencoder.many_to_many.dataset import ManyToManyDataset, collate_many_to_many
from scripts.autoencoder.many_to_many.model import ManyToManyModel
from scripts.autoencoder.sequence_losses import _sequence_loss_and_breakdown, _info_nce_masked_rows
from scripts.autoencoder.prefetch import CudaPrefetcher
from scripts.autoencoder.print_model import print_model_layers_with_shapes
from scripts.autoencoder.run_logger import build_run_logger
from config import ProjectConfig

def _build_models(config: ProjectConfig, warm_start: bool) -> Tuple[TitlesAutoencoder, PeopleAutoencoder]:
    if warm_start:
        mov, per = _load_frozen_autoencoders(config)
        return mov, per
    mov = TitlesAutoencoder(config)
    per = PeopleAutoencoder(config)
    mov.accumulate_stats(); mov.finalize_stats(); mov.build_autoencoder()
    per.accumulate_stats(); per.finalize_stats(); per.build_autoencoder()
    for p in mov.encoder.parameters():
        p.requires_grad = False
    for p in mov.decoder.parameters():
        p.requires_grad = False
    for p in per.encoder.parameters():
        p.requires_grad = False
    for p in per.decoder.parameters():
        p.requires_grad = False
    mov.encoder.eval(); mov.decoder.eval()
    per.encoder.eval(); per.decoder.eval()
    return mov, per

def _build_loader(config: ProjectConfig, mov: TitlesAutoencoder, per: PeopleAutoencoder, seq_len_titles: int, seq_len_people: int):
    ds = ManyToManyDataset(
        db_path=str(Path(config.db_path)),
        movie_fields=mov.fields,
        people_fields=per.fields,
        seq_len_titles=seq_len_titles,
        seq_len_people=seq_len_people,
        movie_limit=None,
        person_limit=None,
        mode_ratio=None,
    )
    num_workers = int(getattr(config, "num_workers", 0) or 0)
    cfg_pf = int(getattr(config, "prefetch_factor", 2) or 0)
    prefetch_factor = None if num_workers == 0 else max(1, cfg_pf)
    pin = bool(torch.cuda.is_available())
    loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        collate_fn=collate_many_to_many,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=pin,
        drop_last=False,
    )
    return loader

def _masked_seq_loss(fields, preds, tgts, mask):
    return _sequence_loss_and_breakdown(fields, preds, tgts, mask)

def train_many_to_many(
    config: ProjectConfig,
    steps: int | None = None,
    save_every: int = 0,
    warm_start: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mov, per = _build_models(config, warm_start=warm_start)
    seq_len_titles = int(getattr(config, "titles_sequence_length", 1) or 1)
    seq_len_people = int(getattr(config, "people_sequence_length", 8) or 8)
    latent_dim = int(config.latent_dim)
    model = ManyToManyModel(
        movie_encoder=mov.encoder,
        people_encoder=per.encoder,
        movie_decoder=mov.decoder,
        people_decoder=per.decoder,
        latent_dim=latent_dim,
        seq_len_titles=seq_len_titles,
        seq_len_people=seq_len_people,
    ).to(device)
    loader = _build_loader(config, mov, per, seq_len_titles, seq_len_people)
    mov.encoder.to(device); mov.decoder.to(device)
    per.encoder.to(device); per.decoder.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, fused=(device.type == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    run_logger = build_run_logger(config)

    it_preview = iter(loader)
    xm0, xp0, yt0, yp0, mt0, mp0, mode0 = next(it_preview)
    xm0 = [x.to(device) for x in xm0]
    xp0 = [x.to(device) for x in xp0]
    _ = print_model_layers_with_shapes(model, xm0)

    prefetch = CudaPrefetcher(loader, device)
    model.train()
    step = 0
    infinite = steps is None or steps <= 0
    total_for_bar = None if infinite else steps

    temp = float(getattr(config, "latent_temperature", 0.07) or 0.07)
    w_lat = float(getattr(config, "latent_loss_weight", 1.0) or 1.0)
    w_rec = float(getattr(config, "recon_loss_weight", 1.0) or 1.0)

    try:
        with tqdm(total=total_for_bar, desc="many_to_many", dynamic_ncols=True, miniters=50) as pbar:
            while infinite or step < steps:
                batch = prefetch.next()
                if batch is None:
                    prefetch = CudaPrefetcher(loader, device)
                    batch = prefetch.next()
                    if batch is None:
                        if infinite:
                            continue
                        break
                xm, xp, yt, yp, mt, mp, mode = batch
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    preds_titles_seq, preds_people_seq, z_m2p, z_p2m = model(xm, xp)

                    rec_titles, titles_break = _masked_seq_loss(mov.fields, preds_titles_seq, yt, mt)
                    rec_people, people_break = _masked_seq_loss(per.fields, preds_people_seq, yp, mp)
                    rec_loss = rec_titles + rec_people

                    with torch.no_grad():
                        flat_yt = [y.view(y.size(0) * y.size(1), *y.shape[2:]) for y in yt]
                        flat_yp = [y.view(y.size(0) * y.size(1), *y.shape[2:]) for y in yp]
                        z_titles_target = mov.encoder(flat_yt).view(y.size(0), -1, latent_dim)
                        z_people_target = per.encoder(flat_yp).view(y.size(0), -1, latent_dim)

                    nce_m2p = _info_nce_masked_rows(z_m2p, z_people_target, mp, temp)
                    nce_p2m = _info_nce_masked_rows(z_p2m, z_titles_target, mt, temp)
                    nce_loss = 0.5 * (nce_m2p + nce_p2m)

                    loss = w_rec * rec_loss + w_lat * nce_loss

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                step += 1
                pbar.update(1)
                run_logger.add_scalars(
                    float(loss.detach().cpu().item()),
                    float(rec_loss.detach().cpu().item()),
                    float(nce_loss.detach().cpu().item()),
                    0.0,
                    opt,
                )
                run_logger.add_field_losses("loss/titles_seq", titles_break)
                run_logger.add_field_losses("loss/people_seq", people_break)
                run_logger.tick(
                    float(loss.detach().cpu().item()),
                    float(rec_loss.detach().cpu().item()),
                    float(nce_loss.detach().cpu().item()),
                )

                if save_every and step % save_every == 0:
                    out_dir = Path(config.model_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), out_dir / f"ManyToManyModel_step_{step}.pt")
    finally:
        out_dir = Path(config.model_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "ManyToManyModel_final.pt")
        run_logger.close()

    return model
