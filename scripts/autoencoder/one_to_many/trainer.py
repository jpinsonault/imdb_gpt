from pathlib import Path
import sqlite3
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts.autoencoder.print_model import print_model_layers_with_shapes
from scripts.autoencoder.run_logger import build_run_logger
from scripts.autoencoder.training_callbacks import SequenceReconstructionLogger
from scripts.autoencoder.ae_loader import _load_frozen_autoencoders
from scripts.autoencoder.one_to_many.dataset import OneToManyDataset, collate_one_to_many
from scripts.autoencoder.prefetch import CudaPrefetcher
from scripts.autoencoder.sequence_losses import _sequence_loss_and_breakdown, _info_nce_masked_rows
from scripts.autoencoder.timing import _GPUEventTimer
from scripts.precompute_movie_people_seq import build_movie_people_seq
from config import ProjectConfig

class OneToManyPredictor(torch.nn.Module):
    def __init__(self, source_encoder: torch.nn.Module, target_decoder: torch.nn.Module, latent_dim: int, seq_len: int, num_layers: int = 4, num_heads: int = 8, ff_mult: int = 8, dropout: float = 0.1):
        super().__init__()
        import math
        self.source_encoder = source_encoder
        self.target_decoder = target_decoder
        self.latent_dim = int(latent_dim)
        self.seq_len = int(seq_len)
        class _PE(torch.nn.Module):
            def __init__(self, d, n):
                super().__init__()
                pe = torch.zeros(n, d)
                pos = torch.arange(0, n, dtype=torch.float32).unsqueeze(1)
                div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div)
                self.register_buffer("pe", pe)
            def forward(self, length: int, batch_size: int):
                x = self.pe[:length]
                return x.unsqueeze(1).expand(length, batch_size, x.size(-1))
        class _Trunk(torch.nn.Module):
            def __init__(self, d, n, L, H, F, p):
                super().__init__()
                self.n = int(n)
                self.d = int(d)
                self.pos = _PE(d, n)
                self.q = torch.nn.Parameter(torch.zeros(self.n, self.d))
                torch.nn.init.xavier_uniform_(self.q)
                layer = torch.nn.TransformerDecoderLayer(d_model=d, nhead=H, dim_feedforward=F * d, dropout=p, batch_first=False, norm_first=True)
                self.dec = torch.nn.TransformerDecoder(layer, num_layers=L)
                self.post = torch.nn.Sequential(
                    torch.nn.LayerNorm(d),
                    torch.nn.Linear(d, F * d),
                    torch.nn.GELU(),
                    torch.nn.Dropout(p),
                    torch.nn.Linear(F * d, d),
                )
                self.norm = torch.nn.LayerNorm(d)
            def forward(self, z):
                mem = z.unsqueeze(0)
                b = z.size(0)
                tgt = self.q.unsqueeze(1).expand(self.n, b, self.d)
                tgt = tgt + self.pos(self.n, b)
                out = self.dec(tgt, mem)
                out = out.transpose(0, 1)
                out = out + self.post(out)
                out = self.norm(out)
                return out
        self.trunk = _Trunk(latent_dim, seq_len, num_layers, num_heads, ff_mult, dropout)
    def forward(self, source_inputs):
        z = self.source_encoder(source_inputs)
        z_seq = self.trunk(z)
        b = z_seq.size(0)
        flat = z_seq.reshape(b * self.seq_len, self.latent_dim)
        outs = self.target_decoder(flat)
        seq_outs = []
        for y in outs:
            seq_outs.append(y.view(b, self.seq_len, *y.shape[1:]))
        return seq_outs

def _ensure_movie_people_seq(db_path: str, seq_len: int):
    conn = sqlite3.connect(str(Path(db_path)))
    has = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='movie_people_seq'").fetchone()
    conn.close()
    if has is None:
        build_movie_people_seq(str(Path(db_path)), seq_len)

def build_sequence_logger(movie_ae, people_ae, predictor, config: ProjectConfig, db_path: str, seq_len: int):
    return SequenceReconstructionLogger(
        movie_ae=movie_ae,
        people_ae=people_ae,
        predictor=predictor,
        db_path=db_path,
        seq_len=seq_len,
        interval_steps=config.callback_interval,
        num_samples=2,
        table_width=38,
    )

def _build_loader_precomputed(config: ProjectConfig, mov, per, seq_len: int):
    ds = OneToManyDataset(
        db_path=str(Path(config.db_path)),
        source_fields=mov.fields,
        target_fields=per.fields,
        seq_len=seq_len,
        movie_limit=None,
        movie_cache_size=10000,
    )
    num_workers = config.num_workers
    prefetch_factor = config.prefetch_factor
    pin = bool(torch.cuda.is_available())
    loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        collate_fn=collate_one_to_many,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=pin,
        drop_last=False,
    )
    return loader

def train_one_to_many(
    config: ProjectConfig,
    steps: int,
    save_every: int,
    **_
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mov, per = _load_frozen_autoencoders(config)
    latent_dim = config.latent_dim
    seq_len = config.people_sequence_length
    lr = config.learning_rate
    wd = config.weight_decay
    temp = config.latent_temperature
    w_lat = config.latent_loss_weight
    w_rec = config.recon_loss_weight
    use_compile = config.compile_trunk
    _ensure_movie_people_seq(config.db_path, seq_len)
    model = OneToManyPredictor(
        source_encoder=mov.encoder,
        target_decoder=per.decoder,
        latent_dim=latent_dim,
        seq_len=seq_len,
    ).to(device)
    if use_compile and hasattr(torch, "compile"):
        try:
            model.trunk = torch.compile(model.trunk, mode="max-autotune")
        except Exception:
            pass
    
    loader = _build_loader_precomputed(config, mov, per, seq_len)
    mov.encoder.to(device)
    per.decoder.to(device)
    per.encoder.to(device)
    opt = torch.optim.AdamW(model.trunk.parameters(), lr=lr, weight_decay=wd, fused=(device.type == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    run_logger = build_run_logger(config)
    seq_logger = build_sequence_logger(mov, per, model, config, str(Path(config.db_path)), seq_len)
    it_preview = iter(loader)
    xm0, _, _ = next(it_preview)
    xm0 = [x.to(device) for x in xm0]
    print_model_layers_with_shapes(model, xm0)
    timer = _GPUEventTimer(print_every=config.log_interval)
    step = 0
    prefetch = CudaPrefetcher(loader, device)
    model.train()
    infinite = steps is None or steps <= 0
    total_for_bar = None if infinite else steps
    try:
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
                    run_logger.add_scalars(
                        float(loss.detach().cpu().item()),
                        float(rec_loss.detach().cpu().item()),
                        float(nce_loss.detach().cpu().item()),
                        out["total"] / 1000.0,
                        opt,
                    )
                    run_logger.add_field_losses("loss/sequence_target", field_breakdown)
                    run_logger.tick(
                        float(loss.detach().cpu().item()),
                        float(rec_loss.detach().cpu().item()),
                        float(nce_loss.detach().cpu().item()),
                    )
                seq_logger.on_batch_end(step)
                if save_every and step % save_every == 0:
                    out_dir = Path(config.model_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), out_dir / f"OneToManyPredictor_step_{step}.pt")
    finally:
        out_dir = Path(config.model_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "OneToManyPredictor_final.pt")
        run_logger.close()
    return model
