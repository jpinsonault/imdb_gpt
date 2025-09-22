import argparse
from pathlib import Path

from scripts.analysis.performance_summary import PerformanceSummary
import torch

from config import ProjectConfig
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder


def _load_frozen_autoencoders(model_dir: Path, db_path: Path) -> tuple[TitlesAutoencoder, PeopleAutoencoder]:
    cfg = ProjectConfig()
    cfg.db_path = str(db_path)
    cfg.model_dir = str(model_dir)
    cfg.use_cache = True
    cfg.refresh_cache = False
    mov = TitlesAutoencoder(cfg)
    per = PeopleAutoencoder(cfg)
    mov.accumulate_stats()
    mov.finalize_stats()
    per.accumulate_stats()
    per.finalize_stats()
    mov.build_autoencoder()
    per.build_autoencoder()
    me = Path(model_dir) / "TitlesAutoencoder_encoder.pt"
    md = Path(model_dir) / "TitlesAutoencoder_decoder.pt"
    pe = Path(model_dir) / "PeopleAutoencoder_encoder.pt"
    pd = Path(model_dir) / "PeopleAutoencoder_decoder.pt"
    mov.encoder.load_state_dict(torch.load(me, map_location="cpu"))
    mov.decoder.load_state_dict(torch.load(md, map_location="cpu"))
    per.encoder.load_state_dict(torch.load(pe, map_location="cpu"))
    per.decoder.load_state_dict(torch.load(pd, map_location="cpu"))
    for p in mov.encoder.parameters():
        p.requires_grad = False
    for p in mov.decoder.parameters():
        p.requires_grad = False
    for p in per.encoder.parameters():
        p.requires_grad = False
    for p in per.decoder.parameters():
        p.requires_grad = False
    mov.encoder.eval()
    mov.decoder.eval()
    per.encoder.eval()
    per.decoder.eval()
    return mov, per


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--table-width", type=int, default=38)
    parser.add_argument("--similarity-out", type=str, default=None)
    parser.add_argument("--similarity-max-items", type=int, default=50000)
    parser.add_argument("--plot-dir", type=str, default="plots")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    db_path = Path(args.db)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mov, per = _load_frozen_autoencoders(model_dir, db_path)
    mov.encoder.to(device)
    mov.decoder.to(device)
    per.encoder.to(device)
    per.decoder.to(device)
    mov.device = device
    per.device = device

    runner = PerformanceSummary(
        movie_ae=mov,
        people_ae=per,
        max_rows=args.rows,
        batch_size=args.batch_size,
        table_width=args.table_width,
        similarity_out=args.similarity_out,
        similarity_max_items=args.similarity_max_items,
        plot_dir=args.plot_dir,
    )
    runner.run()


if __name__ == "__main__":
    main()
