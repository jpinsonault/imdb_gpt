import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from image_encoding.dataset import ImageFolderDataset
from image_encoding.autoencoder import ConvAutoencoder
from image_encoding.callbacks import ReconstructionSaver


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--runs_dir", type=str, default="runs/image_autoencoder")

    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=128)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--recon_every", type=int, default=1)
    parser.add_argument("--max_recon_samples", type=int, default=8)

    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    device = torch.device(args.device)

    dataset = ImageFolderDataset(
        root=args.data_dir,
        image_size=(args.image_size, args.image_size),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = ConvAutoencoder(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        image_size=args.image_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    sample_batch = next(iter(loader))
    recon_saver = ReconstructionSaver(
        output_dir=run_dir,
        sample_batch=sample_batch,
        every_n_epochs=args.recon_every,
        max_samples=args.max_recon_samples,
    )

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_batches = 0

        for batch in loader:
            batch = batch.to(device)

            recon = model(batch)
            loss = F.mse_loss(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        print(f"epoch {epoch} loss {avg_loss:.6f}")

        recon_saver.maybe_save(epoch, model, device)

        ckpt_path = os.path.join(run_dir, f"model_epoch_{epoch:04d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": vars(args),
            },
            ckpt_path,
        )


if __name__ == "__main__":
    main()
