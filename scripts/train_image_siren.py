import argparse

from config import project_config, ensure_dirs
from scripts.image_siren.trainer import ImageSirenTrainer


def main():
    parser = argparse.ArgumentParser(description="Train conditional image SIREN on AE latents")
    parser.add_argument(
        "--ae-checkpoint",
        type=str,
        required=True,
        help="Path to trained ConvAutoencoder checkpoint (from train_image_autoencoder.py)",
    )
    args = parser.parse_args()

    ensure_dirs(project_config)

    trainer = ImageSirenTrainer(
        cfg=project_config,
        ae_checkpoint=args.ae_checkpoint,
    )
    trainer.train()


if __name__ == "__main__":
    main()
