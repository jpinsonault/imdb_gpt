# scripts/train_many_to_many.py
from config import project_config
from scripts.autoencoder.many_to_many.trainer import train_many_to_many

def main():
    _ = train_many_to_many(
        config=project_config,
        steps=project_config.max_training_steps,
        save_every=project_config.save_interval,
        warm_start=False,
    )

if __name__ == "__main__":
    main()
