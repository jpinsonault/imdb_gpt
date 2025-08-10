from pathlib import Path
from config import project_config
from scripts.autoencoder.joint.training import build_joint_trainer

def main():
    db_path = Path(project_config["data_dir"]) / "imdb.db"
    joint_trainer = build_joint_trainer(project_config, warm=False, db_path=db_path)
    joint_trainer.train()


if __name__ == "__main__":
    main()
