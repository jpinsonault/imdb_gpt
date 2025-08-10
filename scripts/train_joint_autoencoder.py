from pathlib import Path
from config import project_config
from scripts.autoencoder.joint.training import build_joint_trainer, train_joint

def main():
    db_path = Path(project_config["data_dir"]) / "imdb.db"
    joint_model, loader, logger, mov_ae, per_ae, total_edges = build_joint_trainer(project_config, warm=False, db_path=db_path)
    train_joint(project_config, joint_model, loader, logger, mov_ae, per_ae, total_edges, db_path)

if __name__ == "__main__":
    main()
