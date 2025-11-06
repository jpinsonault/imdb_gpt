from config import project_config, ensure_dirs
from scripts.path_siren.trainer import PathSirenTrainer

def main():
    ensure_dirs(project_config)
    trainer = PathSirenTrainer(project_config)
    trainer.train()

if __name__ == "__main__":
    main()
