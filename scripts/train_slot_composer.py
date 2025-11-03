import logging
from config import project_config, ensure_dirs
from scripts.slot_composer.trainer import SlotComposerTrainer

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("slot composer training entrypoint")
    ensure_dirs(project_config)
    logging.info(
        "config model_dir=%s db_path=%s batch_size=%d slots=%d epochs=%d",
        project_config.model_dir,
        project_config.db_path,
        project_config.batch_size,
        project_config.slot_people_count,
        project_config.slot_epochs,
    )
    trainer = SlotComposerTrainer(project_config)
    trainer.train()
    logging.info("slot composer training finished")

if __name__ == "__main__":
    main()
