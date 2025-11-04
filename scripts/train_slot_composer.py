import argparse
import logging
from config import project_config, ensure_dirs
from scripts.slot_composer.set_flow_trainer import SetFlowSlotComposerTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("slot composer training entrypoint (set flow)")
    ensure_dirs(project_config)
    logging.info(
        "config model_dir=%s db_path=%s batch_size=%d slots=%d epochs=%d steps=%d",
        project_config.model_dir,
        project_config.db_path,
        project_config.batch_size,
        project_config.slot_people_count,
        project_config.slot_epochs,
        project_config.slot_layers,
    )
    trainer = SetFlowSlotComposerTrainer(
        project_config,
        steps=project_config.slot_layers,
        path_weight=1.0,
        resume=bool(args.resume),
    )
    trainer.train()
    logging.info("slot composer training finished (set flow)")

if __name__ == "__main__":
    main()
