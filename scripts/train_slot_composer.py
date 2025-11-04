import argparse
import logging
from config import project_config, ensure_dirs
from scripts.slot_composer.fm_trainer import FlowMatchingSlotComposerTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("slot composer training entrypoint (flow-matching)")
    ensure_dirs(project_config)
    trainer = FlowMatchingSlotComposerTrainer(project_config, resume=bool(args.resume))
    trainer.train()

if __name__ == "__main__":
    main()
