import os
import json
from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class ExperimentResult:
    task: str
    rep: str
    run_id: str
    seed: int
    train_regime: str
    metric_name: str
    train_loss: float
    val_loss: float
    test_loss: float
    primary_metric: float
    primary_metric_regime: str
    metrics_by_regime: Dict[str, float]
    samples_path: str


class StageTracker:
    """
    Minimal persistence: know what is done + store results.
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.status_path = os.path.join(run_dir, "status.json")
        self.data = {"stages": {}}
        if os.path.exists(self.status_path):
            try:
                with open(self.status_path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {"stages": {}}

    def _save(self):
        tmp = self.status_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.status_path)

    @staticmethod
    def key(task: str, rep: str, seed: int) -> str:
        return f"{task}__{rep}__s{seed}"

    def is_done(self, task: str, rep: str, seed: int) -> bool:
        return self.data["stages"].get(self.key(task, rep, seed), {}).get("done", False)

    def mark_done(self, result: ExperimentResult):
        k = self.key(result.task, result.rep, result.seed)
        self.data["stages"][k] = {"done": True, "result": asdict(result)}
        self._save()

    def all_results(self):
        return self.data.get("stages", {})
