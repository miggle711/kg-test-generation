"""
dataset.py

Loads instance definitions from data/instances/*.json + data/patches/*.diff
into the Dict shape run_baseline/run_kg_augmented expect (repo, base_commit,
patch, code_file, test_file).

The dataset itself (see data/README.md) is a deliberately non-cherry-picked
sample of real psf/requests commits: clean single-function changes,
multi-function/multi-method commits, an ambiguous class-vs-method case
(same shape as issue #14's regression), and commits whose patch doesn't
resolve to any function at all. ncluded on purpose so the pipeline's
real failure/rejection rates on representative input are visible, not
hidden by only keeping instances that are already known to work.
"""

import json
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_instance(name: str) -> Dict:
    """Load a single instance by name (its data/instances/<name>.json
    stem) into the Dict shape run_baseline/run_kg_augmented expect.
    """
    instance_path = DATA_DIR / "instances" / f"{name}.json"
    raw = json.loads(instance_path.read_text())
    patch_path = DATA_DIR / raw["patch_file"]
    return {
        "name": raw["name"],
        "repo": raw["repo"],
        "base_commit": raw["base_commit"],
        "patch": patch_path.read_text(),
        "code_file": raw["code_file"],
        "test_file": raw["test_file"],
    }


def load_all_instances() -> List[Dict]:
    """Load every instance in data/instances/, sorted by name for a
    deterministic order across runs.
    """
    instance_paths = sorted((DATA_DIR / "instances").glob("*.json"))
    return [load_instance(p.stem) for p in instance_paths]
