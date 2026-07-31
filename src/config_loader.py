from __future__ import annotations

from pathlib import Path
import yaml


_DEFAULT_CONFIG = Path(__file__).parent.parent / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)

    root   = cfg["paths"]["dataset_root"]
    outdir = cfg["paths"]["output_dir"]

    for ds_cfg in cfg["datasets"].values():
        base = f"{root}/{ds_cfg['subdir']}"
        ds_cfg["train_images"] = f"{base}/train/images"
        ds_cfg["train_masks"]  = f"{base}/train/masks"
        ds_cfg["test_images"]  = f"{base}/test/images"
        ds_cfg["test_masks"]   = f"{base}/test/masks"

    expl     = cfg.get("explainability", {})
    resolved = {}
    for ds_name, filename in expl.get("image_paths", {}).items():
        subdir = cfg["datasets"][ds_name]["subdir"]
        resolved[ds_name] = f"{root}/{subdir}/test/images/{filename}"
    if expl:
        expl["image_paths"] = resolved

    cfg["analysis"] = {"output_file": f"{outdir}/analysis.txt"}

    return cfg
