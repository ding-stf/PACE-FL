from __future__ import annotations
import csv,json,logging,re,shutil
from pathlib import Path
from typing import Any
import yaml

def result_directory(root: Path,dataset: str,method: str,seed: int,run_tag: str="",overwrite: bool=False)->Path:
    if run_tag and not re.fullmatch(r"[A-Za-z0-9_.-]+",run_tag):
        raise ValueError("run_tag may contain only letters, digits, '_', '-', and '.'")
    out=root/dataset/method
    if run_tag:
        out=out/run_tag
    out=out/f"seed_{seed}"
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Result directory already exists: {out}. Use a unique --run_tag or explicit --overwrite.")
        shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)
    return out
def save_config(path: Path,config: dict[str,Any])->None:
    with path.open("w",encoding="utf-8") as f: yaml.safe_dump(config,f,sort_keys=False)
def write_rows(path: Path,rows: list[dict])->None:
    if not rows: return
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def configure_logging(path: Path)->None:
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",handlers=[logging.FileHandler(path,encoding="utf-8"),logging.StreamHandler()],force=True)
