import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import torch

from evaluate import run_evaluation
from experiment import TASKS, config_for, load_splits, prepare_run, train_all
from report import build_report


REPO_FILES = ("models.py", "experiment.py", "evaluate.py", "report.py", "main.py",
              "requirements.txt", "README.md", ".gitignore")


def read_histories(folder):
    histories = {}
    for task in TASKS:
        file = folder / f"history_{task}.csv"
        if not file.is_file():
            raise FileNotFoundError(f"Belum ada {file.name}. Jalankan training dulu.")
        with file.open(encoding="utf-8", newline="") as stream:
            histories[task] = [
                {key: int(value) if key == "epoch" else float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)
            ]
    return histories


def export_source_zip(project, target):
    from zipfile import ZIP_DEFLATED, ZipFile

    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for filename in REPO_FILES:
            archive.write(project / filename, arcname=filename)
    return target


def run(cfg, data, folder, device_name="auto", stage="all", use_fid=None, identity=None):
    folder = Path(folder)
    identity = identity or {"name": "", "nim": "", "class": "", "github": ""}
    if stage == "report":
        return build_report(folder, identity)

    selected = ("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA belum tersedia. Pilih GPU di Runtime Colab atau gunakan device='cpu'.")
    device = torch.device(selected)
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    use_fid = cfg.profile == "full" if use_fid is None else use_fid
    prepare_run(folder, cfg, device)
    splits = load_splits(data, cfg, folder)
    print(f"{cfg.profile} | {device} | split {[len(part) for part in splits]}", flush=True)
    if stage in {"all", "train"}:
        histories = train_all(cfg, splits, folder, device)
    else:
        histories = read_histories(folder)

    if stage in {"all", "evaluate"}:
        run_evaluation(cfg, splits, histories, folder, device, use_fid)
    if stage == "all":
        return build_report(folder, identity)
    return folder


def main():
    parser = argparse.ArgumentParser(description="Eksperimen Fashion-MNIST enam model generatif")
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--stage", choices=("all", "train", "evaluate", "report"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fid", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--identity", type=Path, help="JSON dengan name, nim, class, github.")
    args = parser.parse_args()
    cfg = config_for(args.profile, args.workers)
    folder = args.output or Path("runs") / args.profile
    identity = json.loads(args.identity.read_text()) if args.identity else None
    run(cfg, args.data, folder, args.device, args.stage, args.fid, identity)


if __name__ == "__main__":
    main()
