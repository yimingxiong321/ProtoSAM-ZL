"""Read-only server preflight for ProtoSAM P0 evidence and P1 execution."""

import argparse
import json
from pathlib import Path


RECOVERY_RUNS = (
    "dinov2_l14_mri_cca_grid_8_res_672_liver_dense_fg_hard_attn_soft_prompt_a05_fold_1",
    "dinov2_l14_mri_cca_grid_8_res_672_liver_dense_fg_hard_attn_soft_prompt_a05_fold_3",
    "dinov2_l14_mri_cca_grid_8_res_672_liver_dense_fg_hard_attn_soft_prompt_a05_fold_4",
)


def find_directory(root, name):
    direct = root / name
    if direct.is_dir():
        return direct
    return next((path for path in root.rglob(name) if path.is_dir()), None)


def sacred_status(run_dir):
    if run_dir is None:
        return {"directory": None, "complete": False, "missing": ["directory"]}
    missing = []
    for filename in ("metrics.json", "config.json", "run.json"):
        matches = list(run_dir.rglob(filename))
        if not matches or all(path.stat().st_size == 0 for path in matches):
            missing.append(filename)
    sources = [
        path for source_dir in run_dir.rglob("source") if source_dir.is_dir()
        for path in source_dir.rglob("*") if path.is_file() and path.stat().st_size > 0
    ]
    if not sources:
        missing.append("source/")
    return {
        "directory": str(run_dir),
        "complete": not missing,
        "missing": missing,
        "source_file_count": len(sources),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    required = {
        "config_ssl_upload.py": root / "config_ssl_upload.py",
        "validation_protosam.py": root / "validation_protosam.py",
        "dataloaders/": root / "dataloaders",
        "models/": root / "models",
        "SAM checkpoint": root / "pretrained_model/sam_vit_h.pth",
        "CHAOST2 data": root / "data/CHAOST2/chaos_MR_T2_normalized_672",
    }
    assets = {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "nonempty": path.stat().st_size > 0 if path.is_file() else (
                any(path.iterdir()) if path.is_dir() else False
            ),
        }
        for name, path in required.items()
    }
    hub_root = root / "pretrained_model/hub"
    dino_files = []
    if hub_root.is_dir():
        dino_files = [
            str(path) for path in hub_root.rglob("*")
            if path.is_file() and "dinov2" in str(path).lower()
            and ("vitl14" in path.name.lower() or "dinov2_vitl14" in str(path).lower())
        ]
    assets["DINOv2-L/14 hub cache"] = {
        "path": str(hub_root),
        "exists": bool(dino_files),
        "nonempty": bool(dino_files),
        "matches": dino_files[:10],
    }
    evidence = {
        name: sacred_status(find_directory(root, name)) for name in RECOVERY_RUNS
    }
    payload = {
        "project_root": str(root),
        "assets": assets,
        "recovered_sacred_runs": evidence,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    asset_ok = all(item["exists"] and item["nonempty"] for item in assets.values())
    evidence_ok = all(item["complete"] for item in evidence.values())
    if not asset_ok or not evidence_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
