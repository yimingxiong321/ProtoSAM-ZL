"""
Fix: add missing [None, ...] to _soft_mask_input in soft_prompt mode.
SAM's predict() expects mask_input to be 3D (1, H, W), not 2D (H, W).

Usage:
  cd /share/home/huafuchen01/zl/R2Seg-main/ProtoSAM-main/
  python3 outputs/patch_fix_soft_prompt.py
  ./outputs/run_ablation.sh soft_prompt liver mri 1 "" 0
"""

PROTOSAM_PY = "models/ProtoSAM.py"

old_line = "            _soft_mask_input = ((_soft_mask_input - 0.5) * 20).astype(np.float32)\n"
new_lines = (
    "            _soft_mask_input = ((_soft_mask_input - 0.5) * 20).astype(np.float32)\n"
    "            _soft_mask_input = _soft_mask_input[None, ...]\n"
)

with open(PROTOSAM_PY, "r", encoding="utf-8") as f:
    code = f.read()

if "_soft_mask_input[None, ...]" in code:
    print("Fix already applied, skipping.")
elif old_line in code:
    code = code.replace(old_line, new_lines, 1)
    with open(PROTOSAM_PY, "w", encoding="utf-8") as f:
        f.write(code)
    print("[+] Fixed: added [None, ...] to _soft_mask_input")
    print("    Run: ./outputs/run_ablation.sh soft_prompt liver mri 1 '' 0")
else:
    print("ERROR: Could not find the line to fix.")
    exit(1)
