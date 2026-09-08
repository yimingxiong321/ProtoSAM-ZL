"""
Add oracle_cca ablation mode to an already-v2-patched ProtoSAM.py.
This is a delta patch: requires patch_ablation_v2.py to have been applied first.

oracle_cca: ALPNet produces real noisy output (NOT replaced by GT).
Instead of CCA's confidence formula, the connected component with
highest GT IoU is selected. This isolates CCA selection quality
from ALPNet mask quality.

Usage:
  cd /share/home/huafuchen01/zl/R2Seg-main/ProtoSAM-main/
  python3 outputs/patch_ablation_v2.py     # ensure v2 base patch is applied
  python3 outputs/patch_oracle_cca.py      # then apply this delta
"""
import os

PROTOSAM_PY = "models/ProtoSAM.py"

old_cca_block = (
    "        if self.use_cca:\n"
    "            _min_area = 500 if 'area_filter' in self.ablation_fixes else 0\n"
    "            conn_components = cca(_pred, output_logits, return_cc=True, min_area=_min_area)\n"
    "            conf=None\n"
    "        else:\n"
    "            conn_components, conf = get_connected_components(_pred, output_logits, return_conf=True)"
)

new_cca_block = (
    '        if self.ablation_mode == "oracle_cca" and gt_mask is not None:\n'
    "            _gt_r = F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0).float(),\n"
    "                                  size=_pred.shape[-2:], mode='nearest')[0][0].cpu().numpy()\n"
    "            _cca_raw = cv2.connectedComponentsWithStats(_pred.astype(np.uint8), connectivity=8)\n"
    "            _best_iou, _best_key = 0.0, 0\n"
    "            for _j in range(1, _cca_raw[0]):\n"
    "                _comp = (_cca_raw[1] == _j).astype(np.uint8)\n"
    "                _inter = float((_comp * _gt_r).sum())\n"
    "                _union = float(_comp.sum() + _gt_r.sum() - _inter)\n"
    "                _iou = _inter / (_union + 1e-6)\n"
    "                if _iou > _best_iou:\n"
    "                    _best_iou = _iou\n"
    "                    _best_key = _j\n"
    "            if _best_key == 0 and _cca_raw[0] > 1:\n"
    "                _best_key = max(range(1, _cca_raw[0]),\n"
    "                                key=lambda _jj: _cca_raw[2][_jj][cv2.CC_STAT_AREA])\n"
    "            if _best_key > 0:\n"
    "                _new_cca = list(_cca_raw)\n"
    "                _new_cca[0] = 2\n"
    "                _new_cca[1] = np.where(_cca_raw[1] != _best_key, 0, 1)\n"
    "                _new_cca[2] = _cca_raw[2][[0, _best_key]]\n"
    "                _new_cca[3] = _cca_raw[3][[0, _best_key]]\n"
    "                conn_components = tuple(_new_cca)\n"
    "            else:\n"
    "                conn_components = (1, np.zeros_like(_pred, dtype=np.uint8),\n"
    "                                   np.zeros((1, 5), dtype=np.int32),\n"
    "                                   np.zeros((1, 2), dtype=np.float64))\n"
    "            conf = None\n"
    "        elif self.use_cca:\n"
    "            _min_area = 500 if 'area_filter' in self.ablation_fixes else 0\n"
    "            conn_components = cca(_pred, output_logits, return_cc=True, min_area=_min_area)\n"
    "            conf=None\n"
    "        else:\n"
    "            conn_components, conf = get_connected_components(_pred, output_logits, return_conf=True)"
)

with open(PROTOSAM_PY, "r", encoding="utf-8") as f:
    code = f.read()

if "oracle_cca" in code:
    print("oracle_cca already present in ProtoSAM.py, skipping.")
elif old_cca_block in code:
    code = code.replace(old_cca_block, new_cca_block, 1)
    with open(PROTOSAM_PY, "w", encoding="utf-8") as f:
        f.write(code)
    print("[+] Added oracle_cca branch to ProtoSAM.py")
    print("    Run: ./outputs/run_ablation.sh oracle_cca liver mri 1 '' 0")
else:
    print("ERROR: Could not find CCA call block in ProtoSAM.py")
    print("Make sure v2 patch is applied first:")
    print("  python3 outputs/patch_ablation_v2.py")
    print("Or the CCA block format has changed. Check manually.")
    exit(1)
