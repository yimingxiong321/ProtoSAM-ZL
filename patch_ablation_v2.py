"""
Patch ProtoSAM.py, util/utils.py, validation_protosam.py, and config_ssl_upload.py
to support ablation experiments.

Ablation modes:
  "none"          - baseline (unchanged)
  "oracle_coarse" - replace ALPNet output with GT mask before CCA
  "oracle_prompt" - skip CCA, use GT bbox + GT center point as SAM prompt (early return)
  "coarse_only"   - return coarse prediction without SAM refinement
  "oracle_mask"   - use GT mask as SAM mask_input prompt (early return)

Ablation fixes (list, applied on top of baseline):
  "multimask_score" - multimask_output=True + best_pred_idx=argmax(score)
  "neg_points"      - use_neg_points=True
  "area_filter"     - CCA min_area=500 threshold
  "weighted_agg"    - pred = sum(m*s) instead of sum(masks)

Usage on server:
  cd /share/home/huafuchen01/zl/R2Seg-main/ProtoSAM-main/
  python3 outputs/patch_ablation_v2.py

To restore originals:
  cp models/ProtoSAM.py.bak models/ProtoSAM.py
  cp util/utils.py.bak util/utils.py
  cp validation_protosam.py.bak validation_protosam.py
  cp config_ssl_upload.py.bak config_ssl_upload.py
"""
import shutil
import os

PROTOSAM_PY = "models/ProtoSAM.py"
UTILS_PY = "util/utils.py"
VAL_PY = "validation_protosam.py"
CONFIG_PY = "config_ssl_upload.py"

PATCHED_MARKER = "# ABLATION_V2_PATCHED"


def restore_backup(filepath):
    if os.path.exists(filepath + ".bak"):
        shutil.copy2(filepath + ".bak", filepath)
        print(f"  Restored {filepath} from backup")
        return True
    return False


def patch_protosam():
    with open(PROTOSAM_PY, "r", encoding="utf-8") as f:
        code = f.read()

    if PATCHED_MARKER in code:
        print("[ProtoSAM] Already patched (v2), restoring backup first...")
        if not restore_backup(PROTOSAM_PY):
            print("  ERROR: No backup found. Cannot restore.")
            return
        with open(PROTOSAM_PY, "r", encoding="utf-8") as f:
            code = f.read()
    elif "ablation_mode" in code:
        print("[ProtoSAM] Found v1 patch, restoring backup first...")
        if not restore_backup(PROTOSAM_PY):
            print("  ERROR: No backup found. Cannot restore.")
            return
        with open(PROTOSAM_PY, "r", encoding="utf-8") as f:
            code = f.read()

    shutil.copy2(PROTOSAM_PY, PROTOSAM_PY + ".bak")
    changes = 0

    # 1. Add ablation params to __init__ signature
    old_init = "use_neg_points=False, ):"
    new_init = "use_neg_points=False, ablation_mode='none', ablation_fixes=None, ):"
    if old_init in code:
        code = code.replace(old_init, new_init, 1)
        changes += 1
        print("  [+] Added ablation params to __init__ signature")
    else:
        print("  [!] Could not find __init__ marker")

    # 2. Store ablation params in __init__ body
    old_store = "        self.coarse_pred_only = coarse_pred_only\n"
    new_store = (
        "        self.coarse_pred_only = coarse_pred_only\n"
        "        self.ablation_mode = ablation_mode\n"
        "        self.ablation_fixes = ablation_fixes or []\n"
    )
    if old_store in code:
        code = code.replace(old_store, new_store, 1)
        changes += 1
        print("  [+] Added ablation param storage")

    # 3. Add gt_mask param to forward signature
    old_fwd = "def forward(self, query_image, coarse_model_input, degrees_rotate=0):"
    new_fwd = "def forward(self, query_image, coarse_model_input, degrees_rotate=0, gt_mask=None):"
    if old_fwd in code:
        code = code.replace(old_fwd, new_fwd, 1)
        changes += 1
        print("  [+] Added gt_mask param to forward()")

    # 4. Modify coarse_pred_only check to also handle ablation coarse_only
    old_coarse = "        if self.coarse_pred_only: "
    new_coarse = '        if self.coarse_pred_only or self.ablation_mode == "coarse_only": '
    if old_coarse in code:
        code = code.replace(old_coarse, new_coarse, 1)
        changes += 1
        print("  [+] Added coarse_only mode to coarse_pred_only check")

    # 5. Insert ablation block after the second softmax (after resize, before CCA).
    #    This single block handles oracle_coarse, oracle_prompt, and oracle_mask.
    #    The marker is the second occurrence of softmax+argmax, preceded by "# output_p = output_logits"
    second_softmax_marker = (
        "        # output_p = output_logits\n"
        "        output_p = output_logits.softmax(dim=1)\n"
        "        pred = output_p.argmax(dim=1)[0]"
    )

    ablation_block = (
        "\n"
        "        # " + PATCHED_MARKER + "\n"
        "        # === ABLATION: oracle_coarse ===\n"
        '        if self.ablation_mode == "oracle_coarse" and gt_mask is not None:\n'
        "            _gt_r = F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0).float(),\n"
        "                                  size=output_logits.shape[-2:], mode='nearest')[0][0]\n"
        "            output_logits = torch.zeros_like(output_logits)\n"
        "            output_logits[0, 1] = _gt_r * 20\n"
        "            output_logits[0, 0] = (1 - _gt_r) * 20\n"
        "            output_p = output_logits.softmax(dim=1)\n"
        "            pred = output_p.argmax(dim=1)[0]\n"
        "        # === ABLATION: oracle_prompt (early return) ===\n"
        '        if self.ablation_mode == "oracle_prompt" and gt_mask is not None:\n'
        "            _gt_np = gt_mask.cpu().numpy().astype(np.uint8)\n"
        "            if _gt_np.max() == 0:\n"
        "                _ep = F.interpolate(pred.unsqueeze(0).unsqueeze(0).float(), size=original_size, mode='nearest')[0][0]\n"
        "                return _ep, [0]\n"
        "            _gt_r = cv2.resize(_gt_np, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)\n"
        "            _ys, _xs = np.where(_gt_r > 0)\n"
        "            _bbox = np.array([_xs.min(), _ys.min(), _xs.max(), _ys.max()])\n"
        "            _bboxes = np.array([_bbox])\n"
        "            _cy, _cx = int(_ys.mean()), int(_xs.mean())\n"
        "            _sam_pts = np.array([[[_cx, _cy]]])\n"
        "            _sam_neg_pts = [None]\n"
        "            _qry = query_image\n"
        "            if self.sam_trans is not None:\n"
        "                _qry = self.sam_trans.apply_image_torch(_qry[0])\n"
        "                _qry = self.sam_trans.preprocess(_qry)\n"
        "                _qry = _qry.permute(1, 2, 0).detach().cpu().numpy()\n"
        "            else:\n"
        "                _qry = _qry.permute(1, 2, 0).detach().cpu().numpy()\n"
        "            _qry = ((_qry - _qry.min()) / (_qry.max() - _qry.min()) * 255).astype(np.uint8)\n"
        "            _masks, _scores = self.predict_w_points_bbox(_sam_pts, _bboxes, _sam_neg_pts, _qry, pred, return_logits=True if self.training else False)\n"
        "            _pred_out = sum(_masks)\n"
        "            if not self.training:\n"
        "                _pred_out = _pred_out > 0\n"
        "            _pred_out = torch.tensor(_pred_out).float().to(output_p.device)\n"
        "            _pred_out = F.interpolate(_pred_out.unsqueeze(0).unsqueeze(0), size=original_size, mode='nearest')[0][0]\n"
        "            return _pred_out, _scores\n"
        "        # === ABLATION: oracle_mask (early return) ===\n"
        '        if self.ablation_mode == "oracle_mask" and gt_mask is not None:\n'
        "            _gt_np = gt_mask.cpu().numpy().astype(np.uint8)\n"
        "            if _gt_np.max() == 0:\n"
        "                _ep = F.interpolate(pred.unsqueeze(0).unsqueeze(0).float(), size=original_size, mode='nearest')[0][0]\n"
        "                return _ep, [0]\n"
        "            _gt_r = cv2.resize(_gt_np, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)\n"
        "            _sam_masks = np.stack([_gt_r.astype(np.float32)])\n"
        "            _qry = query_image\n"
        "            if self.sam_trans is not None:\n"
        "                _qry = self.sam_trans.apply_image_torch(_qry[0])\n"
        "                _qry = self.sam_trans.preprocess(_qry)\n"
        "                _qry = _qry.permute(1, 2, 0).detach().cpu().numpy()\n"
        "            else:\n"
        "                _qry = _qry.permute(1, 2, 0).detach().cpu().numpy()\n"
        "            _qry = ((_qry - _qry.min()) / (_qry.max() - _qry.min()) * 255).astype(np.uint8)\n"
        "            _masks, _scores = self.predict_w_masks(_sam_masks, _qry, original_size)\n"
        "            _pred_out = sum(_masks)\n"
        "            if not self.training:\n"
        "                _pred_out = _pred_out > 0\n"
        "            _pred_out = torch.tensor(_pred_out).float().to(output_p.device)\n"
        "            _pred_out = F.interpolate(_pred_out.unsqueeze(0).unsqueeze(0), size=original_size, mode='nearest')[0][0]\n"
        "            return _pred_out, _scores\n"
        "        # === END ABLATION BLOCK ===\n"
    )

    if second_softmax_marker in code:
        code = code.replace(second_softmax_marker, second_softmax_marker + ablation_block, 1)
        changes += 1
        print("  [+] Added ablation block (oracle_coarse + oracle_prompt + oracle_mask)")
    else:
        print("  [!] Could not find second softmax marker for ablation block insertion")

    # 6. Modify CCA call to support area_filter
    old_cca_call = "            conn_components = cca(_pred, output_logits, return_cc=True)"
    new_cca_call = (
        "            _min_area = 500 if 'area_filter' in self.ablation_fixes else 0\n"
        "            conn_components = cca(_pred, output_logits, return_cc=True, min_area=_min_area)"
    )
    if old_cca_call in code:
        code = code.replace(old_cca_call, new_cca_call, 1)
        changes += 1
        print("  [+] Added area_filter support to CCA call")

    # 7. Modify predict_w_points_bbox for multimask_score fix
    old_mm = "multimask_output=False if self.use_cca else True"
    new_mm = "multimask_output=False if (self.use_cca and 'multimask_score' not in self.ablation_fixes) else True"
    if old_mm in code:
        code = code.replace(old_mm, new_mm, 1)
        changes += 1
        print("  [+] Added multimask_score fix to predict_w_points_bbox")

    old_idx = "            best_pred_idx = 0"
    new_idx = "            best_pred_idx = np.argmax(score) if 'multimask_score' in self.ablation_fixes else 0"
    if old_idx in code:
        code = code.replace(old_idx, new_idx, 1)
        changes += 1
        print("  [+] Added multimask_score best_pred_idx selection")

    # 8. Modify mask aggregation for weighted_agg fix
    old_agg = "        pred = sum(masks)"
    new_agg = (
        "        if 'weighted_agg' in self.ablation_fixes:\n"
        "            pred = sum(m * s for m, s in zip(masks, scores))\n"
        "        else:\n"
        "            pred = sum(masks)"
    )
    if old_agg in code:
        code = code.replace(old_agg, new_agg, 1)
        changes += 1
        print("  [+] Added weighted_agg fix to mask aggregation")

    with open(PROTOSAM_PY, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"  Done patching {PROTOSAM_PY} ({changes} changes)")


def patch_utils():
    with open(UTILS_PY, "r", encoding="utf-8") as f:
        code = f.read()

    if PATCHED_MARKER in code:
        print("[utils] Already patched (v2), restoring backup first...")
        if not restore_backup(UTILS_PY):
            print("  ERROR: No backup found. Cannot restore.")
            return
        with open(UTILS_PY, "r", encoding="utf-8") as f:
            code = f.read()
    elif "ablation" in code:
        print("[utils] Found v1 patch, restoring backup first...")
        if not restore_backup(UTILS_PY):
            print("  ERROR: No backup found. Cannot restore.")
            return
        with open(UTILS_PY, "r", encoding="utf-8") as f:
            code = f.read()

    shutil.copy2(UTILS_PY, UTILS_PY + ".bak")

    # Add min_area parameter to cca() function
    old_sig = "def cca(query_pred_original, query_pred_logits, return_conf=False, return_cc=False):"
    new_sig = "def cca(query_pred_original, query_pred_logits, return_conf=False, return_cc=False, min_area=0):"
    if old_sig in code:
        code = code.replace(old_sig, new_sig, 1)
        print("  [+] Added min_area param to cca()")
    else:
        print("  [!] Could not find cca() signature")

    # Add area filtering after get_connected_components call
    old_get_cc = "    cca_output, cca_conf = get_connected_components(query_pred_original, query_pred_logits, return_conf=True)\n"
    new_get_cc = (
        "    cca_output, cca_conf = get_connected_components(query_pred_original, query_pred_logits, return_conf=True)\n"
        "    # " + PATCHED_MARKER + "\n"
        "    if min_area > 0:\n"
        "        for j in range(1, cca_output[0]):\n"
        "            area = cca_output[2][j][cv2.CC_STAT_AREA]\n"
        "            if area < min_area:\n"
        "                cca_conf[j] = 0\n"
    )
    if old_get_cc in code:
        code = code.replace(old_get_cc, new_get_cc, 1)
        print("  [+] Added area filtering logic to cca()")
    else:
        print("  [!] Could not find get_connected_components call in cca()")

    with open(UTILS_PY, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"  Done patching {UTILS_PY}")


def patch_validation():
    with open(VAL_PY, "r", encoding="utf-8") as f:
        code = f.read()

    if PATCHED_MARKER in code:
        print("[validation] Already patched (v2), restoring backup first...")
        if not restore_backup(VAL_PY):
            print("  ERROR: No backup found. Cannot restore.")
            return
        with open(VAL_PY, "r", encoding="utf-8") as f:
            code = f.read()
    elif "ablation_mode" in code:
        print("[validation] Found v1 patch, restoring backup first...")
        if not restore_backup(VAL_PY):
            print("  ERROR: No backup found. Cannot restore.")
            return
        with open(VAL_PY, "r", encoding="utf-8") as f:
            code = f.read()

    shutil.copy2(VAL_PY, VAL_PY + ".bak")

    # 1. Add ablation params to model constructor
    old_model = 'use_neg_points=_config["use_neg_points"],) '
    new_model = (
        'use_neg_points=_config["use_neg_points"],\n'
        '                    ablation_mode=_config.get("ablation_mode", "none"),\n'
        '                    ablation_fixes=_config.get("ablation_fixes", None),) '
    )
    if old_model in code:
        code = code.replace(old_model, new_model, 1)
        print("  [+] Added ablation params to model constructor")
    else:
        print("  [!] Could not find model constructor marker")

    # 2. Pass gt_mask to model() call
    old_call = "query_pred, scores = model(\n                        query_images, coarse_model_input, degrees_rotate=0)"
    new_call = (
        "query_pred, scores = model(\n"
        "                        query_images, coarse_model_input, degrees_rotate=0,\n"
        '                        gt_mask=query_labels[0].to(query_images.device) if _config.get("ablation_mode", "none") != "none" else None)'
    )
    if old_call in code:
        code = code.replace(old_call, new_call, 1)
        print("  [+] Added gt_mask pass-through to model() call")
    else:
        print("  [!] Could not find model() call marker")

    # 3. Add marker comment
    code += '\n# ' + PATCHED_MARKER + '\n'

    with open(VAL_PY, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"  Done patching {VAL_PY}")


def patch_config():
    with open(CONFIG_PY, "r", encoding="utf-8") as f:
        code = f.read()

    if PATCHED_MARKER in code:
        print("[config] Already patched (v2), restoring backup first...")
        if not restore_backup(CONFIG_PY):
            print("  ERROR: No backup found. Cannot restore.")
            return
        with open(CONFIG_PY, "r", encoding="utf-8") as f:
            code = f.read()
    elif "ablation_mode" in code:
        print("[config] Found v1 patch, restoring backup first...")
        if not restore_backup(CONFIG_PY):
            print("  ERROR: No backup found. Cannot restore.")
            return
        with open(CONFIG_PY, "r", encoding="utf-8") as f:
            code = f.read()

    shutil.copy2(CONFIG_PY, CONFIG_PY + ".bak")

    # Insert INSIDE the cfg() function body, right before the @ex.config_hook
    # Must be at 4-space indentation to be inside cfg()
    insert_marker = "\n\n@ex.config_hook"
    config_entries = (
        "\n\n    # " + PATCHED_MARKER + "\n"
        '    ablation_mode = "none"\n'
        "    ablation_fixes = None\n"
    )
    if insert_marker in code:
        code = code.replace(insert_marker, config_entries + insert_marker, 1)
        print("  [+] Added ablation config inside cfg() function")
    else:
        # Fallback: insert before the first @ex. decorator after cfg
        print("  [!] Could not find @ex.config_hook, using fallback")
        code = code.rstrip() + '\n\n# ' + PATCHED_MARKER + '\nablation_mode = "none"\nablation_fixes = None\n'

    with open(CONFIG_PY, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"  Done patching {CONFIG_PY}")


if __name__ == "__main__":
    if not os.path.exists(PROTOSAM_PY):
        print("ERROR: Run this script from the ProtoSAM-main/ directory")
        exit(1)

    print("=== Patching ProtoSAM.py ===")
    patch_protosam()
    print("\n=== Patching util/utils.py ===")
    patch_utils()
    print("\n=== Patching validation_protosam.py ===")
    patch_validation()
    print("\n=== Patching config_ssl_upload.py ===")
    patch_config()

    print("\n" + "=" * 50)
    print("All patches applied. Backups saved as .bak files.")
    print("\nTo restore originals:")
    print("  cp models/ProtoSAM.py.bak models/ProtoSAM.py")
    print("  cp util/utils.py.bak util/utils.py")
    print("  cp validation_protosam.py.bak validation_protosam.py")
    print("  cp config_ssl_upload.py.bak config_ssl_upload.py")
    print("\nRun experiments with:")
    print("  ./outputs/run_ablation.sh <mode> <organ> <modality> <label_set> [fixes]")
    print("Modes: none, oracle_coarse, oracle_prompt, coarse_only, oracle_mask")
    print("Fixes: multimask_score, neg_points, area_filter, weighted_agg")