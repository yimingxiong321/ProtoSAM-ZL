"""
Add soft_prompt ablation mode to an already-v2-patched ProtoSAM.py.
Delta patch: requires patch_ablation_v2.py (+ optional patch_oracle_cca.py).

soft_prompt: Keep CCA + point/bbox extraction (same as baseline), BUT also
pass ALPNet's soft probability map as SAM mask_input (dense prompt).
Tests whether combining geometric prompts with dense probability info
recovers organ regions that argmax/CCA discarded.

Pipeline: ALPNet -> argmax -> CCA -> bbox+point (baseline flow)
                -> ALSO: prob[0,1] -> resize 256x256 -> logits -> mask_input
          -> SAM.predict(point=..., box=..., mask_input=...)

Also fixes the predict_w_masks uint8 bug in oracle_mask mode.

Usage:
  cd /share/home/huafuchen01/zl/R2Seg-main/ProtoSAM-main/
  python3 outputs/patch_ablation_v2.py
  python3 outputs/patch_oracle_cca.py    # already applied
  python3 outputs/patch_soft_mask.py     # already applied
  python3 outputs/patch_soft_prompt.py   # then this delta

  ./outputs/run_ablation.sh soft_prompt liver mri 1 "" 0
"""

PROTOSAM_PY = "models/ProtoSAM.py"

with open(PROTOSAM_PY, "r", encoding="utf-8") as f:
    code = f.read()

changes = 0

# 1. Add sam_mask_input param to predict_w_points_bbox
old_sig = "def predict_w_points_bbox(self, sam_input_points, bboxes, sam_neg_input_points, qry_img, pred, return_logits=False):"
new_sig = "def predict_w_points_bbox(self, sam_input_points, bboxes, sam_neg_input_points, qry_img, pred, return_logits=False, sam_mask_input=None):"
if old_sig in code:
    code = code.replace(old_sig, new_sig, 1)
    changes += 1
    print("  [+] Added sam_mask_input param to predict_w_points_bbox")
else:
    print("  [!] Could not find predict_w_points_bbox signature (may already be patched)")

# 2. Replace commented-out mask_input line with active one
old_commented = "                # mask_input=sam_mask_input,"
new_active = "                mask_input=sam_mask_input if sam_mask_input is not None else None,"
if old_commented in code:
    code = code.replace(old_commented, new_active, 1)
    changes += 1
    print("  [+] Activated mask_input in predict_w_points_bbox predict call")
else:
    print("  [!] Could not find commented mask_input line (may already be patched)")

# 3. Add soft_prompt mode block (after END ABLATION BLOCK, before CCA)
#    This does NOT early-return. It sets a flag so the normal CCA+SAM flow
#    runs, but with mask_input appended.
marker = "        # === END ABLATION BLOCK ===\n"
soft_prompt_block = (
    "        # === ABLATION: soft_prompt (flag, no early return) ===\n"
    "        _soft_mask_input = None\n"
    '        if self.ablation_mode == "soft_prompt":\n'
    "            _fg_prob = output_p[0, 1].detach().cpu().numpy()\n"
    "            _soft_mask_input = cv2.resize(_fg_prob, (256, 256), interpolation=cv2.INTER_LINEAR)\n"
    "            _soft_mask_input = ((_soft_mask_input - 0.5) * 20).astype(np.float32)\n"
    "        # === END ABLATION BLOCK ===\n"
)
if marker in code and "soft_prompt" not in code:
    code = code.replace(marker, soft_prompt_block, 1)
    changes += 1
    print("  [+] Added soft_prompt flag block")
else:
    print("  [!] Could not find marker or soft_prompt already present")

# 4. Pass _soft_mask_input to predict_w_points_bbox call
old_call = "masks, scores = self.predict_w_points_bbox(sam_input_points, bboxes, sam_neg_input_points, query_image, pred, return_logits=True if self.training else False)"
new_call = "masks, scores = self.predict_w_points_bbox(sam_input_points, bboxes, sam_neg_input_points, query_image, pred, return_logits=True if self.training else False, sam_mask_input=_soft_mask_input if self.ablation_mode == 'soft_prompt' else None)"
if old_call in code:
    code = code.replace(old_call, new_call, 1)
    changes += 1
    print("  [+] Added soft_mask_input pass-through to predict_w_points_bbox call")
else:
    print("  [!] Could not find predict_w_points_bbox call (may have different formatting)")

# 5. Fix the predict_w_masks uint8 bug (affects oracle_mask)
old_bug = "mask_input=in_mask[None, ...].astype(np.uint8)"
new_fix = "mask_input=in_mask[None, ...].astype(np.float32)"
if old_bug in code:
    code = code.replace(old_bug, new_fix, 1)
    changes += 1
    print("  [+] Fixed predict_w_masks uint8 bug (now float32)")
else:
    print("  [!] Could not find predict_w_masks uint8 bug (may already be fixed)")

if changes > 0:
    with open(PROTOSAM_PY, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"\n[+] Done patching ProtoSAM.py ({changes} changes)")
    print("    Run: ./outputs/run_ablation.sh soft_prompt liver mri 1 '' 0")
    print("    Also re-run oracle_mask to see fixed result:")
    print("    ./outputs/run_ablation.sh oracle_mask liver mri 1 '' 0")
else:
    print("\nNo changes made. ProtoSAM.py may already be fully patched.")
