"""
Add soft_mask ablation mode to an already-v2-patched ProtoSAM.py.
Delta patch: requires patch_ablation_v2.py (and optionally patch_oracle_cca.py).

soft_mask: bypass CCA and point/bbox entirely. Pass ALPNet's soft
probability map directly to SAM as mask_input (dense prompt).
Tests whether SAM can refine a coarse probability map without
geometric prompts (point/box).

Pipeline: ALPNet -> output_p[0,1] (prob) -> resize 256x256 -> logits -> SAM.predict(mask_input=...)

Usage on server:
  cd /share/home/huafuchen01/zl/R2Seg-main/ProtoSAM-main/
  python3 outputs/patch_ablation_v2.py        # base patch
  python3 outputs/patch_oracle_cca.py          # (optional, already applied)
  python3 outputs/patch_soft_mask.py           # then this delta

  ./outputs/run_ablation.sh soft_mask liver mri 1 "" 0

Also recommended (upper bound for mask_input approach):
  ./outputs/run_ablation.sh oracle_mask liver mri 1 "" 0
"""

PROTOSAM_PY = "models/ProtoSAM.py"

marker = "        # === END ABLATION BLOCK ===\n"

soft_mask_code = (
    "        # === ABLATION: soft_mask (early return) ===\n"
    '        if self.ablation_mode == "soft_mask":\n'
    "            _fg_prob = output_p[0, 1].detach().cpu().numpy()\n"
    "            if _fg_prob.max() < 0.01:\n"
    "                _ep = F.interpolate(pred.unsqueeze(0).unsqueeze(0).float(), size=original_size, mode='nearest')[0][0]\n"
    "                return _ep, [0]\n"
    "            _mask_input = cv2.resize(_fg_prob, (256, 256), interpolation=cv2.INTER_LINEAR)\n"
    "            _mask_input = ((_mask_input - 0.5) * 20).astype(np.float32)\n"
    "            _mask_input = _mask_input[None, ...]\n"
    "            _qry = query_image\n"
    "            if self.sam_trans is not None:\n"
    "                _qry = self.sam_trans.apply_image_torch(_qry[0])\n"
    "                _qry = self.sam_trans.preprocess(_qry)\n"
    "                _qry = _qry.permute(1, 2, 0).detach().cpu().numpy()\n"
    "            else:\n"
    "                _qry = _qry.permute(1, 2, 0).detach().cpu().numpy()\n"
    "            _qry = ((_qry - _qry.min()) / (_qry.max() - _qry.min() + 1e-8) * 255).astype(np.uint8)\n"
    "            self.predictor.set_image(_qry)\n"
    "            _masks, _scores, _ = self.predictor.predict(\n"
    "                mask_input=_mask_input,\n"
    "                multimask_output=True\n"
    "            )\n"
    "            _best_idx = int(np.argmax(_scores))\n"
    "            _pred_out = _masks[_best_idx]\n"
    "            if not self.training:\n"
    "                _pred_out = _pred_out > 0\n"
    "            _pred_out = torch.tensor(_pred_out).float().to(output_p.device)\n"
    "            _pred_out = F.interpolate(_pred_out.unsqueeze(0).unsqueeze(0), size=original_size, mode='nearest')[0][0]\n"
    "            return _pred_out, [float(_scores[_best_idx])]\n"
    "        # === END ABLATION BLOCK ===\n"
)

with open(PROTOSAM_PY, "r", encoding="utf-8") as f:
    code = f.read()

if "soft_mask" in code and "ablation_mode" in code:
    print("soft_mask already present in ProtoSAM.py, skipping.")
elif marker in code:
    code = code.replace(marker, soft_mask_code, 1)
    with open(PROTOSAM_PY, "w", encoding="utf-8") as f:
        f.write(code)
    print("[+] Added soft_mask mode to ProtoSAM.py")
    print("    Run: ./outputs/run_ablation.sh soft_mask liver mri 1 '' 0")
    print("    Also run oracle_mask for upper bound: ./outputs/run_ablation.sh oracle_mask liver mri 1 '' 0")
else:
    print("ERROR: Could not find marker in ProtoSAM.py")
    print("Make sure v2 patch is applied first: python3 outputs/patch_ablation_v2.py")
    exit(1)
