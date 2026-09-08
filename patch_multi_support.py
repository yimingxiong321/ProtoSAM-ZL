# patch_multi_support.py
#
# Multi-Support Consensus: run ALPNet with all 3 support parts, average logits,
# then do CCA + SAM once. Targets the FP problem (Precision 69.98% vs oracle 84.44%).
#
# Changes:
# 1. validation_protosam.py: when ablation_mode='multi_support', create 3 inputs
# 2. ProtoSAM.py forward(): when input is a list, run coarse model N times, average logits

import re

# === Patch 1: validation_protosam.py ===
fpath = 'validation_protosam.py'
with open(fpath, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the block where coarse_model_input is created and model is called
# We need to wrap it in a conditional for multi_support mode

old_block = '''                coarse_model_input = InputFactory.create_input(
                                          input_type=_config["base_model"],
                                          query_image=query_images,
                                          support_images=support_images,
                                          support_labels=support_fg_mask,
                                          isval=True,
                                          val_wsize=_config["val_wsize"],
                                          gts=query_labels,
                  )
                coarse_model_input.to(torch.device("cuda"))
                      
                query_pred, scores = model(
                        query_images, coarse_model_input, degrees_rotate=0,
                        gt_mask=query_labels[0].to(query_images.device) if _config.get("ablation_mode", "none") != "none" else None)'''

new_block = '''                # MULTI_SUPPORT_PATCHED
                if _config.get("ablation_mode") == "multi_support" and is_alp_ds:
                    coarse_model_inputs = []
                    for _p in range(len(all_support_images)):
                        _si, _sl = update_support_set_by_scan_part(all_support_images, all_support_fg_mask, _p)
                        _cmi = InputFactory.create_input(
                                  input_type=_config["base_model"],
                                  query_image=query_images,
                                  support_images=_si,
                                  support_labels=_sl,
                                  isval=True,
                                  val_wsize=_config["val_wsize"],
                                  gts=query_labels,
                        )
                        _cmi.to(torch.device("cuda"))
                        coarse_model_inputs.append(_cmi)
                    query_pred, scores = model(
                        query_images, coarse_model_inputs, degrees_rotate=0,
                        gt_mask=query_labels[0].to(query_images.device))
                else:
                    coarse_model_input = InputFactory.create_input(
                                              input_type=_config["base_model"],
                                              query_image=query_images,
                                              support_images=support_images,
                                              support_labels=support_fg_mask,
                                              isval=True,
                                              val_wsize=_config["val_wsize"],
                                              gts=query_labels,
                      )
                    coarse_model_input.to(torch.device("cuda"))
                      
                    query_pred, scores = model(
                            query_images, coarse_model_input, degrees_rotate=0,
                            gt_mask=query_labels[0].to(query_images.device) if _config.get("ablation_mode", "none") != "none" else None)'''

if old_block in content:
    content = content.replace(old_block, new_block)
    print(f"Patched {fpath}: added multi_support branch")
else:
    # Try with different whitespace
    print(f"WARNING: exact block not found in {fpath}, trying flexible match...")
    # Check if already patched
    if "MULTI_SUPPORT_PATCHED" in content:
        print(f"  Already patched, skipping")
    else:
        print(f"  ERROR: Could not find the block to replace")
        print(f"  Please check the file manually")

with open(fpath, 'w', encoding='utf-8') as f:
    f.write(content)


# === Patch 2: ProtoSAM.py forward() ===
fpath2 = 'models/ProtoSAM.py'
with open(fpath2, 'r', encoding='utf-8') as f:
    content2 = f.read()

# Check if already patched
if "MULTI_SUPPORT_PATCHED" in content2:
    print(f"{fpath2} already patched, skipping")
else:
    # Replace the coarse model call block
    old_coarse = '''        coarse_model_input.set_query_images(rotated_img)
        output_logits_rot = self.coarse_segmentation_model(coarse_model_input)'''

    new_coarse = '''        # MULTI_SUPPORT_PATCHED: support list averaging
        if isinstance(coarse_model_input, list):
            _all_logits = []
            for _cmi in coarse_model_input:
                _cmi.set_query_images(rotated_img)
                _logits_rot = self.coarse_segmentation_model(_cmi)
                if degrees_rotate != 0:
                    _logits = reverse_tensor(_logits_rot, rot_h, rot_w, -degrees_rotate)
                else:
                    _logits = _logits_rot
                _all_logits.append(_logits)
            output_logits = torch.stack(_all_logits, dim=0).mean(dim=0)
            output_logits_rot = output_logits  # for debug viz
        else:
            coarse_model_input.set_query_images(rotated_img)
            output_logits_rot = self.coarse_segmentation_model(coarse_model_input)'''

    if old_coarse in content2:
        content2 = content2.replace(old_coarse, new_coarse)
        print(f"Patched {fpath2}: added multi_support averaging in forward()")
    else:
        print(f"WARNING: Could not find coarse model call in {fpath2}")

    with open(fpath2, 'w', encoding='utf-8') as f:
        f.write(content2)


print("\n" + "=" * 60)
print("Multi-Support Consensus patch complete!")
print("=" * 60)
print("\nUsage:")
print("  ./run_ablation.sh multi_support liver mri 1 '' 0")
print("\nWhat it does:")
print("  - Runs ALPNet 3 times (one per support part)")
print("  - Averages the output_logits (soft ensemble)")
print("  - Runs CCA + SAM once on the averaged prediction")
print("  - Expected: FP reduction -> Precision increase")
print("\nKey metric to watch:")
print("  Precision (baseline 5-fold: 69.98%, oracle upper bound: 84.44%)")
