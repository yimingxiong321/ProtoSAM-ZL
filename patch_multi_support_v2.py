# patch_multi_support_v2.py - Fixed version with correct string matching
#
# The original patch failed because validation_protosam.py has extra params
# (original_sz, img_sz) that weren't in the match string.

fpath = 'validation_protosam.py'
with open(fpath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and replace lines 376-391 (0-indexed: 375-390)
# The exact block to replace:
start_marker = "                coarse_model_input = InputFactory.create_input("
end_marker = "                        gt_mask=query_labels[0].to(query_images.device) if _config.get(\"ablation_mode\", \"none\") != \"none\" else None)"

start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if start_marker in line and start_idx is None:
        # Verify this is the right block (not in a comment or elsewhere)
        if i > 0 and 'with torch.no_grad' in lines[i-1]:
            start_idx = i
    if start_idx is not None and end_marker in line:
        end_idx = i
        break

if start_idx is None or end_idx is None:
    print(f"ERROR: Could not find block to replace")
    print(f"  start_idx={start_idx}, end_idx={end_idx}")
    exit(1)

print(f"Found block at lines {start_idx+1}-{end_idx+1}")
print("Original block:")
for i in range(start_idx, end_idx+1):
    print(f"  {i+1}: {lines[i].rstrip()}")

# New replacement block
new_lines = """                # MULTI_SUPPORT_PATCHED
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
                                original_sz=query_images.shape[-2:],
                                img_sz=query_images.shape[-2:],
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
                                            original_sz=query_images.shape[-2:],
                                            img_sz=query_images.shape[-2:],
                                            gts=query_labels,
                    )
                    coarse_model_input.to(torch.device("cuda"))
                        
                    query_pred, scores = model(
                            query_images, coarse_model_input, degrees_rotate=0,
                            gt_mask=query_labels[0].to(query_images.device) if _config.get("ablation_mode", "none") != "none" else None)
""".splitlines(True)

# Replace
lines[start_idx:end_idx+1] = new_lines

with open(fpath, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"\nPatched {fpath} successfully!")
print(f"Replaced lines {start_idx+1}-{end_idx+1} with {len(new_lines)} new lines")
