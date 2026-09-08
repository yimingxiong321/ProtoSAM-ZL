"""Fix indentation in the SPEN FG_PROT_MODE check."""
import os

GRID_PROTO_PY = "models/grid_proto_fewshot.py"

old_block = """                        else:
                        fg_mode = FG_PROT_MODE if F.avg_pool2d(_msk, k_size).max(
                        ) >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask'
                        # TODO figure out kernel size"""

new_block = """                        else:
                            fg_mode = FG_PROT_MODE if F.avg_pool2d(_msk, k_size).max(
                            ) >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask'
                            # TODO figure out kernel size"""

with open(GRID_PROTO_PY, "r", encoding="utf-8") as f:
    code = f.read()

if old_block in code:
    code = code.replace(old_block, new_block, 1)
    with open(GRID_PROTO_PY, "w", encoding="utf-8") as f:
        f.write(code)
    print("[+] Fixed indentation in FG_PROT_MODE else block")
else:
    print("[!] Could not find the block to fix")
    print("Check manually around line 292 of grid_proto_fewshot.py")