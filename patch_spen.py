"""
Patch grid_proto_fewshot.py to support SPEN (Self-guided Prototype ENhancement).

Adds 'spen' as a new cls_name option. When cls_name='spen', the model uses
SPENProtoMatcher instead of MultiProtoAsConv, enabling:
  - ALPG: adaptive local prototype generation (FPS-based)
  - QLPE: query-guided prototype weighting (Sinkhorn OT)

Three sub-modes selectable via proto_grid_size interpretation:
  spen_alpg: ALPG only
  spen_qlpe: fixed grid + QLPE
  spen_full: ALPG + QLPE (both)

Usage on server:
  cd /share/home/huafuchen01/zl/R2Seg-main/ProtoSAM-main/
  cp outputs/spen.py models/spen.py
  python3 outputs/patch_spen.py

  # Run experiments:
  ./outputs/run_ablation.sh spen_alpg liver mri 1 "" 0
  ./outputs/run_ablation.sh spen_full liver mri 1 "" 0
  ./outputs/run_ablation.sh spen_qlpe liver mri 1 "" 0
"""
import shutil
import os

GRID_PROTO_PY = "models/grid_proto_fewshot.py"

MARKER = "# SPEN_PATCHED"


def restore_backup(filepath):
    if os.path.exists(filepath + ".bak"):
        shutil.copy2(filepath + ".bak", filepath)
        return True
    return False


with open(GRID_PROTO_PY, "r", encoding="utf-8") as f:
    code = f.read()

if MARKER in code:
    print("Already patched, restoring backup first...")
    if not restore_backup(GRID_PROTO_PY):
        print("ERROR: No backup found.")
        exit(1)
    with open(GRID_PROTO_PY, "r", encoding="utf-8") as f:
        code = f.read()

shutil.copy2(GRID_PROTO_PY, GRID_PROTO_PY + ".bak")
changes = 0

# 1. Add import for SPENProtoMatcher at the top
old_import = "from .alpmodule import MultiProtoAsConv"
new_import = (
    "from .alpmodule import MultiProtoAsConv\n"
    "from .spen import SPENProtoMatcher  # " + MARKER + "\n"
)
if old_import in code:
    code = code.replace(old_import, new_import, 1)
    changes += 1
    print("  [+] Added SPENProtoMatcher import")
else:
    print("  [!] Could not find import marker")

# 2. Add SPEN branch in get_cls() method
old_cls = (
    '        if self.config[\'cls_name\'] == \'grid_proto\':\n'
    '            embed_dim = 256\n'
    "            if 'dinov2_b14' in self.config['which_model']:\n"
    '                embed_dim = 768\n'
    "            elif 'dinov2_l14' in self.config['which_model']:\n"
    '                embed_dim = 1024\n'
    '            self.cls_unit = MultiProtoAsConv(proto_grid=[proto_hw, proto_hw], feature_hw=self.config["feature_hw"], embed_dim=embed_dim)  # when treating it as ordinary prototype\n'
    '            print(f"cls unit feature hw: {self.cls_unit.feature_hw}")\n'
    '        else:\n'
    '            raise NotImplementedError(\n'
    '                f\'Backbone network {self.config["which_model"]} not implemented\')'
)

new_cls = (
    '        if self.config[\'cls_name\'] == \'grid_proto\':\n'
    '            embed_dim = 256\n'
    "            if 'dinov2_b14' in self.config['which_model']:\n"
    '                embed_dim = 768\n'
    "            elif 'dinov2_l14' in self.config['which_model']:\n"
    '                embed_dim = 1024\n'
    '            self.cls_unit = MultiProtoAsConv(proto_grid=[proto_hw, proto_hw], feature_hw=self.config["feature_hw"], embed_dim=embed_dim)  # when treating it as ordinary prototype\n'
    '            print(f"cls unit feature hw: {self.cls_unit.feature_hw}")\n'
    "        elif self.config['cls_name'].startswith('spen'):\n"
    '            embed_dim = 1024\n'
    "            if 'dinov2_b14' in self.config['which_model']:\n"
    '                embed_dim = 768\n'
    "            elif 'dinov2_l14' in self.config['which_model']:\n"
    '                embed_dim = 1024\n'
    "            _spen_mode = self.config['cls_name']  # spen_alpg, spen_qlpe, or spen_full\n"
    '            self.cls_unit = SPENProtoMatcher(\n'
    '                proto_grid=[proto_hw, proto_hw],\n'
    '                feature_hw=self.config["feature_hw"],\n'
    '                embed_dim=embed_dim,\n'
    '                spen_mode=_spen_mode,\n'
    '                k_max=proto_hw * proto_hw,  # use proto_grid_size^2 as k_max\n'
    '                Cs=200,\n'
    '            )\n'
    '            print(f"SPEN cls unit: mode={_spen_mode}, k_max={proto_hw*proto_hw}")\n'
    '        else:\n'
    '            raise NotImplementedError(\n'
    '                f\'Backbone network {self.config["which_model"]} not implemented\')'
)

if old_cls in code:
    code = code.replace(old_cls, new_cls, 1)
    changes += 1
    print("  [+] Added SPEN branch to get_cls()")
else:
    print("  [!] Could not find get_cls() marker")
    # Try a simpler approach: just add before the else/raise
    old_else = (
        "        else:\n"
        "            raise NotImplementedError(\n"
        "                f'Backbone network {self.config[\"which_model\"]} not implemented')"
    )
    spen_branch = (
        "        elif self.config['cls_name'].startswith('spen'):\n"
        "            embed_dim = 1024\n"
        "            if 'dinov2_b14' in self.config['which_model']:\n"
        "                embed_dim = 768\n"
        "            elif 'dinov2_l14' in self.config['which_model']:\n"
        "                embed_dim = 1024\n"
        "            _spen_mode = self.config['cls_name']\n"
        "            self.cls_unit = SPENProtoMatcher(\n"
        "                proto_grid=[proto_hw, proto_hw],\n"
        "                feature_hw=self.config['feature_hw'],\n"
        "                embed_dim=embed_dim,\n"
        "                spen_mode=_spen_mode,\n"
        "                k_max=proto_hw * proto_hw,\n"
        "                Cs=200,\n"
        "            )\n"
        "            print(f'SPEN cls unit: mode={_spen_mode}, k_max={proto_hw*proto_hw}')\n"
        "        else:\n"
        "            raise NotImplementedError(\n"
        "                f'Backbone network {self.config[\"which_model\"]} not implemented')"
    )
    if old_else in code:
        code = code.replace(old_else, spen_branch, 1)
        changes += 1
        print("  [+] Added SPEN branch (fallback method)")
    else:
        print("  [!] Fallback also failed")

# 3. Handle FG_PROT_MODE check: SPEN modes always use 'mask' mode internally
#    The forward() checks avg_pool to decide fg_mode. SPEN doesn't use avg_pool
#    so we need to bypass that check.
old_fg_check = (
    "                        fg_mode = FG_PROT_MODE if F.avg_pool2d(_msk, k_size).max(\n"
    "                        ) >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask'"
)
new_fg_check = (
    "                        _is_spen = self.config.get('cls_name', '').startswith('spen')\n"
    "                        if _is_spen:\n"
    "                            fg_mode = 'mask'  # SPEN handles its own prototype generation\n"
    "                        else:\n"
    "                        fg_mode = FG_PROT_MODE if F.avg_pool2d(_msk, k_size).max(\n"
    "                        ) >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask'"
)
if old_fg_check in code:
    code = code.replace(old_fg_check, new_fg_check, 1)
    changes += 1
    print("  [+] Added SPEN bypass for FG_PROT_MODE check")
else:
    print("  [!] Could not find FG_PROT_MODE check (may have different formatting)")

with open(GRID_PROTO_PY, "w", encoding="utf-8") as f:
    f.write(code)

print(f"\nDone patching {GRID_PROTO_PY} ({changes} changes)")
print("\nIMPORTANT: Copy spen.py to models/ directory:")
print("  cp outputs/spen.py models/spen.py")
print("\nRun experiments:")
print("  ./outputs/run_ablation.sh spen_alpg liver mri 1 '' 0   # ALPG only")
print("  ./outputs/run_ablation.sh spen_full liver mri 1 '' 0   # ALPG + QLPE")
print("  ./outputs/run_ablation.sh spen_qlpe liver mri 1 '' 0   # QLPE only (fixed grid)")
print("\nTo restore: cp models/grid_proto_fewshot.py.bak models/grid_proto_fewshot.py")
