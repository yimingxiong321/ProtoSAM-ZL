# fix_spen_branch.py - Fix two critical bugs that prevented SPEN from running.
#
# Run this on the SERVER from the ProtoSAM-main directory:
#   python fix_spen_branch.py
#
# Bug 1: SPEN elif branch was added to get_encoder() instead of get_cls()
#        -> When which_model='dinov2_l14', the dinov2_l14 branch matches first,
#           so the SPEN branch is dead code (never reached).
#        -> get_cls() has no SPEN branch, so cls_name='spen_alpg' would crash
#           with NotImplementedError.
#
# Bug 2: config_ssl_upload.py has no fallback to set clsname from ablation_mode
#        -> Even if run_ablation.sh fails to pass clsname=spen_alpg, the config
#           should override it.

import re
import sys

def fix_grid_proto():
    fpath = 'models/grid_proto_fewshot.py'
    with open(fpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the misplaced SPEN branch in get_encoder() and get_cls()
    # Strategy: identify the get_encoder and get_cls method boundaries,
    # then move the SPEN elif from get_encoder to get_cls.

    # Find method boundaries
    get_encoder_start = None
    get_cls_start = None
    for i, line in enumerate(lines):
        if 'def get_encoder(self)' in line:
            get_encoder_start = i
        if 'def get_cls(self)' in line:
            get_cls_start = i

    if get_encoder_start is None or get_cls_start is None:
        print("ERROR: Could not find get_encoder or get_cls methods")
        return False

    print(f"get_encoder starts at line {get_encoder_start+1}")
    print(f"get_cls starts at line {get_cls_start+1}")

    # Find the SPEN branch in get_encoder (between get_encoder_start and get_cls_start)
    spen_start = None
    spen_end = None
    for i in range(get_encoder_start, get_cls_start):
        if "cls_name'].startswith('spen')" in lines[i] and 'elif' in lines[i]:
            spen_start = i
            # Find the end of this elif block (next elif or else at same indent)
            indent = len(lines[i]) - len(lines[i].lstrip())
            for j in range(i+1, get_cls_start):
                stripped = lines[j].rstrip()
                if stripped and not stripped.startswith(' ' * (indent + 4)):
                    # This line is at the same or lower indent level
                    if stripped.startswith(' ' * indent + 'el') or stripped.startswith(' ' * indent + '}'):
                        spen_end = j
                        break
            if spen_end is None:
                spen_end = get_cls_start  # until end of get_encoder
            break

    if spen_start is None:
        print("SPEN branch not found in get_encoder() - may already be fixed")
    else:
        print(f"Found misplaced SPEN branch at lines {spen_start+1}-{spen_end}")
        # Extract the SPEN branch content
        spen_lines = lines[spen_start:spen_end]
        print(f"  Branch content ({len(spen_lines)} lines):")
        for l in spen_lines:
            print(f"    {l.rstrip()}")

        # Remove from get_encoder
        del lines[spen_start:spen_end]
        # Adjust get_cls_start since we removed lines
        removed = spen_end - spen_start
        get_cls_start -= removed

    # Now add SPEN branch to get_cls() before the else: raise
    # Find the else: raise in get_cls
    spen_insert = """        elif self.config['cls_name'].startswith('spen'):
            embed_dim = 1024
            if 'dinov2_b14' in self.config['which_model']:
                embed_dim = 768
            elif 'dinov2_l14' in self.config['which_model']:
                embed_dim = 1024
            _spen_mode = self.config['cls_name']
            self.cls_unit = SPENProtoMatcher(
                proto_grid=[proto_hw, proto_hw],
                feature_hw=self.config["feature_hw"],
                embed_dim=embed_dim,
                spen_mode=_spen_mode,
                k_max=proto_hw * proto_hw,
                Cs=200,
            )
            print(f'SPEN cls unit: mode={_spen_mode}, k_max={proto_hw*proto_hw}')
"""

    # Find "else:" followed by "raise NotImplementedError" in get_cls
    else_line = None
    for i in range(get_cls_start, min(get_cls_start + 30, len(lines))):
        if 'else:' in lines[i] and i > get_cls_start:
            # Check if next line has raise NotImplementedError
            if i+1 < len(lines) and 'NotImplementedError' in lines[i+1]:
                else_line = i
                break

    if else_line is None:
        print("ERROR: Could not find else: raise in get_cls()")
        return False

    # Check if SPEN branch already exists in get_cls
    already_has_spen = False
    for i in range(get_cls_start, else_line):
        if "cls_name'].startswith('spen')" in lines[i]:
            already_has_spen = True
            break

    if already_has_spen:
        print("SPEN branch already exists in get_cls() - skipping insertion")
    else:
        print(f"Inserting SPEN branch at line {else_line+1}")
        lines.insert(else_line, spen_insert)

    with open(fpath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"Fixed {fpath}")
    return True


def fix_config():
    fpath = 'config_ssl_upload.py'
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    old = """    # # ABLATION_V2_PATCHED
    ablation_mode = "none"
    ablation_fixes = None"""

    new = """    # # ABLATION_V2_PATCHED
    ablation_mode = "none"
    ablation_fixes = None
    # SPEN_FIX: if ablation_mode is a SPEN mode, override clsname
    # This is a fallback in case run_ablation.sh fails to pass clsname correctly
    if ablation_mode and str(ablation_mode).startswith('spen_'):
        clsname = ablation_mode"""

    if old in content:
        content = content.replace(old, new)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {fpath} - added SPEN override")
        return True
    else:
        # Check if already fixed
        if "SPEN_FIX" in content:
            print(f"{fpath} already has SPEN fix - skipping")
            return True
        print(f"WARNING: Could not find ablation_mode block in {fpath}")
        return False


def fix_run_ablation():
    fpath = 'run_ablation.sh'
    with open(fpath, 'rb') as f:
        raw = f.read()

    # Remove BOM if present
    if raw[:3] == b'\xef\xbb\xbf':
        raw = raw[3:]
        print("Removed BOM from run_ablation.sh")

    content = raw.decode('utf-8')

    # Replace bash-only [[ ]] with POSIX case statement
    old_block = """# SPEN modes use clsname=spen_* instead of grid_proto
if [[ "$MODE" == spen_* ]]; then
    CLSNAME="$MODE"
else
    CLSNAME="grid_proto"
fi"""

    new_block = """# SPEN modes use clsname=spen_* instead of grid_proto
# Using POSIX case for compatibility (not bash-only [[ ]])
case "$MODE" in
    spen_*) CLSNAME="$MODE" ;;
    *) CLSNAME="grid_proto" ;;
esac"""

    if old_block in content:
        content = content.replace(old_block, new_block)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {fpath} - replaced [[ ]] with case, removed BOM")
        return True
    elif 'case "$MODE"' in content:
        print(f"{fpath} already uses case statement - checking BOM only")
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {fpath} - removed BOM")
        return True
    else:
        # Just remove BOM
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"{fpath} - BOM removed, [[ ]] block not found (may differ)")
        return True


if __name__ == '__main__':
    print("=" * 60)
    print("Fixing SPEN integration bugs")
    print("=" * 60)

    ok1 = fix_grid_proto()
    ok2 = fix_config()
    ok3 = fix_run_ablation()

    print("\n" + "=" * 60)
    if ok1 and ok2 and ok3:
        print("All fixes applied successfully!")
        print("\nNow re-run the SPEN experiment:")
        print("  ./run_ablation.sh spen_alpg liver mri 1 '' 0")
        print("\nVerify the fix by checking the log for:")
        print("  'SPENProtoMatcher: mode=spen_alpg' (means SPEN is running)")
        print("  NOT 'MultiProtoAsConv' (means baseline is running)")
    else:
        print("Some fixes failed - check warnings above")
    print("=" * 60)
