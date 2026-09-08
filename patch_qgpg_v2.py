import os, shutil

SPEN_PY = "models/spen.py"

with open(SPEN_PY, 'r', encoding='utf-8') as f:
    code = f.read()
code = code.replace("\r\n", "\n").replace("\r", "\n")

if "gate_mode" in code and "keep_ratio" in code:
    print("Already patched. Restoring backup...")
    if os.path.exists(SPEN_PY + ".bak"):
        shutil.copy2(SPEN_PY + ".bak", SPEN_PY)
        with open(SPEN_PY, 'r', encoding='utf-8') as f:
            code = f.read()
        code = code.replace("\r\n", "\n").replace("\r", "\n")
    else:
        print("No backup, using current state.")

shutil.copy2(SPEN_PY, SPEN_PY + ".bak")
n = 0

old_sig = "bootstrap_mode='threshold', topk_ratio=0.2):"
new_sig = "bootstrap_mode='threshold', topk_ratio=0.2,\n                 gate_mode='hard', keep_ratio=0.5):"
if old_sig in code:
    code = code.replace(old_sig, new_sig, 1)
    n += 1; print("  [+] __init__ signature")
else:
    print("  [!] __init__ sig not found")

old_store = "        self.topk_ratio = topk_ratio\n"
new_store = "        self.topk_ratio = topk_ratio\n        self.gate_mode = gate_mode\n        self.keep_ratio = keep_ratio\n"
if old_store in code and "self.gate_mode" not in code:
    code = code.replace(old_store, new_store, 1)
    n += 1; print("  [+] __init__ storage")
else:
    print("  [!] __init__ store not found")

old_prt = 'f"topk_ratio={topk_ratio}")'
new_prt = 'f"topk_ratio={topk_ratio}, gate_mode={gate_mode}, keep_ratio={keep_ratio}")'
if old_prt in code and "gate_mode=" not in code:
    code = code.replace(old_prt, new_prt, 1)
    n += 1; print("  [+] print updated")
else:
    print("  [!] print not found")

new_method_lines = [
    "    def get_prediction_qgpg(self, prototypes, glb_proto, query, qry_fts):",
    "        \"\"\"QGPG v2: hard top-k prototype selection.\"\"\"",
    "        n_protos = prototypes.shape[0]",
    "        n_local = n_protos - 1",
    "        if n_local <= 0:",
    "            return self.get_prediction_gridconv(prototypes, query)",
    "        local_protos = prototypes[:n_local]",
    "        global_proto_vec = prototypes[-1:]",
    "        glb_n = safe_norm(global_proto_vec)",
    "        qry_sim = F.cosine_similarity(query, glb_n[..., None, None], dim=1, eps=1e-4)",
    "        if self.bootstrap_mode == 'topk':",
    "            flat_sim = qry_sim.reshape(-1)",
    "            ratio = min(max(float(self.topk_ratio), 0.0), 1.0)",
    "            k = max(int(flat_sim.numel() * ratio), 1)",
    "            topk_idx = torch.topk(flat_sim, k=k, largest=True).indices",
    "            qry_fg_mask = torch.zeros_like(flat_sim)",
    "            qry_fg_mask[topk_idx] = 1.0",
    "            qry_fg_mask = qry_fg_mask.reshape_as(qry_sim)",
    "        else:",
    "            qry_fg_mask = (qry_sim > self.bootstrap_thresh).float()",
    "        if qry_fg_mask.sum() < 1.0:",
    "            return self.get_prediction_gridconv(prototypes, query)",
    "        qry_fg_vector = torch.sum(qry_fts * qry_fg_mask.unsqueeze(0), dim=(-1, -2)) / (qry_fg_mask.sum() + 1e-5)",
    "        qry_fg_vector = safe_norm(qry_fg_vector)",
    "        gate_logits = F.cosine_similarity(qry_fg_vector, local_protos, dim=1, eps=1e-4)",
    "        if self.gate_mode == 'hard':",
    "            k_keep = max(int(n_local * self.keep_ratio), 1)",
    "            if k_keep >= n_local:",
    "                return self.get_prediction_gridconv(prototypes, query)",
    "            _, topk_idx = torch.topk(gate_logits, k=k_keep, largest=True)",
    "            kept_local = local_protos[topk_idx]",
    "            kept_protos = torch.cat([kept_local, global_proto_vec], dim=0)",
    "            dists = F.conv2d(query, kept_protos[..., None, None]) * 20.0",
    "            pred = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)",
    "        else:",
    "            gate = F.softmax(gate_logits * self.gate_tau, dim=0)",
    "            gate = gate / (gate.max() + 1e-8)",
    "            dists = F.conv2d(query, prototypes[..., None, None]) * 20.0",
    "            gate_full = torch.ones(n_protos, device=query.device)",
    "            gate_full[:n_local] = gate",
    "            dists = dists * gate_full[None, :, None, None]",
    "            pred = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)",
    "        return pred",
    "",
]
new_method = "\n".join(new_method_lines)

start_marker = "    def get_prediction_qgpg(self, prototypes, glb_proto, query, qry_fts):"
end_marker = "    def get_prediction_from_prototypes(self, prototypes, query):"
si = code.find(start_marker)
ei = code.find(end_marker)
if si != -1 and ei != -1 and ei > si:
    code = code[:si] + new_method + code[ei:]
    n += 1; print("  [+] get_prediction_qgpg replaced")
else:
    print("  [!] qgpg boundaries not found")

with open(SPEN_PY, 'w', encoding='utf-8') as f:
    f.write(code)
print(f"Done: {n} changes. Backup: {SPEN_PY}.bak")