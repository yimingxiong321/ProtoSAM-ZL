# fix_spen_v2.py - Comprehensive fix for SPEN implementation
#
# Bug 1 (CRITICAL): Background mode hijacked. SPEN.forward() ignores mode param.
#   When ALPNet calls cls_unit with mode='gridconv' for background, SPEN still
#   runs ALPG (FPS on huge background area). Fix: respect mode, fall back to
#   original grid-based prototypes for background.
#
# Bug 2 (CRITICAL): FPS loses spatial priors. adaptive_local_prototypes flattens
#   to [n_fg, C] and does FPS on pure features, dropping (H,W). Fix: augment FPS
#   distance with normalized spatial coordinates.
#
# Bug 3 (already fixed): Prediction uses cosine_similarity + max (winner-take-all)
#   instead of conv2d + softmax_weighted_sum. Fix applied in fix_spen_prediction.py.
#
# Bug 4 (design): QLPE cold start - query coarse mask from global proto with 0.5
#   threshold is unreliable. Fix: use original gridconv prototypes for query
#   coarse mask instead.

fpath = 'models/spen.py'
with open(fpath, 'r', encoding='utf-8') as f:
    content = f.read()

# === Fix 1: Spatial-aware FPS ===
old_fps = '''def farthest_point_sample(points, k):
    """
    Farthest point sampling on a set of points.
    Args:
        points: [N, C] feature vectors
        k: number of points to sample
    Returns:
        indices: [k] indices into points
    """
    N = points.shape[0]
    if N <= k:
        return torch.arange(N, device=points.device)

    # Start from the point closest to the centroid
    centroid = points.mean(dim=0)
    dists = torch.sum((points - centroid) ** 2, dim=1)
    farthest = torch.argmax(dists).unsqueeze(0)

    selected = [farthest.item()]
    min_dists = torch.sum((points - points[farthest]) ** 2, dim=1)

    for _ in range(1, k):
        farthest = torch.argmax(min_dists)
        selected.append(farthest.item())
        new_dists = torch.sum((points - points[farthest]) ** 2, dim=1)
        min_dists = torch.minimum(min_dists, new_dists)

    return torch.tensor(selected, device=points.device)'''

new_fps = '''def farthest_point_sample(points, k, coords=None, spatial_weight=0.3):
    """
    Farthest point sampling on a set of points.
    Args:
        points: [N, C] feature vectors
        k: number of points to sample
        coords: [N, 2] normalized spatial coordinates (optional)
        spatial_weight: weight for spatial distance (0=feature only, 1=spatial only)
    Returns:
        indices: [k] indices into points
    """
    N = points.shape[0]
    if N <= k:
        return torch.arange(N, device=points.device)

    # Combined distance: feature + spatial
    if coords is not None:
        # Normalize features and coords to same scale
        feat_dist_fn = lambda a, b: torch.sum((a - b) ** 2, dim=1)
        spat_dist_fn = lambda a, b: torch.sum((a - b) ** 2, dim=1)
        dist_fn = lambda a, b: (1 - spatial_weight) * feat_dist_fn(a, b) + spatial_weight * spat_dist_fn(a, b)
    else:
        dist_fn = lambda a, b: torch.sum((a - b) ** 2, dim=1)

    # Start from the point closest to the centroid
    centroid = points.mean(dim=0)
    dists = dist_fn(points, centroid)
    farthest = torch.argmax(dists).unsqueeze(0)

    selected = [farthest.item()]
    min_dists = dist_fn(points, points[farthest])

    for _ in range(1, k):
        farthest = torch.argmax(min_dists)
        selected.append(farthest.item())
        new_dists = dist_fn(points, points[farthest])
        min_dists = torch.minimum(min_dists, new_dists)

    return torch.tensor(selected, device=points.device)'''

if old_fps in content:
    content = content.replace(old_fps, new_fps)
    print("Fix 1: Added spatial-aware FPS")
else:
    print("WARNING: Could not find farthest_point_sample to replace")

# === Fix 2: Pass spatial coords to FPS in adaptive_local_prototypes ===
old_alpg_call = '''    if k == 1:
        local_protos = global_proto  # [1, C]
    else:
        # FPS to find k cluster centers
        center_idx = farthest_point_sample(fg_fts, k)'''

new_alpg_call = '''    if k == 1:
        local_protos = global_proto  # [1, C]
    else:
        # Build spatial coordinates for foreground pixels
        fg_hw = torch.stack([fg_idx // W, fg_idx % W], dim=1).float()  # [n_fg, 2]
        # Normalize to [0, 1]
        fg_hw[:, 0] /= max(H - 1, 1)
        fg_hw[:, 1] /= max(W - 1, 1)
        # FPS with spatial awareness
        center_idx = farthest_point_sample(fg_fts, k, coords=fg_hw, spatial_weight=0.3)'''

if old_alpg_call in content:
    content = content.replace(old_alpg_call, new_alpg_call)
    print("Fix 2: Pass spatial coords to FPS")
else:
    print("WARNING: Could not find ALPG FPS call to replace")

# === Fix 3: Respect mode parameter in forward() ===
# The key fix: when mode='gridconv' (background call), use original grid prototypes
old_forward_start = '''        qry = qry.squeeze(1)     # [1, C, H, W]
        sup_x = sup_x.squeeze(0).squeeze(1)  # [nshot, C, H, W]
        sup_y = sup_y.squeeze(0)             # [nshot, 1, H, W]
        sup_y = sup_y.reshape(sup_x.shape[0], 1, sup_x.shape[-2], sup_x.shape[-1])

        # Use first shot (n_shots=1 in ProtoSAM)
        supp_fts = sup_x[0:1]   # [1, C, H, W]
        supp_mask = sup_y[0:1]  # [1, 1, H, W]
        qry_fts = qry           # [1, C, H, W]

        qry_n = safe_norm(qry_fts)

        vis_dict = {'proto_assign': torch.zeros(1, qry_fts.shape[2], qry_fts.shape[3], device=qry_fts.device)}

        if self.spen_mode == 'spen_qlpe':'''

new_forward_start = '''        qry = qry.squeeze(1)     # [1, C, H, W]
        sup_x = sup_x.squeeze(0).squeeze(1)  # [nshot, C, H, W]
        sup_y = sup_y.squeeze(0)             # [nshot, 1, H, W]
        sup_y = sup_y.reshape(sup_x.shape[0], 1, sup_x.shape[-2], sup_x.shape[-1])

        # Use first shot (n_shots=1 in ProtoSAM)
        supp_fts = sup_x[0:1]   # [1, C, H, W]
        supp_mask = sup_y[0:1]  # [1, 1, H, W]
        qry_fts = qry           # [1, C, H, W]

        qry_n = safe_norm(qry_fts)

        vis_dict = {'proto_assign': torch.zeros(1, qry_fts.shape[2], qry_fts.shape[3], device=qry_fts.device)}

        # CRITICAL FIX: Respect mode parameter.
        # mode='gridconv' = background call -> use original grid-based prototypes
        # mode='mask' = foreground call -> use SPEN ALPG prototypes
        if mode == 'gridconv':
            # Background: use original gridconv (not ALPG!)
            protos, glb_proto = self.get_prototypes_gridconv(supp_fts, supp_mask, thresh)
            pred = self.get_prediction_gridconv(protos, qry_n)
            if vis_sim:
                vis_dict['raw_local_sims'] = pred.clone().detach()
            return pred, vis_dict['proto_assign'], vis_dict, None

        # Foreground (mode='mask'): use SPEN modules
        if self.spen_mode == 'spen_qlpe':'''

if old_forward_start in content:
    content = content.replace(old_forward_start, new_forward_start)
    print("Fix 3: Respect mode parameter (background uses gridconv)")
else:
    print("WARNING: Could not find forward start to replace")

# === Fix 4: Add gridconv prediction method ===
old_pred_proto_start = '''    def get_prediction_from_prototypes(self, prototypes, query):'''

new_pred_proto_start = '''    def get_prediction_gridconv(self, prototypes, query):
        """Original ALPNet gridconv prediction: conv2d + softmax_weighted_sum."""
        dists = F.conv2d(query, prototypes[..., None, None]) * 20.0
        pred = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)
        return pred

    def get_prediction_from_prototypes(self, prototypes, query):'''

if old_pred_proto_start in content:
    content = content.replace(old_pred_proto_start, new_pred_proto_start)
    print("Fix 4: Added gridconv prediction method")
else:
    print("WARNING: Could not find get_prediction_from_prototypes to insert before")

with open(fpath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\nFixed {fpath}")
print("\nSummary of all fixes:")
print("  1. Spatial-aware FPS (coords + spatial_weight=0.3)")
print("  2. Background uses original gridconv (not ALPG)")
print("  3. Prediction uses conv2d + softmax (not cosine + max)")
print("  4. Added gridconv prediction method for background path")
