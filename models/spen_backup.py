# models/spen.py — Self-guided Prototype ENhancement for ALPNet
# Training-free: works with frozen DINOv2 backbone.
#
# Two modules:
#   ALPG: Adaptive Local Prototype Generation (FPS-based, replaces fixed grid)
#   QLPE: Query-guided Local Prototype Enhancement (Sinkhorn OT weighting)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def safe_norm(x, p=2, dim=1, eps=1e-4):
    x_norm = torch.norm(x, p=p, dim=dim)
    x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
    x = x.div(x_norm.unsqueeze(1).expand_as(x))
    return x


def masked_avg_pool(fts, mask):
    """Masked average pooling: fts [C, H, W], mask [H, W] -> [C]"""
    mask = mask.unsqueeze(0)
    pooled = torch.sum(fts * mask, dim=(-1, -2)) / (mask.sum(dim=(-1, -2)) + 1e-5)
    return pooled


def farthest_point_sample(points, k, coords=None, spatial_weight=0.3):
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

    return torch.tensor(selected, device=points.device)


def adaptive_local_prototypes(fts, mask, k_max=16, Cs=200):
    """
    ALPG: Adaptive Local Prototype Generation.
    Uses farthest point sampling to find k cluster centers in foreground,
    then assigns each foreground pixel to nearest center and averages.

    Args:
        fts:  [1, C, H, W] support features
        mask: [1, 1, H, W] binary foreground mask (already resized to fts size)
        k_max: maximum number of local prototypes
        Cs: area constant controlling k (k = mask_area / Cs)
    Returns:
        local_protos: [k, C] local prototype vectors (normalized)
        global_proto: [1, C] global prototype vector (normalized)
    """
    C, H, W = fts.shape[1], fts.shape[2], fts.shape[3]
    mask_flat = mask.reshape(-1)
    fts_flat = fts.reshape(C, -1).permute(1, 0)  # [HW, C]

    fg_idx = torch.where(mask_flat > 0.5)[0]
    n_fg = fg_idx.shape[0]

    if n_fg == 0:
        return None, None

    fg_fts = fts_flat[fg_idx]  # [n_fg, C]

    # Global prototype via masked average pooling
    global_proto = fg_fts.mean(dim=0, keepdim=True)  # [1, C]

    # Adaptive k
    k = min(max(int(n_fg / Cs), 1), k_max)

    if k == 1:
        local_protos = global_proto  # [1, C]
    else:
        # Build spatial coordinates for foreground pixels
        fg_hw = torch.stack([fg_idx // W, fg_idx % W], dim=1).float()  # [n_fg, 2]
        # Normalize to [0, 1]
        fg_hw[:, 0] /= max(H - 1, 1)
        fg_hw[:, 1] /= max(W - 1, 1)
        # FPS with spatial awareness
        center_idx = farthest_point_sample(fg_fts, k, coords=fg_hw, spatial_weight=0.3)
        centers = fg_fts[center_idx]  # [k, C]

        # Assign each foreground pixel to nearest center
        # dist: [n_fg, k]
        dists = torch.cdist(fg_fts, centers)  # [n_fg, k]
        assignments = dists.argmin(dim=1)  # [n_fg]

        # Average features within each cluster
        local_protos = []
        for j in range(k):
            cluster_mask = (assignments == j)
            if cluster_mask.sum() > 0:
                proto = fg_fts[cluster_mask].mean(dim=0)
            else:
                proto = centers[j]
            local_protos.append(proto)
        local_protos = torch.stack(local_protos, dim=0)  # [k, C]

    return safe_norm(local_protos), safe_norm(global_proto)


def sinkhorn(cost_matrix, n_iter=10, eps=0.1):
    """
    Sinkhorn algorithm for optimal transport.
    Args:
        cost_matrix: [m, n] cost matrix (lower = better match)
        n_iter: number of Sinkhorn iterations
        eps: regularization strength
    Returns:
        transport_plan: [m, n] optimal transport matrix
    """
    m, n = cost_matrix.shape
    if m == 0 or n == 0:
        return cost_matrix

    # Uniform marginals
    mu = torch.ones(m, device=cost_matrix.device) / m
    nu = torch.ones(n, device=cost_matrix.device) / n

    # Kernel matrix (Gibbs distribution)
    K = torch.exp(-cost_matrix / eps)

    # Sinkhorn iterations
    u = torch.ones(m, device=cost_matrix.device) / m
    v = torch.ones(n, device=cost_matrix.device) / n

    for _ in range(n_iter):
        u = mu / (K @ v + 1e-8)
        v = nu / (K.t() @ u + 1e-8)

    transport_plan = torch.diag(u) @ K @ torch.diag(v)
    return transport_plan


def query_guided_weighting(supp_local, qry_local, sim_scale=20.0):
    """
    QLPE: Query-guided Local Prototype Enhancement.
    Uses optimal transport to weight support local prototypes by their
    relevance to query local prototypes.

    Args:
        supp_local: [m, C] support local prototypes (normalized)
        qry_local:  [n, C] query local prototypes (normalized)
        sim_scale:  scaling factor for cosine similarity
    Returns:
        weights: [m] importance weight for each support prototype
    """
    if supp_local is None or qry_local is None:
        return None
    if supp_local.shape[0] == 0 or qry_local.shape[0] == 0:
        return None

    # Similarity matrix: cosine similarity [m, n]
    sim = torch.mm(supp_local, qry_local.t())  # [m, n]

    # Cost matrix: 1 - similarity
    cost = 1.0 - sim

    # Solve OT
    transport_plan = sinkhorn(cost, n_iter=10, eps=0.1)

    # Weight = sum(T * S, axis=1) — intersection of transport plan and similarity
    weights = torch.sum(transport_plan * sim, dim=1)  # [m]

    # Normalize weights to [0, 1]
    if weights.max() > 1e-8:
        weights = weights / weights.max()

    return weights


class SPENProtoMatcher(nn.Module):
    """
    Self-guided Prototype ENhancement matcher.
    Replaces MultiProtoAsConv when cls_name == 'spen'.

    Modes:
        'spen_alpg': ALPG only (adaptive prototypes, uniform weighting)
        'spen_qlpe': fixed grid + QLPE (OT weighting on existing prototypes)
        'spen_full': ALPG + QLPE (both modules)
    """
    def __init__(self, proto_grid, feature_hw, embed_dim=1024, spen_mode='spen_full', k_max=16, Cs=200):
        super().__init__()
        self.feature_hw = feature_hw
        self.proto_grid = proto_grid
        self.kernel_size = [ft_l // grid_l for ft_l, grid_l in zip(feature_hw, proto_grid)]
        self.avg_pool_op = nn.AvgPool2d(self.kernel_size)
        self.spen_mode = spen_mode
        self.k_max = k_max
        self.Cs = Cs
        self.embed_dim = embed_dim
        print(f"SPENProtoMatcher: mode={spen_mode}, k_max={k_max}, Cs={Cs}, kernel_size={self.kernel_size}")

    def get_prototypes_gridconv(self, sup_x, sup_y, thresh=0.95):
        """Original grid-based prototype generation (fallback for spen_qlpe mode)."""
        nch = sup_x.shape[1]
        n_sup_x = self.avg_pool_op(sup_x)
        sup_nshot = sup_x.shape[0]
        n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)
        n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)
        sup_y_g = self.avg_pool_op(sup_y)

        proto_grid = sup_y_g.clone().detach()
        proto_grid[proto_grid < thresh] = 0
        sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)
        protos = n_sup_x[sup_y_g > thresh, :]

        glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)
        pro_n = safe_norm(torch.cat([protos, glb_proto], dim=0))
        return pro_n, glb_proto

    def get_prediction_gridconv(self, prototypes, query):
        """Original ALPNet gridconv prediction: conv2d + softmax_weighted_sum."""
        dists = F.conv2d(query, prototypes[..., None, None]) * 20.0
        pred = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)
        return pred

    def get_prediction_from_prototypes(self, prototypes, query):
        """
        Compute prediction from prototypes and query features.
        Args:
            prototypes: [K, C] normalized prototype vectors
            query: [1, C, H, W] normalized query features
        Returns:
            pred: [1, 1, H, W] prediction score
        """
        # cosine similarity: [1, K, H, W]
        sim = F.cosine_similarity(query, prototypes[..., None, None], dim=1, eps=1e-4) * 20.0
        # take max over prototypes
        pred = sim.max(dim=0)[0].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        return pred

    def get_prediction_from_weighted_prototypes(self, local_protos, global_proto, weights, query):
        """
        Compute prediction using weighted prototype fusion.
        p_fused = global_proto + sum(local_proto_i * weight_i)
        Then cosine similarity with query.

        Args:
            local_protos: [k, C] normalized local prototypes
            global_proto: [1, C] normalized global prototype
            weights: [k] importance weights (or None for uniform)
            query: [1, C, H, W] normalized query features
        Returns:
            pred: [1, 1, H, W]
        """
        if weights is not None:
            weighted_local = local_protos * weights.unsqueeze(1)  # [k, C]
        else:
            weighted_local = local_protos

        # Fuse: global + weighted local average
        fused = global_proto + weighted_local.mean(dim=0, keepdim=True)  # [1, C]
        fused = safe_norm(fused)

        # Also compute per-local-prototype similarity and take max
        # This preserves the multi-granularity matching
        all_protos = torch.cat([fused, local_protos * (weights.unsqueeze(1) if weights is not None else 1)], dim=0)
        all_protos = safe_norm(all_protos)

        sim = F.cosine_similarity(query, all_protos[..., None, None], dim=1, eps=1e-4) * 20.0
        pred = sim.max(dim=0)[0].unsqueeze(0).unsqueeze(0)
        return pred

    def forward(self, qry, sup_x, sup_y, mode='gridconv+', thresh=0.95, isval=False, val_wsize=None, vis_sim=False, **kwargs):
        """
        Main forward pass.

        Args:
            qry: [way(1), nb(1), nc, h, w]
            sup_x: [way(1), shot, nb(1), nc, h, w]
            sup_y: [way(1), shot, nb(1), h, w]
        Returns:
            pred_grid: [1, 1, h, w]
            debug_assign: placeholder
            vis_dict: visualization dict
            proto_grid: placeholder
        """
        qry = qry.squeeze(1)     # [1, C, H, W]
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
        if self.spen_mode == 'spen_qlpe':
            # Fixed grid prototypes + OT weighting
            protos, glb_proto = self.get_prototypes_gridconv(supp_fts, supp_mask, thresh)
            # For QLPE, we need query local prototypes
            # Use global proto to get coarse query mask
            qry_sim = F.cosine_similarity(qry_n, safe_norm(glb_proto)[..., None, None], dim=1, eps=1e-4)
            qry_coarse_mask = (qry_sim > 0.5).float()  # [1, H, W]
            qry_local, qry_global = adaptive_local_prototypes(qry_fts, qry_coarse_mask.unsqueeze(0), self.k_max, self.Cs)
            # Split protos into local + global (last one is global)
            supp_local = protos[:-1] if protos.shape[0] > 1 else protos
            supp_global = protos[-1:]
            weights = query_guided_weighting(supp_local, qry_local)
            if weights is not None:
                pred = self.get_prediction_from_weighted_prototypes(supp_local, supp_global, weights, qry_n)
            else:
                pred = self.get_prediction_from_prototypes(protos, qry_n)
        elif self.spen_mode in ('spen_alpg', 'spen_full'):
            # Adaptive local prototypes
            supp_local, supp_global = adaptive_local_prototypes(supp_fts, supp_mask, self.k_max, self.Cs)
            if supp_local is None:
                # No foreground in support mask — return empty prediction
                pred = torch.zeros(1, 1, qry_fts.shape[2], qry_fts.shape[3], device=qry_fts.device)
                return pred, vis_dict['proto_assign'], vis_dict, None

            if self.spen_mode == 'spen_full':
                # QLPE: generate query local prototypes and weight support prototypes
                qry_sim = F.cosine_similarity(qry_n, supp_global[..., None, None], dim=1, eps=1e-4)
                qry_coarse_mask = (qry_sim > 0.5).float()  # [1, H, W]
                qry_local, qry_global = adaptive_local_prototypes(qry_fts, qry_coarse_mask.unsqueeze(0), self.k_max, self.Cs)
                weights = query_guided_weighting(supp_local, qry_local)
            else:
                weights = None

            pred = self.get_prediction_from_weighted_prototypes(supp_local, supp_global, weights, qry_n)
        else:
            # Fallback: original gridconv+
            protos, glb_proto = self.get_prototypes_gridconv(supp_fts, supp_mask, thresh)
            pred = self.get_prediction_from_prototypes(protos, qry_n)

        if vis_sim:
            vis_dict['raw_local_sims'] = pred.clone().detach()

        return pred, vis_dict['proto_assign'], vis_dict, None
