# models/spen.py — Self-guided Prototype ENhancement for ALPNet
# Training-free: works with frozen DINOv2 backbone.
#
# Modules:
#   ALPG: Adaptive Local Prototype Generation (FPS-based, replaces fixed grid)
#   QLPE: Query-guided Local Prototype Enhancement (Sinkhorn OT weighting)
#   QGPG: Query-Guided Prototype Gating (ALPNet grid + query gate)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def safe_norm(x, p=2, dim=1, eps=1e-4):
    x_norm = torch.norm(x, p=p, dim=dim)
    x_norm = torch.clamp(x_norm, min=eps)
    x = x.div(x_norm.unsqueeze(1).expand_as(x))
    return x


def masked_avg_pool(fts, mask):
    """Masked average pooling: fts [C, H, W], mask [H, W] -> [C]"""
    mask = mask.unsqueeze(0)
    pooled = torch.sum(fts * mask, dim=(-1, -2)) / (mask.sum(dim=(-1, -2)) + 1e-5)
    return pooled


def farthest_point_sample(coords, k, deterministic=True):
    """Spatial FPS used by SPENet LFG (paper Sec. 2.2).

    The paper clusters foreground *positions*, not feature vectors.  The first
    centre is sampled randomly and every following centre is the foreground
    coordinate farthest from the already selected set.
    """
    N = coords.shape[0]
    if N <= k:
        return torch.arange(N, device=coords.device)

    if deterministic:
        # Stable evaluation: start from the foreground point closest to the
        # spatial centroid.  Set deterministic=False to reproduce the paper's
        # random first centre during training/augmentation experiments.
        centroid = coords.mean(dim=0, keepdim=True)
        first = torch.sum((coords - centroid) ** 2, dim=1).argmin().item()
    else:
        first = torch.randint(N, (1,), device=coords.device).item()
    selected = [first]
    min_dists = torch.sum((coords - coords[first]) ** 2, dim=1)

    for _ in range(1, k):
        farthest = torch.argmax(min_dists)
        selected.append(farthest.item())
        new_dists = torch.sum((coords - coords[farthest]) ** 2, dim=1)
        min_dists = torch.minimum(min_dists, new_dists)

    return torch.tensor(selected, device=coords.device, dtype=torch.long)


def _sample_features_at_mask_pixels(fts, mask):
    """Sample a low-resolution feature map at foreground mask coordinates.

    SPENet resizes features to the mask resolution before LFG.  Sampling only
    foreground coordinates with ``grid_sample`` is mathematically equivalent
    to bilinear resizing followed by masking, while avoiding a prohibitively
    large C x H x W tensor for DINOv2 features.
    """
    if fts.shape[0] != 1:
        raise ValueError("ALPG expects one support/query image at a time")
    mask_2d = mask.reshape(-1, mask.shape[-2], mask.shape[-1])[0]
    mask_h, mask_w = mask_2d.shape
    fg_idx = torch.where(mask_2d.reshape(-1) > 0.5)[0]
    if fg_idx.numel() == 0:
        return fg_idx, None, None, (mask_h, mask_w), None

    fg_hw = torch.stack([fg_idx // mask_w, fg_idx % mask_w], dim=1).float()
    grid_y = 2.0 * fg_hw[:, 0] / max(mask_h - 1, 1) - 1.0
    grid_x = 2.0 * fg_hw[:, 1] / max(mask_w - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        fts, grid, mode='bilinear', padding_mode='border', align_corners=True
    )[0, :, :, 0].t().contiguous()
    coords = torch.stack([
        fg_hw[:, 0] / max(mask_h - 1, 1),
        fg_hw[:, 1] / max(mask_w - 1, 1),
    ], dim=1)
    return fg_idx, fg_hw, sampled, (mask_h, mask_w), coords


def adaptive_local_prototypes(fts, mask, k_max=24, Cs=50, return_debug=False,
                              reference_mask_size=256, scale_cs=True,
                              cap_by_feature_cells=True,
                              deterministic_fps=True):
    """Paper-faithful Adaptive Local Prototype Generation (ALPG)."""
    fg_idx, fg_hw_abs, fg_fts, mask_hw, coords = _sample_features_at_mask_pixels(fts, mask)
    n_fg = int(fg_idx.numel())

    if n_fg == 0:
        if return_debug:
            return None, None, {'fg_idx': fg_idx, 'fg_hw': None, 'center_idx': None,
                                'assignments': None, 'feature_hw': mask_hw,
                                'n_fg': 0, 'k': 0}
        return None, None

    global_proto = fg_fts.mean(dim=0, keepdim=True)
    # Eq. (3) uses Cs=50 at the paper's 256x256 resolution.  Preserve that
    # physical-area meaning when ProtoSAM runs at 448/672 instead of letting
    # every foreground saturate k_max merely because it has more pixels.
    if scale_cs:
        mask_area_scale = (mask_hw[0] * mask_hw[1]) / float(reference_mask_size ** 2)
    else:
        mask_area_scale = 1.0
    effective_cs = max(float(Cs) * mask_area_scale, 1.0)
    area_k = max(int(n_fg // effective_cs), 1)

    effective_feature_cells = fts.shape[-2] * fts.shape[-1]
    if cap_by_feature_cells:
        mask_2d = mask.reshape(-1, mask.shape[-2], mask.shape[-1])[0].float()
        feature_occupancy = F.interpolate(
            mask_2d[None, None], size=fts.shape[-2:], mode='area'
        )[0, 0]
        effective_feature_cells = max(int((feature_occupancy > 0.25).sum().item()), 1)

    k = min(area_k, int(k_max), n_fg, effective_feature_cells)
    center_idx = farthest_point_sample(coords, k, deterministic=deterministic_fps)
    center_coords = coords[center_idx]
    assignments = torch.cdist(coords, center_coords).argmin(dim=1)
    local_protos = []
    for j in range(k):
        region = assignments == j
        local_protos.append(fg_fts[region].mean(dim=0))
    local_protos = torch.stack(local_protos, dim=0)

    if return_debug:
        return local_protos, global_proto, {
            'fg_idx': fg_idx.detach(),
            'fg_hw': fg_hw_abs.detach(),
            'center_idx': center_idx.detach(),
            'assignments': assignments.detach(),
            'feature_hw': mask_hw,
            'n_fg': n_fg,
            'k': k,
            'area_k': area_k,
            'effective_cs': effective_cs,
            'effective_feature_cells': effective_feature_cells,
        }
    return local_protos, global_proto

def sinkhorn(cost_matrix, n_iter=50, eps=0.1):
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

    # The paper defines valid importance distributions but does not learn them;
    # use the standard uniform marginals and solve in log space for stability.
    work = cost_matrix.float()
    log_mu = work.new_full((m,), -math.log(m))
    log_nu = work.new_full((n,), -math.log(n))
    log_k = -work / eps
    log_u = torch.zeros_like(log_mu)
    log_v = torch.zeros_like(log_nu)
    for _ in range(n_iter):
        log_u = log_mu - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)
        log_v = log_nu - torch.logsumexp(log_k + log_u.unsqueeze(1), dim=0)
    return torch.exp(log_u.unsqueeze(1) + log_k + log_v.unsqueeze(0)).to(cost_matrix.dtype)


def query_guided_weighting(supp_local, qry_local, n_iter=50, eps=0.1,
                           return_debug=False):
    """
    QLPE: Query-guided Local Prototype Enhancement.
    Uses optimal transport to weight support local prototypes by their
    relevance to query local prototypes.

    Args:
        supp_local: [m, C] support local prototypes
        qry_local:  [n, C] query local prototypes
    Returns:
        weights: [m] importance weight for each support prototype
    """
    if supp_local is None or qry_local is None:
        return None
    if supp_local.shape[0] == 0 or qry_local.shape[0] == 0:
        return None

    # Similarity matrix S in Eq. (4).  Keep the MAP/LFG prototypes raw for
    # Eq. (6), and normalize only the copies used to compute cosine similarity.
    supp_n = F.normalize(supp_local, p=2, dim=1, eps=1e-4)
    qry_n = F.normalize(qry_local, p=2, dim=1, eps=1e-4)
    sim = torch.mm(supp_n, qry_n.t())  # [m, n]

    # Cost matrix: 1 - similarity
    cost = 1.0 - sim

    # Solve OT
    transport_plan = sinkhorn(cost, n_iter=n_iter, eps=eps)

    # Weight = sum(T * S, axis=1) — intersection of transport plan and similarity
    # Eq. (5): W* = sum(T* elementwise S, axis=1).
    weights = torch.sum(transport_plan * sim, dim=1)  # [m]

    if return_debug:
        return weights, {
            'prototype_similarity': sim.detach(),
            'transport_plan': transport_plan.detach(),
            'transport_row_sum': transport_plan.sum(dim=1).detach(),
            'transport_col_sum': transport_plan.sum(dim=0).detach(),
        }
    return weights


class SPENProtoMatcher(nn.Module):
    """
    Self-guided Prototype ENhancement matcher.
    Replaces MultiProtoAsConv when cls_name starts with 'spen'.

    Modes:
        'spen_alpg': ALPG only (adaptive prototypes, uniform weighting)
        'spen_qlpe': fixed grid + QLPE (OT weighting on existing prototypes)
        'spen_qlpe_multi': fixed grid + QLPE prior + multi-prototype matching
        'spen_full': ALPG + QLPE (both modules)
        'spen_alpg_multi': ALPG + ALPNet-style multi-prototype matching
        'spen_full_multi': ALPG + QLPE prior + multi-prototype matching
        'spen_qgpg': Query-Guided Prototype Gating (ALPNet grid + query gate)
    """
    def __init__(self, proto_grid, feature_hw, embed_dim=1024,
                 spen_mode='spen_full', k_max=24, Cs=50,
                 gate_tau=2.0, bootstrap_thresh=0.5,
                 bootstrap_mode='threshold', topk_ratio=0.2,
                  gate_mode='hard', keep_ratio=0.5,
                  sinkhorn_iters=50, sinkhorn_eps=0.1,
                  ot_prior_strength=1.0,
                  reference_mask_size=256, scale_cs=True,
                 cap_by_feature_cells=True, deterministic_fps=True):
        super().__init__()
        self.feature_hw = feature_hw
        self.proto_grid = proto_grid
        self.kernel_size = [ft_l // grid_l for ft_l, grid_l in zip(feature_hw, proto_grid)]
        self.avg_pool_op = nn.AvgPool2d(self.kernel_size)
        self.spen_mode = spen_mode
        self.k_max = k_max
        self.Cs = Cs
        self.embed_dim = embed_dim
        self.gate_tau = gate_tau
        self.bootstrap_thresh = bootstrap_thresh
        self.bootstrap_mode = bootstrap_mode
        self.topk_ratio = topk_ratio
        self.gate_mode = gate_mode
        self.keep_ratio = keep_ratio
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_eps = sinkhorn_eps
        self.ot_prior_strength = ot_prior_strength
        self.reference_mask_size = reference_mask_size
        self.scale_cs = scale_cs
        self.cap_by_feature_cells = cap_by_feature_cells
        self.deterministic_fps = deterministic_fps
        print(f"SPENProtoMatcher: mode={spen_mode}, k_max={k_max}, Cs={Cs}, "
              f"kernel_size={self.kernel_size}, gate_tau={gate_tau}, "
              f"bootstrap_thresh={bootstrap_thresh}, bootstrap_mode={bootstrap_mode}, "
              f"topk_ratio={topk_ratio}")

    def _alpg(self, fts, mask, return_debug=False):
        return adaptive_local_prototypes(
            fts, mask, self.k_max, self.Cs,
            return_debug=return_debug,
            reference_mask_size=self.reference_mask_size,
            scale_cs=self.scale_cs,
            cap_by_feature_cells=self.cap_by_feature_cells,
            deterministic_fps=self.deterministic_fps,
        )

    def get_prototypes_gridconv(self, sup_x, sup_y, thresh=0.95,
                                isval=False, val_wsize=None,
                                include_global=True):
        """Generate ALPNet grid prototypes with its exact validation window.

        ``include_global=False`` reproduces ALPNet's background ``gridconv``;
        ``include_global=True`` reproduces foreground ``gridconv+``.
        """
        nch = sup_x.shape[1]
        if isval and val_wsize is not None:
            n_sup_x = F.avg_pool2d(sup_x, val_wsize)
            sup_y_g = F.avg_pool2d(sup_y, val_wsize)
        else:
            n_sup_x = self.avg_pool_op(sup_x)
            sup_y_g = self.avg_pool_op(sup_y)
        sup_nshot = sup_x.shape[0]
        n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)
        n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

        proto_grid = sup_y_g.clone().detach()
        proto_grid[proto_grid < thresh] = 0
        sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)
        protos = n_sup_x[sup_y_g > thresh, :]

        glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)
        if include_global:
            protos = torch.cat([protos, glb_proto], dim=0)
        pro_n = safe_norm(protos)
        return pro_n, glb_proto

    def get_prediction_gridconv(self, prototypes, query):
        """Original ALPNet gridconv prediction: conv2d + softmax_weighted_sum."""
        dists = F.conv2d(query, prototypes[..., None, None]) * 20.0
        pred = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)
        return pred


    def get_prediction_qgpg(self, prototypes, glb_proto, query, qry_fts):
        """QGPG v2: hard top-k prototype selection."""
        n_protos = prototypes.shape[0]
        n_local = n_protos - 1
        if n_local <= 0:
            return self.get_prediction_gridconv(prototypes, query)
        local_protos = prototypes[:n_local]
        global_proto_vec = prototypes[-1:]
        glb_n = safe_norm(global_proto_vec)
        qry_sim = F.cosine_similarity(query, glb_n[..., None, None], dim=1, eps=1e-4)
        if self.bootstrap_mode == 'topk':
            flat_sim = qry_sim.reshape(-1)
            ratio = min(max(float(self.topk_ratio), 0.0), 1.0)
            k = max(int(flat_sim.numel() * ratio), 1)
            topk_idx = torch.topk(flat_sim, k=k, largest=True).indices
            qry_fg_mask = torch.zeros_like(flat_sim)
            qry_fg_mask[topk_idx] = 1.0
            qry_fg_mask = qry_fg_mask.reshape_as(qry_sim)
        else:
            qry_fg_mask = (qry_sim > self.bootstrap_thresh).float()
        if qry_fg_mask.sum() < 1.0:
            return self.get_prediction_gridconv(prototypes, query)
        qry_fg_vector = torch.sum(qry_fts * qry_fg_mask.unsqueeze(0), dim=(-1, -2)) / (qry_fg_mask.sum() + 1e-5)
        qry_fg_vector = safe_norm(qry_fg_vector)
        gate_logits = F.cosine_similarity(qry_fg_vector, local_protos, dim=1, eps=1e-4)
        if self.gate_mode == 'hard':
            k_keep = max(int(n_local * self.keep_ratio), 1)
            if k_keep >= n_local:
                return self.get_prediction_gridconv(prototypes, query)
            _, topk_idx = torch.topk(gate_logits, k=k_keep, largest=True)
            kept_local = local_protos[topk_idx]
            kept_protos = torch.cat([kept_local, global_proto_vec], dim=0)
            dists = F.conv2d(query, kept_protos[..., None, None]) * 20.0
            pred = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)
        else:
            gate = F.softmax(gate_logits * self.gate_tau, dim=0)
            gate = gate / (gate.max() + 1e-8)
            dists = F.conv2d(query, prototypes[..., None, None]) * 20.0
            gate_full = torch.ones(n_protos, device=query.device)
            gate_full[:n_local] = gate
            dists = dists * gate_full[None, :, None, None]
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
        """SPENet Eq. (6): fuse to one prototype, then match the query.

        Importantly, weighted local prototypes are not normalized individually
        and are not appended again for max matching; both behaviours would
        cancel/bypass QLPE weights and are absent from the paper.
        """
        if weights is not None:
            weighted_local = local_protos * weights.unsqueeze(1)  # [k, C]
        else:
            weighted_local = local_protos

        # p_gl_s = p_g_s + Avg(p_l_s * W*)
        fused = global_proto + weighted_local.mean(dim=0, keepdim=True)  # [1, C]
        fused = safe_norm(fused)
        sim = F.cosine_similarity(query, fused[..., None, None], dim=1, eps=1e-4) * 20.0
        pred = sim.unsqueeze(1)
        return pred

    def get_prediction_multi_prototypes(self, local_protos, global_proto,
                                        weights, query, return_debug=False,
                                        prototypes_normalized=False):
        """DINOv2 adaptation: preserve local prototypes during matching.

        This keeps ALPNet's successful per-pixel, multi-prototype aggregation.
        QLPE weights are used as log-priors on prototype selection instead of
        scaling then collapsing all feature vectors into one prototype.
        """
        prototypes = torch.cat([local_protos, global_proto], dim=0)
        if not prototypes_normalized:
            prototypes = F.normalize(prototypes, p=2, dim=1, eps=1e-4)
        dists = F.conv2d(query, prototypes[..., None, None]) * 20.0

        priors = dists.new_ones(prototypes.shape[0])
        if weights is not None and local_protos.shape[0] > 0:
            # Eq. (5) weights include the uniform OT row mass 1/K.  Recover
            # the transported expected cosine, map [-1, 1] to a bounded
            # reliability, then remove its common scale.  The lower bound
            # prevents OT from hard-deleting a prototype and repeating the
            # severe recall loss seen with single-prototype fusion.
            expected_cosine = weights.to(dists.dtype) * local_protos.shape[0]
            local_prior = ((expected_cosine + 1.0) * 0.5).clamp(0.05, 1.0)
            local_prior = local_prior / local_prior.mean().clamp_min(1e-6)
            priors[:local_protos.shape[0]] = local_prior

        attention_logits = dists
        if weights is not None and self.ot_prior_strength != 0:
            log_prior = priors.log()[None, :, None, None]
            attention_logits = dists + self.ot_prior_strength * log_prior

        attention = F.softmax(attention_logits, dim=1)
        pred = torch.sum(attention * dists, dim=1, keepdim=True)
        if return_debug:
            return pred, {
                'multi_proto_priors': priors.detach(),
                'multi_proto_dists': dists.detach(),
                'multi_proto_attention': attention.detach(),
            }
        return pred

    def _query_bootstrap(self, qry_n, global_proto, target_size):
        """Generate Mq* from the support global prototype (paper Sec. 2.1)."""
        qry_sim = F.cosine_similarity(
            qry_n, safe_norm(global_proto)[..., None, None], dim=1, eps=1e-4
        )
        if tuple(qry_sim.shape[-2:]) != tuple(target_size):
            qry_sim = F.interpolate(
                qry_sim.unsqueeze(1), size=target_size,
                mode='bilinear', align_corners=False
            ).squeeze(1)
        if self.bootstrap_mode == 'topk':
            flat = qry_sim.reshape(-1)
            ratio = min(max(float(self.topk_ratio), 0.0), 1.0)
            n_keep = max(int(flat.numel() * ratio), 1)
            idx = torch.topk(flat, k=n_keep, largest=True).indices
            mask = torch.zeros_like(flat)
            mask[idx] = 1.0
            mask = mask.reshape_as(qry_sim)
        else:
            mask = (qry_sim > self.bootstrap_thresh).float()
        return mask, qry_sim

    def forward(self, qry, sup_x, sup_y, mode='gridconv+', thresh=0.95,
                isval=False, val_wsize=None, vis_sim=False, **kwargs):
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

        supp_fts_all = sup_x
        supp_masks_all = sup_y
        # The foreground caller supplies one shot at a time.  Preserve all
        # shots for the ALPNet-compatible background branch.
        supp_fts = sup_x[0:1]
        supp_mask = sup_y[0:1]
        paper_mask = kwargs.get('full_res_mask', supp_mask)
        if paper_mask.dim() == 2:
            paper_mask = paper_mask.unsqueeze(0).unsqueeze(0)
        elif paper_mask.dim() == 3:
            paper_mask = paper_mask.unsqueeze(0)
        paper_mask = paper_mask.to(device=supp_fts.device, dtype=supp_fts.dtype)
        qry_fts = qry           # [1, C, H, W]

        qry_n = safe_norm(qry_fts)

        vis_dict = {'proto_assign': torch.zeros(1, qry_fts.shape[2], qry_fts.shape[3], device=qry_fts.device)}
        weights = None
        qry_coarse_mask = None
        spen_debug = None
        qlpe_debug = None
        qry_debug = None

        # CRITICAL FIX: Respect mode parameter.
        # mode='gridconv' = background call -> use original grid-based prototypes
        # mode='mask' = foreground call -> use SPEN ALPG prototypes
        if mode == 'gridconv':
            # Exact ALPNet background fallback: same shots, pooling window and
            # no extra global background prototype.
            protos, _ = self.get_prototypes_gridconv(
                supp_fts_all, supp_masks_all, thresh,
                isval=isval, val_wsize=val_wsize, include_global=False
            )
            pred = self.get_prediction_gridconv(protos, qry_n)
            if vis_sim:
                vis_dict['raw_local_sims'] = pred.clone().detach()
            return pred, vis_dict['proto_assign'], vis_dict, None

        # Foreground (mode='mask'): use SPEN modules
        if self.spen_mode == 'spen_qgpg':
            # QGPG: ALPNet grid prototypes + query-guided gating
            protos, glb_proto = self.get_prototypes_gridconv(
                supp_fts, supp_mask, thresh, isval=isval,
                val_wsize=val_wsize, include_global=True
            )
            if 0 in protos.shape:
                pred = torch.zeros(1, 1, qry_fts.shape[2], qry_fts.shape[3],
                                   device=qry_fts.device)
            else:
                pred = self.get_prediction_qgpg(protos, glb_proto, qry_n, qry_fts)

        elif self.spen_mode in ('spen_qlpe', 'spen_qlpe_multi'):
            # Fixed grid prototypes + OT weighting
            protos, glb_proto = self.get_prototypes_gridconv(
                supp_fts, supp_mask, thresh, isval=isval,
                val_wsize=val_wsize, include_global=True
            )
            # Split protos into local + global (last one is global)
            supp_local = protos[:-1]
            supp_global = protos[-1:]
            qry_coarse_mask, qry_bootstrap_sim = self._query_bootstrap(
                qry_n, supp_global, paper_mask.shape[-2:]
            )
            if vis_sim:
                qry_local, _, qry_debug = self._alpg(
                    qry_fts, qry_coarse_mask.unsqueeze(0), return_debug=True
                )
            else:
                qry_local, _ = self._alpg(qry_fts, qry_coarse_mask.unsqueeze(0))
            if qry_local is not None:
                if vis_sim:
                    weights, qlpe_debug = query_guided_weighting(
                        supp_local, qry_local, self.sinkhorn_iters,
                        self.sinkhorn_eps, return_debug=True
                    )
                else:
                    weights = query_guided_weighting(
                        supp_local, qry_local, self.sinkhorn_iters,
                        self.sinkhorn_eps
                    )
            if self.spen_mode == 'spen_qlpe_multi':
                if vis_sim:
                    pred, multi_debug = self.get_prediction_multi_prototypes(
                        supp_local, supp_global, weights, qry_n,
                        return_debug=True, prototypes_normalized=True
                    )
                    if qlpe_debug is None:
                        qlpe_debug = {}
                    qlpe_debug.update(multi_debug)
                else:
                    pred = self.get_prediction_multi_prototypes(
                        supp_local, supp_global, weights, qry_n,
                        prototypes_normalized=True
                    )
            elif weights is not None:
                pred = self.get_prediction_from_weighted_prototypes(supp_local, supp_global, weights, qry_n)
            else:
                pred = self.get_prediction_from_weighted_prototypes(supp_local, supp_global, None, qry_n)
        elif self.spen_mode in (
            'spen_alpg', 'spen_full',
            'spen_alpg_multi', 'spen_full_multi'
        ):
            # Adaptive local prototypes
            if vis_sim:
                supp_local, supp_global, spen_debug = self._alpg(
                    supp_fts, paper_mask, return_debug=True
                )
            else:
                supp_local, supp_global = self._alpg(supp_fts, paper_mask)
            if supp_local is None:
                # No foreground in support mask — return empty prediction
                pred = torch.zeros(1, 1, qry_fts.shape[2], qry_fts.shape[3], device=qry_fts.device)
                return pred, vis_dict['proto_assign'], vis_dict, None

            if self.spen_mode in ('spen_full', 'spen_full_multi'):
                # QLPE: generate query local prototypes and weight support prototypes
                qry_coarse_mask, qry_bootstrap_sim = self._query_bootstrap(
                    qry_n, supp_global, paper_mask.shape[-2:]
                )
                if vis_sim:
                    qry_local, _, qry_debug = self._alpg(
                        qry_fts, qry_coarse_mask.unsqueeze(0), return_debug=True
                    )
                else:
                    qry_local, _ = self._alpg(qry_fts, qry_coarse_mask.unsqueeze(0))
                if qry_local is not None:
                    if vis_sim:
                        weights, qlpe_debug = query_guided_weighting(
                            supp_local, qry_local, self.sinkhorn_iters,
                            self.sinkhorn_eps, return_debug=True
                        )
                    else:
                        weights = query_guided_weighting(
                            supp_local, qry_local, self.sinkhorn_iters,
                            self.sinkhorn_eps
                        )
            else:
                weights = None

            if self.spen_mode in ('spen_alpg_multi', 'spen_full_multi'):
                if vis_sim:
                    pred, multi_debug = self.get_prediction_multi_prototypes(
                        supp_local, supp_global, weights, qry_n,
                        return_debug=True
                    )
                    if qlpe_debug is None:
                        qlpe_debug = {}
                    qlpe_debug.update(multi_debug)
                else:
                    pred = self.get_prediction_multi_prototypes(
                        supp_local, supp_global, weights, qry_n
                    )
            else:
                pred = self.get_prediction_from_weighted_prototypes(
                    supp_local, supp_global, weights, qry_n
                )
        else:
            # Fallback: original gridconv+
            protos, glb_proto = self.get_prototypes_gridconv(
                supp_fts, supp_mask, thresh, isval=isval,
                val_wsize=val_wsize, include_global=True
            )
            pred = self.get_prediction_from_prototypes(protos, qry_n)

        if vis_sim:
            vis_dict['raw_local_sims'] = pred.clone().detach()
            vis_dict['sim'] = pred.clone().detach()
            if weights is not None:
                vis_dict['weights'] = weights.detach()
                scale = weights.detach().abs().max().clamp_min(1e-8)
                vis_dict['weights_relative'] = weights.detach() / scale
            if qry_coarse_mask is not None:
                vis_dict['qry_coarse_mask'] = qry_coarse_mask.detach()
                vis_dict['qry_bootstrap_sim'] = qry_bootstrap_sim.detach()
            if spen_debug is not None:
                vis_dict.update({f'spen_{k}': v for k, v in spen_debug.items()})
            if qry_debug is not None:
                vis_dict.update({f'qry_{k}': v for k, v in qry_debug.items()})
            if qlpe_debug is not None:
                vis_dict.update(qlpe_debug)

        return pred, vis_dict['proto_assign'], vis_dict, None
