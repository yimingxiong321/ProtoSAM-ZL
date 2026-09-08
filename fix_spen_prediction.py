# fix_spen_prediction.py - Fix SPEN's prediction aggregation to match original ALPNet
#
# Bug: SPEN uses cosine_similarity + max (winner-take-all)
#      Original ALPNet uses conv2d + softmax_weighted_sum (smooth blending)
#      This causes Recall to drop from 85% to 70%.
#
# Fix: Replace get_prediction_from_prototypes and get_prediction_from_weighted_prototypes
#      to use conv2d + softmax, matching the original gridconv/gridconv+ logic.

import re

fpath = 'models/spen.py'
with open(fpath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace get_prediction_from_prototypes
old_pred_proto = '''    def get_prediction_from_prototypes(self, prototypes, query):
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
        pred = sim.max(dim=1)[0].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        return pred'''

new_pred_proto = '''    def get_prediction_from_prototypes(self, prototypes, query):
        """
        Compute prediction from prototypes and query features.
        Uses conv2d + softmax_weighted_sum, matching original ALPNet gridconv+.
        Args:
            prototypes: [K, C] normalized prototype vectors
            query: [1, C, H, W] normalized query features
        Returns:
            pred: [1, 1, H, W] prediction score
        """
        # dot product via conv2d: [1, K, H, W] (cosine sim when both normalized)
        dists = F.conv2d(query, prototypes[..., None, None]) * 20.0
        # softmax-weighted sum (smooth blending, not winner-take-all)
        pred = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)
        return pred  # [1, 1, H, W]'''

if old_pred_proto in content:
    content = content.replace(old_pred_proto, new_pred_proto)
    print("Fixed get_prediction_from_prototypes")
else:
    print("WARNING: Could not find get_prediction_from_prototypes to replace")

# Replace get_prediction_from_weighted_prototypes
old_weighted = '''    def get_prediction_from_weighted_prototypes(self, local_protos, global_proto, weights, query):
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
        pred = sim.max(dim=1)[0].unsqueeze(0).unsqueeze(0)
        return pred'''

new_weighted = '''    def get_prediction_from_weighted_prototypes(self, local_protos, global_proto, weights, query):
        """
        Compute prediction using conv2d + softmax (matching original ALPNet gridconv+).
        Combines local prototypes and global prototype, all used as conv kernels.

        Args:
            local_protos: [k, C] normalized local prototypes
            global_proto: [1, C] normalized global prototype
            weights: [k] importance weights (or None for uniform)
            query: [1, C, H, W] normalized query features
        Returns:
            pred: [1, 1, H, W]
        """
        # Apply weights to local prototypes
        if weights is not None:
            weighted_local = local_protos * weights.unsqueeze(1)  # [k, C]
        else:
            weighted_local = local_protos

        # Concatenate: global + weighted local prototypes
        all_protos = torch.cat([global_proto, weighted_local], dim=0)  # [1+k, C]
        all_protos = safe_norm(all_protos)

        # conv2d + softmax_weighted_sum (same as original gridconv+)
        dists = F.conv2d(query, all_protos[..., None, None]) * 20.0  # [1, 1+k, H, W]
        pred = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)  # [1, 1, H, W]
        return pred'''

if old_weighted in content:
    content = content.replace(old_weighted, new_weighted)
    print("Fixed get_prediction_from_weighted_prototypes")
else:
    print("WARNING: Could not find get_prediction_from_weighted_prototypes to replace")

with open(fpath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\nFixed {fpath}")
print("Key change: cosine_similarity + max  ->  conv2d + softmax_weighted_sum")
print("This matches the original ALPNet gridconv+ prediction strategy.")
