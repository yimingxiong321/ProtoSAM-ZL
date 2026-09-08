"""Training-free dense foreground matching ablations.

These matchers are inspired by dense support/query matching methods such as
DCAMA, but deliberately contain no learned attention or cost aggregation.  The
background branch is delegated to the unmodified ALPNet matcher; only the
foreground representation is changed.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .alpmodule import MultiProtoAsConv


class DenseForegroundMatcher(nn.Module):
    """Replace foreground grid averages with dense support-token evidence.

    ``hard`` modes keep only support feature cells whose area-downsampled
    foreground occupancy exceeds ``hard_threshold``. ``soft`` modes retain
    every cell and use its continuous occupancy as a probability weight.

    The original modes use normalized LogSumExp. Modes ending in ``_attn`` use
    the ALPNet-compatible softmax-weighted expected cosine similarity instead.
    """

    def __init__(
        self,
        proto_grid,
        feature_hw,
        embed_dim=768,
        dense_mode="dense_fg_soft",
        temperature=0.05,
        hard_threshold=0.95,
        score_scale=20.0,
        query_chunk_size=512,
        global_anchor=True,
        null_strength=1.0,
        eps=1e-6,
    ):
        super().__init__()
        valid_modes = {
            "dense_fg_hard", "dense_fg_soft",
            "dense_fg_hard_attn", "dense_fg_soft_attn",
            "dense_fg_hard_attn_null",
        }
        if dense_mode not in valid_modes:
            raise ValueError(f"Unknown dense foreground mode: {dense_mode}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        # Reuse the original implementation verbatim for background matching.
        self.baseline = MultiProtoAsConv(
            proto_grid=proto_grid,
            feature_hw=feature_hw,
            embed_dim=embed_dim,
        )
        self.kernel_size = self.baseline.kernel_size
        self.feature_hw = feature_hw
        self.dense_mode = dense_mode
        self.aggregation = (
            "attention_expectation" if "_attn" in dense_mode
            else "normalized_lse"
        )
        self.temperature = float(temperature)
        self.hard_threshold = float(hard_threshold)
        self.score_scale = float(score_scale)
        self.query_chunk_size = int(query_chunk_size)
        self.global_anchor = bool(global_anchor)
        self.use_background_null = dense_mode.endswith("_null")
        self.null_strength = float(null_strength)
        if self.null_strength < 0:
            raise ValueError("null_strength must be non-negative")
        self.eps = float(eps)

    def _occupancy(self, full_res_mask, fallback_mask, size, dtype, device):
        mask = full_res_mask if full_res_mask is not None else fallback_mask
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, device=device)
        mask = mask.to(device=device, dtype=dtype)
        while mask.ndim > 4 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            mask = mask[:, None]
        elif mask.ndim != 4:
            raise ValueError(f"Unsupported support mask shape: {tuple(mask.shape)}")
        # Area interpolation preserves partial foreground occupancy at boundary
        # feature cells, unlike nearest-neighbour interpolation.
        return F.interpolate(mask, size=size, mode="area").clamp_(0.0, 1.0)

    def _weighted_lse(self, query_tokens, support_tokens, weights):
        """Normalized weighted log-sum-exp over support tokens, in chunks."""
        tau = self.temperature
        log_weights = torch.log(weights.clamp_min(self.eps))
        log_mass = torch.log(weights.sum().clamp_min(self.eps))
        chunks = []
        for start in range(0, query_tokens.shape[0], self.query_chunk_size):
            query_chunk = query_tokens[start:start + self.query_chunk_size]
            correlation = query_chunk @ support_tokens.transpose(0, 1)
            evidence = tau * (
                torch.logsumexp(correlation / tau + log_weights[None], dim=1)
                - log_mass
            )
            chunks.append(evidence)
        return torch.cat(chunks, dim=0)

    def _weighted_attention_expectation(self, query_tokens, support_tokens, weights):
        """Softmax-weighted expected cosine, matching ALPNet score semantics.

        Occupancy acts only as an attention prior. The returned value remains
        in cosine-similarity units and receives ``score_scale`` later, exactly
        like the baseline prototype similarities.
        """
        expectation, _, _ = self._weighted_attention_statistics(
            query_tokens, support_tokens, weights
        )
        return expectation

    def _weighted_attention_statistics(self, query_tokens, support_tokens, weights):
        """Return expected cosine, normalized log evidence, and entropy.

        The log evidence uses a normalized support prior. Consequently,
        duplicating an identical support token cannot make foreground evidence
        stronger merely by increasing the number of tokens.
        """
        tau = self.temperature
        log_weights = torch.where(
            weights > 0,
            torch.log(weights.clamp_min(self.eps)),
            torch.full_like(weights, float("-inf")),
        )
        log_mass = torch.log(weights.sum().clamp_min(self.eps))
        nonzero_count = int((weights > 0).sum().detach().cpu())
        entropy_scale = max(float(torch.log(torch.tensor(max(nonzero_count, 2))).item()), self.eps)
        expectation_chunks = []
        evidence_chunks = []
        entropy_chunks = []
        for start in range(0, query_tokens.shape[0], self.query_chunk_size):
            query_chunk = query_tokens[start:start + self.query_chunk_size]
            correlation = query_chunk @ support_tokens.transpose(0, 1)
            attention_logits = correlation / tau + log_weights[None]
            attention = F.softmax(attention_logits, dim=1)
            expectation_chunks.append((attention * correlation).sum(dim=1))
            evidence_chunks.append(
                torch.logsumexp(attention_logits - log_mass, dim=1)
            )
            entropy = -(attention * torch.log(attention.clamp_min(self.eps))).sum(dim=1)
            entropy_chunks.append(entropy / entropy_scale)
        return (
            torch.cat(expectation_chunks, dim=0),
            torch.cat(evidence_chunks, dim=0),
            torch.cat(entropy_chunks, dim=0),
        )

    def _dense_foreground(
        self, qry, sup_x, sup_y, full_res_mask, vis_sim, background_score=None,
    ):
        # Match MultiProtoAsConv's public input shapes.
        qry = qry.squeeze(1)                         # [Bq, C, Hq, Wq]
        sup_x = sup_x.squeeze(0).squeeze(1)          # [Bs, C, Hs, Ws]
        sup_y = sup_y.squeeze(0).reshape(
            sup_x.shape[0], 1, sup_x.shape[-2], sup_x.shape[-1]
        )
        if qry.ndim != 4 or sup_x.ndim != 4:
            raise ValueError(
                f"Dense matcher expects 2-D feature maps, got qry={tuple(qry.shape)}, "
                f"support={tuple(sup_x.shape)}"
            )

        occupancy = self._occupancy(
            full_res_mask, sup_y, sup_x.shape[-2:], sup_x.dtype, sup_x.device
        )
        if occupancy.shape[0] == 1 and sup_x.shape[0] > 1:
            occupancy = occupancy.expand(sup_x.shape[0], -1, -1, -1)

        raw_support_tokens = sup_x.permute(0, 2, 3, 1).reshape(-1, sup_x.shape[1])
        soft_weights = occupancy.reshape(-1)
        if "_hard" in self.dense_mode:
            weights = (soft_weights > self.hard_threshold).to(soft_weights.dtype)
        else:
            weights = soft_weights

        # In the hard ablation, absence of >threshold tokens must fall back to
        # the baseline global prototype, not silently change into the soft mode.
        fallback_used = bool(weights.sum() <= self.eps)

        support_tokens = F.normalize(raw_support_tokens, dim=1, eps=self.eps)
        query_tokens = F.normalize(
            qry.permute(0, 2, 3, 1).reshape(-1, qry.shape[1]), dim=1, eps=self.eps
        )
        dense_similarity = None
        foreground_log_evidence = None
        attention_entropy = None
        if not fallback_used:
            if self.aggregation == "attention_expectation":
                dense_similarity, foreground_log_evidence, attention_entropy = (
                    self._weighted_attention_statistics(
                    query_tokens, support_tokens, weights
                    )
                )
            else:
                dense_similarity = self._weighted_lse(
                    query_tokens, support_tokens, weights
                )

        component_scores = [] if dense_similarity is None else [dense_similarity]
        global_similarity = None
        if self.global_anchor:
            # Preserve ALPNet's effective global branch exactly: it uses the
            # existing nearest-neighbour feature-resolution mask, not the new
            # continuous area occupancy used by P2's dense local evidence.
            anchor_weights = sup_y.reshape(-1).to(raw_support_tokens.dtype)
            global_proto = (raw_support_tokens * anchor_weights[:, None]).sum(dim=0)
            global_proto = global_proto / anchor_weights.sum().clamp_min(self.eps)
            global_proto = F.normalize(global_proto, dim=0, eps=self.eps)
            global_similarity = query_tokens @ global_proto
            component_scores.append(global_similarity)

        if not component_scores:
            component_scores.append(torch.zeros_like(query_tokens[:, 0]))

        components = torch.stack(component_scores, dim=1) * self.score_scale
        score = (F.softmax(components, dim=1) * components).sum(dim=1)
        foreground_match_probability = None
        null_penalty = None
        if self.use_background_null:
            if background_score is None:
                raise ValueError(
                    "dense_fg_hard_attn_null requires the background score"
                )
            background_similarity = (
                background_score.reshape(-1).to(score) / self.score_scale
            )
            if foreground_log_evidence is None:
                if global_similarity is None:
                    foreground_log_evidence = torch.full_like(
                        background_similarity, float("-inf")
                    )
                else:
                    foreground_log_evidence = global_similarity / self.temperature
            background_log_evidence = background_similarity / self.temperature
            foreground_match_probability = torch.sigmoid(
                foreground_log_evidence - background_log_evidence
            )
            # Product-of-evidence calibration: retain the original foreground
            # score, but penalize it when the dense support bank cannot beat the
            # existing background evidence. There is no learned projection or
            # decoder; null_strength=1 is the parameter-free default.
            null_penalty = self.null_strength * torch.log(
                foreground_match_probability.clamp_min(self.eps)
            )
            score = score + null_penalty
        bq, _, hq, wq = qry.shape
        score_map = score.reshape(bq, 1, hq, wq)
        assignment = components.argmax(dim=1).reshape(bq, hq, wq).float()
        vis_dict = {
            "proto_assign": assignment.detach(),
            "dense_similarity": (
                dense_similarity.reshape(bq, 1, hq, wq).detach()
                if dense_similarity is not None else None
            ),
            "global_similarity": (
                global_similarity.reshape(bq, 1, hq, wq).detach()
                if global_similarity is not None else None
            ),
            "dense_occupancy": occupancy.detach(),
            "dense_effective_token_count": float(weights.sum().detach().cpu()),
            "dense_nonzero_token_count": int((weights > 0).sum().detach().cpu()),
            "dense_hard_fallback": fallback_used,
            "dense_temperature": self.temperature,
            "dense_aggregation": self.aggregation,
            "dense_uses_background_null": self.use_background_null,
            "dense_null_strength": self.null_strength,
            "dense_foreground_match_probability": (
                foreground_match_probability.reshape(bq, 1, hq, wq).detach()
                if foreground_match_probability is not None else None
            ),
            "dense_null_penalty": (
                null_penalty.reshape(bq, 1, hq, wq).detach()
                if null_penalty is not None else None
            ),
            "dense_attention_entropy": (
                attention_entropy.reshape(bq, 1, hq, wq).detach()
                if attention_entropy is not None else None
            ),
        }
        if vis_sim:
            # Avoid materializing QxS in debug output; the aggregated dense map is
            # the quantity used for prediction and scales linearly with image size.
            vis_dict["raw_local_sims"] = components.transpose(0, 1).reshape(
                len(component_scores), bq, hq, wq
            ).permute(1, 0, 2, 3).detach()
        return score_map, [assignment], vis_dict, occupancy.detach()

    def forward(
        self, qry, sup_x, sup_y, mode, thresh, isval=False, val_wsize=None,
        vis_sim=False, get_prototypes=False, full_res_mask=None,
        background_score=None, **kwargs,
    ):
        if mode == "gridconv":
            return self.baseline(
                qry, sup_x, sup_y, mode=mode, thresh=thresh, isval=isval,
                val_wsize=val_wsize, vis_sim=vis_sim,
                get_prototypes=get_prototypes, **kwargs,
            )
        return self._dense_foreground(
            qry, sup_x, sup_y, full_res_mask=full_res_mask, vis_sim=vis_sim,
            background_score=background_score,
        )
