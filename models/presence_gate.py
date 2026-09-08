"""Support-conditioned target-presence scores for ProtoSAM.

This module implements the P1 *score-only* experiment.  It deliberately does
not make a present/absent decision and does not claim conformal calibration.
The returned scores can be evaluated and calibrated offline on patient-wise
validation splits.
"""

from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PresenceScores:
    """Serializable image-level scores and extraction diagnostics."""

    max_probability: float
    topk_mean_probability: float
    predicted_area_ratio: float
    max_fg_bg_likelihood_ratio: float
    fg_bg_likelihood_ratio: float
    probability_srqs_strength: float
    probability_srqs_compactness: float
    probability_srqs_purity: float
    probability_srqs_score: float
    likelihood_srqs_strength: float
    likelihood_srqs_compactness: float
    likelihood_srqs_purity: float
    likelihood_srqs_score: float
    patchcore_candidate_mean_similarity: float
    patchcore_candidate_worst_similarity: float
    patchcore_support_coverage: float
    patchcore_bidirectional_score: float
    patchcore_candidate_count: int
    patchcore_largest_candidate_tokens: int
    foreground_token_count: int
    background_prototype_count: int
    foreground_core_fallback: bool
    background_core_fallback: bool

    def to_dict(self):
        return asdict(self)


class SimilarityResponseQualityScorer(nn.Module):
    """Training-free SRQS port from the public MedVeriSeg implementation.

    The source implementation scores a flattened square similarity map.  This
    adaptation accepts any 2-D response map, while preserving its top-k
    strength, weighted spatial compactness, and largest-component purity
    calculations.  It returns evidence only; thresholds remain an offline
    calibration decision.
    """

    def __init__(
        self,
        topk_ratio=0.05,
        active_quantile=0.80,
        active_abs_floor=0.40,
        smooth_kernel=3,
        strength_tau=1.20,
        strength_temp=0.60,
        compactness_tau=0.18,
        score_weights=(0.35, 0.30, 0.35),
        eps=1e-6,
    ):
        super().__init__()
        if not 0 < topk_ratio <= 1:
            raise ValueError("topk_ratio must be in (0, 1]")
        if not 0 <= active_quantile <= 1:
            raise ValueError("active_quantile must be in [0, 1]")
        if smooth_kernel < 1 or smooth_kernel % 2 == 0:
            raise ValueError("smooth_kernel must be a positive odd integer")
        if strength_temp <= 0 or compactness_tau <= 0:
            raise ValueError("SRQS temperatures must be positive")
        if len(score_weights) != 3 or sum(score_weights) <= 0:
            raise ValueError("score_weights must contain three positive-sum values")

        self.topk_ratio = float(topk_ratio)
        self.active_quantile = float(active_quantile)
        self.active_abs_floor = float(active_abs_floor)
        self.smooth_kernel = int(smooth_kernel)
        self.strength_tau = float(strength_tau)
        self.strength_temp = float(strength_temp)
        self.compactness_tau = float(compactness_tau)
        weight_sum = float(sum(score_weights))
        self.score_weights = tuple(float(w) / weight_sum for w in score_weights)
        self.eps = float(eps)

    @staticmethod
    def _largest_component_energy(mask, weight_map):
        """Return the largest 8-connected component's response energy."""
        mask_cpu = mask.detach().to(device="cpu", dtype=torch.bool)
        weights_cpu = weight_map.detach().to(device="cpu", dtype=torch.float32)
        height, width = mask_cpu.shape
        visited = torch.zeros_like(mask_cpu)
        best_energy = 0.0
        neighbors = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        )
        for y in range(height):
            for x in range(width):
                if not bool(mask_cpu[y, x]) or bool(visited[y, x]):
                    continue
                queue = [(y, x)]
                visited[y, x] = True
                component_energy = 0.0
                while queue:
                    cy, cx = queue.pop()
                    component_energy += float(weights_cpu[cy, cx])
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and bool(mask_cpu[ny, nx])
                            and not bool(visited[ny, nx])
                        ):
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                best_energy = max(best_energy, component_energy)
        return best_energy

    def forward(self, response_map):
        if not torch.is_tensor(response_map):
            response_map = torch.as_tensor(response_map)
        response_map = response_map.squeeze().float()
        if response_map.ndim != 2:
            raise ValueError(
                f"response_map must resolve to [H,W], got {tuple(response_map.shape)}"
            )
        if not bool(torch.isfinite(response_map).all()):
            raise ValueError("response_map contains non-finite values")

        flat = response_map.flatten()
        q50 = torch.quantile(flat, 0.50)
        q95 = torch.quantile(flat, 0.95)
        topk = max(1, int(flat.numel() * self.topk_ratio))
        topk_mean = torch.topk(flat, k=topk).values.mean()
        strength_raw = torch.clamp(
            (topk_mean - q50) / (q95 - q50 + self.eps), min=0.0
        )
        strength = torch.sigmoid(
            (strength_raw - self.strength_tau) / self.strength_temp
        )

        score_map = torch.relu(
            (response_map - q50) / (q95 - q50 + self.eps)
        )
        if self.smooth_kernel > 1:
            score_map = F.avg_pool2d(
                score_map[None, None],
                kernel_size=self.smooth_kernel,
                stride=1,
                padding=self.smooth_kernel // 2,
            )[0, 0]

        compactness = score_map.new_tensor(0.0)
        purity = score_map.new_tensor(0.0)
        positive = score_map[score_map > 0]
        if positive.numel() > 0:
            threshold = max(
                float(torch.quantile(positive, self.active_quantile)),
                self.active_abs_floor,
            )
            active = score_map >= threshold
            if bool(active.any()):
                ys, xs = torch.nonzero(active, as_tuple=True)
                active_weights = score_map[ys, xs]
                coords = torch.stack((ys.float(), xs.float()), dim=1)
                center = (active_weights[:, None] * coords).sum(dim=0) / (
                    active_weights.sum() + self.eps
                )
                distances = torch.norm(coords - center[None], dim=1)
                height, width = score_map.shape
                diagonal = math.sqrt((height - 1) ** 2 + (width - 1) ** 2)
                spread = (active_weights * distances).sum() / (
                    active_weights.sum() * diagonal + self.eps
                )
                compactness = torch.exp(
                    -spread / self.compactness_tau
                ).clamp(0.0, 1.0)
                largest_energy = self._largest_component_energy(active, score_map)
                purity = score_map.new_tensor(
                    largest_energy / (float(active_weights.sum()) + self.eps)
                ).clamp(0.0, 1.0)

        w_strength, w_compactness, w_purity = self.score_weights
        score = (
            w_strength * strength
            + w_compactness * compactness
            + w_purity * purity
        )
        return {
            "strength": float(strength.detach().cpu()),
            "compactness": float(compactness.detach().cpu()),
            "purity": float(purity.detach().cpu()),
            "score": float(score.detach().cpu()),
        }


class CandidatePatchCoreScorer(nn.Module):
    """PatchCore-inspired 1-NN verification restricted to coarse candidates.

    PatchCore's exact full-image maximum anomaly score is inappropriate here:
    query background is expected to differ from the support foreground memory.
    We therefore preserve its local patch aggregation, foreground memory bank,
    and nearest-neighbour distance, but score each coarse connected component
    as a possible target and retain the best component.
    """

    def __init__(
        self,
        patch_size=3,
        candidate_threshold=0.5,
        min_candidate_tokens=2,
        support_coverage_fraction=0.5,
        eps=1e-6,
    ):
        super().__init__()
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        if not 0.0 < support_coverage_fraction <= 1.0:
            raise ValueError("support_coverage_fraction must be in (0, 1]")
        self.patch_size = int(patch_size)
        self.candidate_threshold = float(candidate_threshold)
        self.min_candidate_tokens = int(min_candidate_tokens)
        self.support_coverage_fraction = float(support_coverage_fraction)
        self.eps = float(eps)

    def _aggregate(self, features):
        if self.patch_size > 1:
            features = F.avg_pool2d(
                features[None],
                kernel_size=self.patch_size,
                stride=1,
                padding=self.patch_size // 2,
            )[0]
        return F.normalize(features, dim=0, eps=self.eps)

    @staticmethod
    def _components(mask):
        mask = mask.detach().to(device="cpu", dtype=torch.bool)
        height, width = mask.shape
        visited = torch.zeros_like(mask)
        components = []
        neighbors = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        )
        for y in range(height):
            for x in range(width):
                if not bool(mask[y, x]) or bool(visited[y, x]):
                    continue
                queue = [(y, x)]
                visited[y, x] = True
                component = []
                while queue:
                    cy, cx = queue.pop()
                    component.append((cy, cx))
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and bool(mask[ny, nx])
                            and not bool(visited[ny, nx])
                        ):
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                components.append(component)
        return components

    def forward(
        self,
        support_features,
        query_features,
        foreground_core,
        foreground_probability,
    ):
        support_aggregated = self._aggregate(support_features)
        query_aggregated = self._aggregate(query_features)
        memory = support_aggregated[:, foreground_core].T
        if memory.shape[0] == 0:
            raise ValueError("PatchCore foreground memory is empty")

        candidate_mask = foreground_probability >= self.candidate_threshold
        components = self._components(candidate_mask)
        eligible = [
            component
            for component in components
            if len(component) >= self.min_candidate_tokens
        ]
        if not eligible and components:
            eligible = [max(components, key=len)]
        if not eligible:
            return {
                "mean_similarity": -1.0,
                "worst_similarity": -1.0,
                "support_coverage": -1.0,
                "bidirectional_score": -1.0,
                "candidate_count": 0,
                "largest_candidate_tokens": 0,
            }

        best = None
        for component in eligible:
            ys = torch.tensor(
                [item[0] for item in component],
                device=query_features.device,
                dtype=torch.long,
            )
            xs = torch.tensor(
                [item[1] for item in component],
                device=query_features.device,
                dtype=torch.long,
            )
            candidate_tokens = query_aggregated[:, ys, xs].T
            similarities = candidate_tokens @ memory.T
            query_to_support = similarities.max(dim=1).values
            support_to_query = similarities.max(dim=0).values
            coverage_count = max(
                1,
                int(math.ceil(
                    support_to_query.numel() * self.support_coverage_fraction
                )),
            )
            mean_similarity = query_to_support.mean()
            # Inverting PatchCore's maximum 1-NN distance is the minimum 1-NN
            # similarity over candidate patches.
            worst_similarity = query_to_support.min()
            support_coverage = support_to_query.topk(coverage_count).values.mean()
            bidirectional = 0.5 * (mean_similarity + support_coverage)
            candidate_result = {
                "mean_similarity": float(mean_similarity.detach().cpu()),
                "worst_similarity": float(worst_similarity.detach().cpu()),
                "support_coverage": float(support_coverage.detach().cpu()),
                "bidirectional_score": float(bidirectional.detach().cpu()),
                "tokens": len(component),
            }
            if best is None or (
                candidate_result["bidirectional_score"]
                > best["bidirectional_score"]
            ):
                best = candidate_result

        return {
            "mean_similarity": best["mean_similarity"],
            "worst_similarity": best["worst_similarity"],
            "support_coverage": best["support_coverage"],
            "bidirectional_score": best["bidirectional_score"],
            "candidate_count": len(eligible),
            "largest_candidate_tokens": max(len(item) for item in eligible),
        }


class SupportConditionedPresenceScorer(nn.Module):
    """Compute P0 baselines and a P1 support-conditioned likelihood ratio.

    Foreground tokens are taken from an eroded, high-occupancy support core;
    background evidence uses spatial grid prototypes outside a dilated support
    foreground.  Both classes additionally receive one global anchor, and the
    local/global branches are mixed with equal prior weight.

    Presence evidence is computed over the full query feature map.  It does
    not use the coarse segmentation to create components, deliberately
    separating "is the support target present?" from "where is it?".  The
    coarse probabilities are retained only as P0 comparison scores.
    """

    def __init__(
        self,
        temperature=0.05,
        probability_topk_fraction=0.01,
        foreground_occupancy=0.95,
        background_occupancy=0.05,
        morphology_kernel=3,
        background_grid=4,
        query_chunk_size=512,
        eps=1e-6,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < probability_topk_fraction <= 1:
            raise ValueError("probability_topk_fraction must be in (0, 1]")
        if morphology_kernel < 1 or morphology_kernel % 2 == 0:
            raise ValueError("morphology_kernel must be a positive odd integer")

        self.temperature = float(temperature)
        self.probability_topk_fraction = float(probability_topk_fraction)
        self.foreground_occupancy = float(foreground_occupancy)
        self.background_occupancy = float(background_occupancy)
        self.morphology_kernel = int(morphology_kernel)
        self.background_grid = int(background_grid)
        self.query_chunk_size = int(query_chunk_size)
        self.eps = float(eps)
        self.last_likelihood_ratio_map = None
        self.srqs = SimilarityResponseQualityScorer()
        self.patchcore = CandidatePatchCoreScorer()

    @staticmethod
    def _feature_map(features, name):
        if not torch.is_tensor(features):
            raise TypeError(f"{name} must be a tensor")
        # Standard ALPNet shapes are [way, shot, batch, C, H, W] for support
        # and [query, batch, C, H, W] for query.  P1 is intentionally 1-shot.
        while features.ndim > 3:
            features = features[0]
        if features.ndim != 3:
            raise ValueError(f"{name} must resolve to [C,H,W], got {tuple(features.shape)}")
        return features

    @staticmethod
    def _mask_2d(mask, device, dtype):
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
        mask = mask.to(device=device, dtype=dtype)
        while mask.ndim > 2:
            mask = mask[0]
        if mask.ndim != 2:
            raise ValueError(f"support_mask must resolve to [H,W], got {tuple(mask.shape)}")
        return mask

    def _support_regions(self, support_mask, feature_hw, device, dtype):
        mask = self._mask_2d(support_mask, device, dtype)[None, None]
        occupancy = F.interpolate(mask, size=feature_hw, mode="area").clamp(0.0, 1.0)
        pure_foreground = (occupancy >= self.foreground_occupancy).to(dtype)
        occupied = (occupancy > self.background_occupancy).to(dtype)
        pure_background = (occupancy <= self.background_occupancy).to(dtype)

        pad = self.morphology_kernel // 2
        foreground_core = 1.0 - F.max_pool2d(
            1.0 - pure_foreground,
            kernel_size=self.morphology_kernel,
            stride=1,
            padding=pad,
        )
        dilated_occupied = F.max_pool2d(
            occupied,
            kernel_size=self.morphology_kernel,
            stride=1,
            padding=pad,
        )
        background_core = (1.0 - dilated_occupied) * pure_background

        foreground_fallback = bool(foreground_core.sum() < 1)
        if foreground_fallback:
            foreground_core = pure_foreground
        if foreground_core.sum() < 1:
            foreground_core = (occupancy >= 0.5).to(dtype)

        background_fallback = bool(background_core.sum() < 1)
        if background_fallback:
            background_core = pure_background
        if background_core.sum() < 1:
            background_core = (occupancy < 0.5).to(dtype)

        if foreground_core.sum() < 1 or background_core.sum() < 1:
            raise ValueError("support mask does not contain usable foreground and background regions")
        return (
            foreground_core[0, 0].bool(),
            background_core[0, 0].bool(),
            occupancy[0, 0],
            (foreground_fallback, background_fallback),
        )

    def _background_prototypes(self, support_features, background_mask):
        _, height, width = support_features.shape
        prototypes = []
        y_edges = torch.linspace(0, height, self.background_grid + 1).round().long()
        x_edges = torch.linspace(0, width, self.background_grid + 1).round().long()
        for yi in range(self.background_grid):
            for xi in range(self.background_grid):
                y0, y1 = int(y_edges[yi]), int(y_edges[yi + 1])
                x0, x1 = int(x_edges[xi]), int(x_edges[xi + 1])
                region_mask = background_mask[y0:y1, x0:x1]
                if not bool(region_mask.any()):
                    continue
                region = support_features[:, y0:y1, x0:x1]
                prototypes.append(region[:, region_mask].mean(dim=1))

        if not prototypes:
            prototypes.append(support_features[:, background_mask].mean(dim=1))
        return F.normalize(torch.stack(prototypes), dim=1, eps=self.eps)

    def _global_prototype(self, support_features, weights):
        weights = weights.to(dtype=support_features.dtype)
        denominator = weights.sum().clamp_min(self.eps)
        prototype = (support_features * weights[None]).sum(dim=(1, 2)) / denominator
        return F.normalize(prototype, dim=0, eps=self.eps)

    def _logmeanexp_similarity(self, query_tokens, support_bank):
        tau = self.temperature
        normalizer = math.log(max(1, support_bank.shape[0]))
        chunks = []
        for start in range(0, query_tokens.shape[0], self.query_chunk_size):
            correlation = query_tokens[start:start + self.query_chunk_size] @ support_bank.T
            chunks.append(tau * (torch.logsumexp(correlation / tau, dim=1) - normalizer))
        return torch.cat(chunks)

    def _mix_equal_prior(self, local_evidence, global_evidence):
        branches = torch.stack((local_evidence, global_evidence), dim=1)
        return self.temperature * (
            torch.logsumexp(branches / self.temperature, dim=1) - math.log(2.0)
        )

    def forward(
        self,
        support_features,
        query_features,
        support_mask,
        foreground_probability,
    ):
        support_features = self._feature_map(support_features, "support_features")
        query_features = self._feature_map(query_features, "query_features")
        if foreground_probability.ndim == 3:
            foreground_probability = foreground_probability[0]
        if foreground_probability.ndim != 2:
            raise ValueError("foreground_probability must be [H,W] or [1,H,W]")
        if foreground_probability.shape != query_features.shape[-2:]:
            foreground_probability = F.interpolate(
                foreground_probability[None, None],
                size=query_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        foreground_core, background_core, occupancy, fallbacks = self._support_regions(
            support_mask,
            support_features.shape[-2:],
            support_features.device,
            support_features.dtype,
        )
        foreground_tokens = support_features[:, foreground_core].T
        foreground_tokens = F.normalize(foreground_tokens, dim=1, eps=self.eps)
        background_prototypes = self._background_prototypes(
            support_features, background_core
        )
        foreground_global = self._global_prototype(support_features, occupancy)
        background_global = self._global_prototype(
            support_features, background_core.to(support_features.dtype)
        )
        query_tokens = F.normalize(
            query_features.permute(1, 2, 0).reshape(-1, query_features.shape[0]),
            dim=1,
            eps=self.eps,
        )

        foreground_local = self._logmeanexp_similarity(query_tokens, foreground_tokens)
        background_local = self._logmeanexp_similarity(query_tokens, background_prototypes)
        foreground_evidence = self._mix_equal_prior(
            foreground_local, query_tokens @ foreground_global
        )
        background_evidence = self._mix_equal_prior(
            background_local, query_tokens @ background_global
        )
        likelihood_ratio = (foreground_evidence - background_evidence).reshape(
            query_features.shape[-2:]
        )

        flat_probability = foreground_probability.flatten()
        probability_count = max(
            1,
            int(math.ceil(flat_probability.numel() * self.probability_topk_fraction)),
        )
        flat_likelihood_ratio = likelihood_ratio.flatten()
        likelihood_count = max(
            1,
            int(math.ceil(flat_likelihood_ratio.numel() * self.probability_topk_fraction)),
        )
        self.last_likelihood_ratio_map = likelihood_ratio.detach()
        probability_srqs = self.srqs(foreground_probability)
        likelihood_srqs = self.srqs(likelihood_ratio)
        patchcore = self.patchcore(
            support_features,
            query_features,
            foreground_core,
            foreground_probability,
        )
        return PresenceScores(
            max_probability=float(flat_probability.max().detach().cpu()),
            topk_mean_probability=float(
                flat_probability.topk(probability_count).values.mean().detach().cpu()
            ),
            predicted_area_ratio=float(
                (foreground_probability >= 0.5).float().mean().detach().cpu()
            ),
            max_fg_bg_likelihood_ratio=float(
                flat_likelihood_ratio.max().detach().cpu()
            ),
            fg_bg_likelihood_ratio=float(
                flat_likelihood_ratio.topk(likelihood_count).values.mean().detach().cpu()
            ),
            probability_srqs_strength=probability_srqs["strength"],
            probability_srqs_compactness=probability_srqs["compactness"],
            probability_srqs_purity=probability_srqs["purity"],
            probability_srqs_score=probability_srqs["score"],
            likelihood_srqs_strength=likelihood_srqs["strength"],
            likelihood_srqs_compactness=likelihood_srqs["compactness"],
            likelihood_srqs_purity=likelihood_srqs["purity"],
            likelihood_srqs_score=likelihood_srqs["score"],
            patchcore_candidate_mean_similarity=patchcore["mean_similarity"],
            patchcore_candidate_worst_similarity=patchcore["worst_similarity"],
            patchcore_support_coverage=patchcore["support_coverage"],
            patchcore_bidirectional_score=patchcore["bidirectional_score"],
            patchcore_candidate_count=patchcore["candidate_count"],
            patchcore_largest_candidate_tokens=patchcore[
                "largest_candidate_tokens"
            ],
            foreground_token_count=int(foreground_tokens.shape[0]),
            background_prototype_count=int(background_prototypes.shape[0] + 1),
            foreground_core_fallback=fallbacks[0],
            background_core_fallback=fallbacks[1],
        )
