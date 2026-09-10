"""Functional helpers for Polyp support selection matched to query dataset."""

import os
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

POLYP_DATASET_NAMES = (
    "Kvasir",
    "CVC-ClinicDB",
    "CVC-ColonDB",
    "ETIS-LaribPolypDB",
)
VALID_UNMATCHED_POLICIES = ("skip", "fallback_all", "error")


def infer_polyp_dataset_from_filename(path: str) -> str:
    """Infer sub-dataset for flat TrainDataset paths (no folder tag).

    ProtoSAM TrainDataset mixes Kvasir + Clinic with distinct filename styles:
    - Kvasir train/test: hash-like stems (e.g. cju1h89h6xbnx08352k2790o9)
    - Clinic train/test: numeric stems (e.g. 101.png)
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.isdigit():
        return "CVC-ClinicDB"
    return "Kvasir"


def dataset_name_from_path(
    path: str,
    dataset_names: Sequence[str] = POLYP_DATASET_NAMES,
) -> str:
    """Infer Polyp sub-dataset name from an image/mask path."""
    for name in dataset_names:
        if name in path:
            return name
    return infer_polyp_dataset_from_filename(path)


def build_support_pool_by_dataset(
    pool_paths: Sequence[Tuple[str, str]],
    dataset_names: Sequence[str] = POLYP_DATASET_NAMES,
) -> Dict[str, List[int]]:
    """Group global support-pool indices by sub-dataset."""
    pools: Dict[str, List[int]] = {name: [] for name in dataset_names}
    pools[""] = []
    for index, (image_path, _mask_path) in enumerate(pool_paths):
        name = dataset_name_from_path(image_path, dataset_names)
        key = name if name in pools else ""
        pools[key].append(index)
    return pools


def resolve_query_dataset(sample_batched) -> str:
    """Read query sub-dataset from the Polyp dataloader batch."""
    case = sample_batched.get("case")
    if case is None:
        return ""
    if isinstance(case, (list, tuple)):
        return str(case[0])
    return str(case)


def resolve_pool_indices(
    pools: Dict[str, List[int]],
    query_dataset: str,
    unmatched_policy: str = "skip",
) -> Tuple[Optional[List[int]], str]:
    """Return support-pool indices for *query_dataset* or apply fallback policy."""
    if unmatched_policy not in VALID_UNMATCHED_POLICIES:
        raise ValueError(
            f"polyp_unmatched_support_policy must be one of {VALID_UNMATCHED_POLICIES}, "
            f"got {unmatched_policy!r}"
        )

    indices = list(pools.get(query_dataset, []))
    if indices:
        return indices, query_dataset

    if unmatched_policy == "fallback_all":
        merged = sorted({i for idxs in pools.values() for i in idxs})
        return merged, query_dataset
    if unmatched_policy == "error":
        raise ValueError(
            f"No train support pool for query dataset {query_dataset!r}. "
            f"Available pools: {[k for k, v in pools.items() if v]}"
        )
    return None, query_dataset


def random_support_indices(
    pool_indices: Sequence[int],
    n_support: int = 1,
    seed: int = 42,
    query_index: int = 0,
) -> List[int]:
    """Deterministic random support indices within a pool."""
    if not pool_indices:
        raise ValueError("random_support_indices received an empty pool")
    if n_support > len(pool_indices):
        raise ValueError(
            f"n_support={n_support} exceeds pool size {len(pool_indices)}"
        )
    rng = random.Random(int(seed) + int(query_index))
    return rng.sample(list(pool_indices), k=int(n_support))


def load_supports_from_indices(
    indices: Sequence[int],
    pool_paths: Sequence[Tuple[str, str]],
    load_item: Callable[[str, str], Tuple[torch.Tensor, torch.Tensor, str]],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """Load support images/masks for the given global pool indices."""
    images: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    cases: List[str] = []
    for index in indices:
        image_path, mask_path = pool_paths[index]
        image, mask, case = load_item(image_path, mask_path)
        images.append(image)
        masks.append(mask)
        cases.append(case)
    return images, masks, cases


def select_random_matched_support(
    pool_paths: Sequence[Tuple[str, str]],
    pools: Dict[str, List[int]],
    query_dataset: str,
    load_item: Callable[[str, str], Tuple[torch.Tensor, torch.Tensor, str]],
    n_support: int = 1,
    seed: int = 42,
    query_index: int = 0,
    unmatched_policy: str = "skip",
) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], Optional[dict]]:
    """Pick random support(s) from the same sub-dataset as the query."""
    pool_indices, _ = resolve_pool_indices(pools, query_dataset, unmatched_policy)
    if pool_indices is None:
        return None, None, {"skipped": True, "query_dataset": query_dataset}

    chosen = random_support_indices(
        pool_indices, n_support=n_support, seed=seed, query_index=query_index
    )
    images, masks, cases = load_supports_from_indices(chosen, pool_paths, load_item)
    info = {
        "skipped": False,
        "query_dataset": query_dataset,
        "selected_indices": chosen,
        "n_pool": len(pool_indices),
        "support_datasets": cases,
    }
    return images, masks, info


def slice_pool_embeddings(
    support_embs: torch.Tensor,
    pool_indices: Sequence[int],
) -> torch.Tensor:
    """Select rows from a precomputed embedding matrix."""
    if not pool_indices:
        raise ValueError("slice_pool_embeddings received an empty index list")
    idx = torch.as_tensor(pool_indices, dtype=torch.long)
    return support_embs.index_select(0, idx)


def pool_summary(pools: Dict[str, List[int]]) -> Dict[str, int]:
    """Return {dataset_name: pool_size} for non-empty pools."""
    return {name: len(indices) for name, indices in pools.items() if indices}
