"""Evaluation related utilities"""

from typing import Any, Mapping

from numpy import mean, ndarray, ones, quantile
from surface_distance import (  # type: ignore
    compute_robust_hausdorff,
    compute_surface_distances,
)
from torch import from_numpy

from algorithm.ndv import (
    calculate_jacobian_determinants,
    calculate_non_diffeomorphic_volume,
)


def compute_summary_statistics(values: ndarray, mask: ndarray | None = None) -> Mapping[str, Any]:
    """Compute summary statistics for a numpy array."""
    if mask is not None:
        mask = mask.astype("bool")
        if not mask.any():
            return {}
        values = values[mask]
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "q25": float(quantile(values, 0.25)),
        "q50": float(quantile(values, 0.50)),
        "q75": float(quantile(values, 0.75)),
        "q95": float(quantile(values, 0.95)),
        "q99": float(quantile(values, 0.99)),
    }


def compute_regularity_metrics(displacement_field: ndarray) -> Mapping[str, Any]:
    """Compute regularity metrics for a displacement field."""
    determinants = calculate_jacobian_determinants(
        from_numpy(displacement_field).movedim(-1, 0)[None]
    )
    return {
        "ndv": calculate_non_diffeomorphic_volume(determinants).item(),
        "det_std": determinants["000"].std().item(),
    }


def compute_similarity_metrics(
    reference: ndarray, registered: ndarray, mask: ndarray | None = None
) -> Mapping[str, Any]:
    """Compute similarity metrics between images."""
    similarity_absolute_diff_volume: ndarray = abs(reference - registered)
    similarity_absolute_metrics = compute_summary_statistics(similarity_absolute_diff_volume, mask)
    similarity_squared_diff_volume: ndarray = (reference - registered) ** 2
    similarity_squared_metrics = compute_summary_statistics(similarity_squared_diff_volume, mask)
    return {
        "absolute": similarity_absolute_metrics,
        "squared": similarity_squared_metrics,
    }


def compute_tissue_overlap_metrics(
    reference: ndarray,
    registered: ndarray,
    label_to_name: Mapping[int, str],
    mask: ndarray | None = None,
) -> Mapping[str, Any]:
    """Compute segmentation metrics between label images."""
    dice_scores: dict[str, float] = {}
    surface_distances: dict[str, Any] = {}
    for label, name in label_to_name.items():
        label_mask: ndarray = reference == label
        registered_mask: ndarray = registered == label
        if mask is not None:
            label_mask = label_mask & mask
            registered_mask = registered_mask & mask
        if label_mask.sum() == 0:
            continue
        dice_scores[name] = _dice(label_mask, registered_mask).item()
        if registered_mask.sum() == 0:
            continue
        label_surface_distances = compute_surface_distances(
            label_mask,
            registered_mask,
            ones(3),
        )
        surface_distances.setdefault("q95", {})[name] = compute_robust_hausdorff(
            label_surface_distances,
            95.0,
        )
        surface_distances.setdefault("q99", {})[name] = compute_robust_hausdorff(
            label_surface_distances,
            99.0,
        )
        surface_distances.setdefault("max", {})[name] = compute_robust_hausdorff(
            label_surface_distances,
            100.0,
        )
    if dice_scores:
        dice_scores["mean"] = float(mean(list(dice_scores.values())))
    if surface_distances:
        for surface_distance_quantiles in surface_distances.values():
            surface_distance_quantiles["mean"] = float(
                mean(list(surface_distance_quantiles.values()))
            )
    return {
        "dice": dice_scores,
        "surface_distance": surface_distances,
    }


def _dice(mask_1: ndarray, mask_2: ndarray) -> ndarray:
    intersection = (mask_1 & mask_2).sum()
    union = mask_1.sum() + mask_2.sum()
    return 2 * intersection / union
