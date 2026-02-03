import logging
from typing import Final, Tuple

import numpy as np
import pandas as pd
from enum import Enum
from av2.evaluation.detection.utils import DetectionCfg
from av2.structures.cuboid import ORDERED_CUBOID_COL_NAMES
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt
from av2.utils.constants import EPS
from av2.evaluation.detection.constants import (
    InterpType,
)

DTS_COLUMN_NAMES: Final[Tuple[str, ...]] = tuple(ORDERED_CUBOID_COL_NAMES) + ("score",)
GTS_COLUMN_NAMES: Final[Tuple[str, ...]] = tuple(ORDERED_CUBOID_COL_NAMES) + ("num_interior_pts",)
UUID_COLUMN_NAMES: Final[Tuple[str, ...]] = (
    "log_id",
    "timestamp_ns",
    "category",
)

logger = logging.getLogger(__name__)

class TruePositiveErrorNames(str, Enum):
    """True positive error names."""

    ATE = "ATE"
    ASE = "ASE"
    AOE = "AOE"


class MetricNames(str, Enum):
    """Metric names."""

    AP = "AP"
    ATE = TruePositiveErrorNames.ATE.value
    ASE = TruePositiveErrorNames.ASE.value
    AOE = TruePositiveErrorNames.AOE.value
    CDS = "CDS"
    RECALL = "RECALL"


def summarize_metrics(
    dts: pd.DataFrame,
    gts: pd.DataFrame,
    cfg: DetectionCfg,
) -> pd.DataFrame:
    """Calculate and print the 3D object detection metrics.

    Args:
        dts: (N,14) Table of detections.
        gts: (M,15) Table of ground truth annotations.
        cfg: Detection configuration.

    Returns:
        The summary metrics.
    """
    # Sample recall values in the [0, 1] interval.
    recall_interpolated: NDArrayFloat = np.linspace(0, 1, cfg.num_recall_samples, endpoint=True)

    # Initialize the summary metrics.
    summary = pd.DataFrame(
        {s.value: cfg.metrics_defaults[i] for i, s in enumerate(tuple(MetricNames))}, index=cfg.categories
    )

    average_precisions = pd.DataFrame({t: 0.0 for t in cfg.affinity_thresholds_m}, index=cfg.categories)
    average_recall = pd.DataFrame({t: 0.0 for t in cfg.affinity_thresholds_m}, index=cfg.categories)
    for category in cfg.categories:
        # Find detections that have the current category.
        is_category_dts = dts["category"] == category

        # Only keep detections if they match the category and have NOT been filtered.
        is_valid_dts = np.logical_and(is_category_dts, dts["is_evaluated"])

        # Get valid detections and sort them in descending order.
        category_dts = dts.loc[is_valid_dts].sort_values(by="score", ascending=False).reset_index(drop=True)

        # Find annotations that have the current category.
        is_category_gts = gts["category"] == category

        # Compute number of ground truth annotations.
        num_gts = gts.loc[is_category_gts, "is_evaluated"].sum()

        # Cannot evaluate without ground truth information.
        if num_gts == 0:
            continue

        for affinity_threshold_m in cfg.affinity_thresholds_m:
            true_positives: NDArrayBool = category_dts[affinity_threshold_m].astype(bool).to_numpy()

            # Continue if there aren't any true positives.
            if len(true_positives) == 0:
                continue

            # Compute average precision for the current threshold.
            threshold_average_precision, _, recall = compute_average_precision(true_positives, recall_interpolated, num_gts)

            # Record the average precision.
            average_precisions.loc[category, affinity_threshold_m] = threshold_average_precision
            average_recall.loc[category, affinity_threshold_m] = recall

        mean_average_precisions: NDArrayFloat = average_precisions.loc[category].to_numpy().mean()
        mean_average_recall: NDArrayFloat = average_recall.loc[category].to_numpy().mean()

        # Select only the true positives for each instance.
        middle_idx = len(cfg.affinity_thresholds_m) // 2
        middle_threshold = cfg.affinity_thresholds_m[middle_idx]
        is_tp_t = category_dts[middle_threshold].to_numpy().astype(bool)

        # Initialize true positive metrics.
        tp_errors: NDArrayFloat = np.array(cfg.tp_normalization_terms)

        # Check whether any true positives exist under the current threshold.
        has_true_positives = np.any(is_tp_t)

        # If true positives exist, compute the metrics.
        if has_true_positives:
            tp_error_cols = [str(x.value) for x in TruePositiveErrorNames]
            tp_errors = category_dts.loc[is_tp_t, tp_error_cols].to_numpy().mean(axis=0)

        # Convert errors to scores.
        tp_scores = 1 - np.divide(tp_errors, cfg.tp_normalization_terms)

        # Compute Composite Detection Score (CDS).
        cds = mean_average_precisions * np.mean(tp_scores)
        summary.loc[category] = np.array([mean_average_precisions, *tp_errors, cds, mean_average_recall])

    # Return the summary.
    return summary, average_recall

def compute_average_precision(
    tps: NDArrayBool, recall_interpolated: NDArrayFloat, num_gts: int
) -> Tuple[float, NDArrayFloat]:
    """Compute precision and recall, interpolated over N fixed recall points.

    Args:
        tps: True positive detections (ranked by confidence).
        recall_interpolated: Interpolated recall values.
        num_gts: Number of annotations of this class.

    Returns:
        The average precision and interpolated precision values.
    """
    cum_tps: NDArrayInt = np.cumsum(tps)
    cum_fps: NDArrayInt = np.cumsum(~tps)
    cum_fns: NDArrayInt = num_gts - cum_tps

    # Compute precision.
    precision: NDArrayFloat = cum_tps / (cum_tps + cum_fps + EPS)

    # Compute recall.
    recall: NDArrayFloat = cum_tps / (cum_tps + cum_fns)

    # Interpolate precision -- VOC-style.
    precision = interpolate_precision(precision)

    # Evaluate precision at different recalls.
    precision_interpolated: NDArrayFloat = np.interp(recall_interpolated, recall, precision, right=0)  # type: ignore

    average_precision: float = np.mean(precision_interpolated)
    recall3d: float = cum_tps[-1] / num_gts
    return average_precision, precision_interpolated, recall3d

def interpolate_precision(precision: NDArrayFloat, interpolation_method: InterpType = InterpType.ALL) -> NDArrayFloat:
    r"""Interpolate the precision at each sampled recall.

    This function smooths the precision-recall curve according to the method introduced in Pascal
    VOC:

    Mathematically written as:
        $$p_{\text{interp}}(r) = \max_{\tilde{r}: \tilde{r} \geq r} p(\tilde{r})$$

    See equation 2 in http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.6629&rep=rep1&type=pdf
        for more information.

    Args:
        precision: Precision at all recall levels (N,).
        interpolation_method: Accumulation method.

    Returns:
        (N,) The interpolated precision at all sampled recall levels.

    Raises:
        NotImplementedError: If the interpolation method is not implemented.
    """
    precision_interpolated: NDArrayFloat
    if interpolation_method == InterpType.ALL:
        precision_interpolated = np.maximum.accumulate(precision[::-1])[::-1]
    else:
        raise NotImplementedError("This interpolation method is not implemented!")
    return precision_interpolated