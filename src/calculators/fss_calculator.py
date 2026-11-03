from collections.abc import Sequence

import numpy as np
from scipy.ndimage import uniform_filter


def calculate_fss(
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.0,
    window_sizes: Sequence[float] = np.arange(1, 26, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate FSS components for 3D cloud occurrence.

    Parameters
    ----------
    pred, tar : np.ndarray
        Shape (T, Z, H, W).
        Hydrometeor mixing ratio for one species.

    threshold : float
        Threshold for defining cloud occurrence.

    window_sizes : list[int, ...]
        Odd-sized horizontal neighborhoods.

    Returns
    -------
    numerator : np.ndarray
        Shape (n_windows,).
        Sum of squared fraction differences.

    denominator : np.ndarray
        Shape (n_windows,).
        Sum of squared forecast and target fractions.
    """
    for n in window_sizes:
        if n < 1 or n % 2 == 0:
            raise ValueError("Window sizes must be positive odd integers")

    dtype = prediction.dtype
    cloud_pred = (prediction > threshold).astype(dtype)
    cloud_tar = (target > threshold).astype(dtype)

    numerator = np.zeros(len(window_sizes), dtype=dtype)
    denominator = np.zeros(len(window_sizes), dtype=dtype)

    H, W = prediction.shape[-2:]

    for i, n in enumerate(window_sizes):
        if n == 1:
            fraction_pred = cloud_pred
            fraction_tar = cloud_tar

        else:
            size = (1, 1, n, n)

            valid_count = uniform_filter(
                np.ones((H, W), dtype=dtype), size=n, mode="constant", cval=0.0
            )

            fraction_pred = (
                uniform_filter(cloud_pred, size=size, mode="constant", cval=0.0)
                / valid_count
            )

            fraction_tar = (
                uniform_filter(cloud_tar, size=size, mode="constant", cval=0.0)
                / valid_count
            )

        numerator[i] = np.sum((fraction_pred - fraction_tar) ** 2)
        denominator[i] = np.sum(fraction_pred**2) + np.sum(fraction_tar**2)

    return numerator, denominator
