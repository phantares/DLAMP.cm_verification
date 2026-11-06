import numpy as np
from skimage.metrics import structural_similarity


def calculate_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    cloud_threshold: float,
    q0: float,
    q_max: float,
    window_size: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    pred, target : ndarray
        Shape (N, H, W).

    cloud_threshold : float
        Threshold for defining cloudy grid cells.

    q0 : float
        Fixed reference mixing ratio.

    q_max: float
        Fixed maximun mixing ratio.

    win_size : int
        Odd SSIM window size.

    Returns
    -------
    ssim_sum : ndarray, shape (Z,)
        Sum of SSIM scores for each pressure level.

    n_valid : ndarray, shape (Z,)
        Number of valid cases for each level.
    """
    if prediction.shape != target.shape:
        raise ValueError("Shapes must match.")

    N = np.size(prediction, 0)

    ssim_sum = 0
    n_valid = 0

    denominator = np.log10(1.0 + q_max / q0)

    for n in range(N):
        p = prediction[n,]
        t = target[n,]

        if not (np.any(p > cloud_threshold) or np.any(t > cloud_threshold)):
            continue

        p_norm = np.log10(1.0 + p / q0) / denominator
        t_norm = np.log10(1.0 + t / q0) / denominator

        p_norm = np.clip(p_norm, 0.0, 1.0)
        t_norm = np.clip(t_norm, 0.0, 1.0)

        score = structural_similarity(
            t_norm,
            p_norm,
            data_range=1.0,
            win_size=window_size,
            gaussian_weights=False,
        )

        if np.isfinite(score):
            ssim_sum += score
            n_valid += 1

    return ssim_sum, n_valid
