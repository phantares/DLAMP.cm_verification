import numpy as np


def calculate_vertical_wasserstein(
    prediction: np.ndarray,
    target: np.ndarray,
    p: np.ndarray,
):
    """
    Column-wise vertical Wasserstein distance.

    Parameters
    ----------
    prediction, target: np.ndarray
        Shape (T, Z, H, W).
        Hydrometeor mixing ratio [kg/kg].
        Values <= 0 are treated as zero.

    p : np.ndarray
        Pressure levels [hPa], shape (Z,).

    Returns
    -------
    wd_sum : np.ndarray
        Sum of valid WD.

    wd_sample: np.ndarray
        Number of valid samples.
    """

    q_pred = np.asarray(prediction, dtype=float)
    q_tar = np.asarray(target, dtype=float)
    p = np.asarray(p, dtype=float)

    dp = np.empty_like(p)
    dp[0] = abs(p[1] - p[0]) / 2.0
    dp[-1] = abs(p[-1] - p[-2]) / 2.0
    dp[1:-1] = np.abs(p[:-2] - p[2:]) / 2.0
    dp = dp[None, :, None, None]

    mass_pred = q_pred * dp
    mass_tar = q_tar * dp

    total_pred = mass_pred.sum(axis=-3)
    total_tar = mass_tar.sum(axis=-3)

    valid = (total_pred > 0) & (total_tar > 0)

    prob_pred = np.divide(
        mass_pred,
        total_pred[:, None, :, :],
        out=np.zeros_like(mass_pred),
        where=valid[:, None, :, :],
    )
    prob_tar = np.divide(
        mass_tar,
        total_tar[:, None, :, :],
        out=np.zeros_like(mass_tar),
        where=valid[:, None, :, :],
    )

    cdf_pred = np.cumsum(
        prob_pred,
        axis=1,
    )
    cdf_tar = np.cumsum(
        prob_tar,
        axis=1,
    )

    dz = np.diff(-p)
    wd = np.sum(
        np.abs(cdf_pred[:, :-1] - cdf_tar[:, :-1]) * dz[None, :, None, None],
        axis=1,
    )

    return np.sum(wd, where=valid), np.sum(valid)
