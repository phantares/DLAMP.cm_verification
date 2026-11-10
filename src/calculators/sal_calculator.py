import numpy as np
from scipy import ndimage


def calculate_sal(prediction: np.ndarray, target: np.ndarray, threshold: float):
    """
    Calculate SAL for one pair of 2D mixing ratio fields.

    Parameters
    ----------
    prediction, target : ndarray, shape (H, W)
        Nonnegative hydrometeor mixing ratio [kg/kg].

    threshold : float
        Fixed cloud-object identification threshold [kg/kg].

    Returns
    -------
    S, A, L : float
        SAL components. NaN if either field has no objects.
    """
    if prediction.shape != target.shape:
        raise ValueError("Shapes must match.")

    N = np.size(prediction, 0)

    sal_s_sum = 0
    sal_a_sum = 0
    sal_l_sum = 0
    n_valid = 0

    for n in range(N):
        p = prediction[n,]
        t = target[n,]

        pred = _sal_field_properties(p, threshold)
        tar = _sal_field_properties(t, threshold)

        if pred is None or tar is None:
            continue

        # Structure
        S = 2.0 * (pred["volume"] - tar["volume"]) / (pred["volume"] + tar["volume"])
        sal_s_sum += S

        # Amplitude
        A = 2.0 * (pred["mass"] - tar["mass"]) / (pred["mass"] + tar["mass"])
        sal_a_sum += A

        # Location: L1 + L2
        H, W = prediction.shape[-2:]
        dmax = np.hypot(H - 1, W - 1)

        L1 = np.linalg.norm(pred["center"] - tar["center"]) / dmax
        L2 = 2.0 * abs(pred["spread"] - tar["spread"]) / dmax
        L = L1 + L2
        sal_l_sum += L

        n_valid += 1

    return sal_s_sum, sal_a_sum, sal_l_sum, n_valid


def _sal_field_properties(q: np.ndarray, threshold: float):
    """
    Calculate properties for one 2D hydrometeor field.

    Returns
    -------
    total_mass : float
        Sum of mixing ratio over the entire domain.
    center : ndarray, shape (2,)
        Mixing-ratio-weighted center (y, x).
    volume : float
        Mass-weighted normalized object volume.
    spread : float
        Mass-weighted object distance from the field center.
    n_objects : int
        Number of identified objects.
    """
    H, W = q.shape
    dtype = q.dtype

    total_mass = np.sum(q)

    if total_mass <= 0:
        return None

    y = np.arange(H, dtype=dtype)
    x = np.arange(W, dtype=dtype)

    cy = np.dot(y, np.sum(q, axis=1)) / total_mass
    cx = np.dot(x, np.sum(q, axis=0)) / total_mass

    center = np.array([cy, cx])

    mask = q > threshold

    structure = ndimage.generate_binary_structure(2, 2)
    labels, n_objects = ndimage.label(
        mask,
        structure=structure,
    )

    if n_objects == 0:
        return None

    yy, xx = np.nonzero(mask)
    ids = labels[yy, xx]
    values = q[yy, xx]

    mass = np.bincount(
        ids,
        weights=values,
        minlength=n_objects + 1,
    )[1:]

    maxima = ndimage.maximum(
        q,
        labels=labels,
        index=np.arange(1, n_objects + 1),
    )
    maxima = np.atleast_1d(maxima)

    volume_obj = mass / maxima

    volume = np.sum(
        mass * volume_obj,
    ) / np.sum(mass)

    cy_obj = (
        np.bincount(
            ids,
            weights=values * yy,
            minlength=n_objects + 1,
        )[1:]
        / mass
    )

    cx_obj = (
        np.bincount(
            ids,
            weights=values * xx,
            minlength=n_objects + 1,
        )[1:]
        / mass
    )

    distance = np.hypot(
        cy_obj - center[0],
        cx_obj - center[1],
    )

    spread = np.sum(
        mass * distance,
    ) / np.sum(mass)

    return {
        "mass": total_mass,
        "center": center,
        "volume": volume,
        "spread": spread,
        "n_objects": n_objects,
    }
