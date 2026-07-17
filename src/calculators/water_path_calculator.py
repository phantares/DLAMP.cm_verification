import numpy as np

from constants import GRAVITY


def calculate_water_path(
    q: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    """
    Compute column-integrated water path (precipitable water).

    Parameters
    ----------
    q : np.ndarray
        Specific humidity [kg/kg].
        Shape: (z, h, w)    — 3-D
            or (t, z, h, w) — 4-D
    p : np.ndarray
        Pressure levels [hPa].
        Shape: (z,)         — 1-D level vector (broadcast automatically).

    Returns
    -------
    wp : np.ndarray
        Column-integrated water path [kg/m²].
        Shape: (h, w)    when q is 3-D
            or (t, h, w) when q is 4-D

    Notes
    -----
    * Integration uses the trapezoidal rule over pressure (Pa after unit
      conversion) via the hydrostatic relation:
          WP = -1/g ∫ q dp   [kg/m²]
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)

    ndim = q.ndim
    if ndim not in (3, 4):
        raise ValueError(f"q must be 3-D or 4-D, got {ndim}-D.")

    p_top_to_sfc = p[-1] > p[0]

    if p.ndim == 1:
        p = p.reshape((-1,) + (1,) * 2)
        if ndim == 4:
            p = p[np.newaxis]
    elif p.shape != q.shape:
        raise ValueError(
            f"p shape {p.shape} is incompatible with q shape {q.shape}. "
            "Provide either a 1-D level vector or a full pressure field "
            "matching q."
        )

    p_pa = p * 100.0
    wp = np.trapezoid(q, x=p_pa, axis=-3)

    if not p_top_to_sfc:
        wp = -wp

    wp = wp / GRAVITY  # [Pa·(kg/kg)] / [m/s²]  =  kg/m²

    return wp
