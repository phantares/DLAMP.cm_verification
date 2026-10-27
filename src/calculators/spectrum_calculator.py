import numpy as np
from scipy.signal.windows import tukey


def calculate_isotropic_spectrum_sum(
    data: np.ndarray, grid_spacing_km: float, tukey_alpha: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Accumulate isotropic power spectra over multiple 2D fields.

    Parameters
    ----------
    data : np.ndarray
        Input array with shape [case, H, W].

    grid_spacing_km : float
        Horizontal grid spacing in kilometers.
        The same spacing is assumed for both x and y directions.

    tukey_alpha : float, optional
        Shape parameter of the Tukey window.
        0.0 means no tapering.
        1.0 is equivalent to a Hann window.
        A value around 0.1-0.2 provides mild edge tapering.

    Returns
    -------
    energy_sum : np.ndarray
        Sum of isotropic power spectra over all valid cases.
        Shape is [n_bins].

    count : np.ndarray
        Number of valid case spectra contributing to each radial bin.
        Shape is [n_bins].
    """
    data = np.asarray(data)

    if data.ndim != 3:
        raise ValueError("data shape must be [B, H, W]")

    if grid_spacing_km <= 0:
        raise ValueError("grid_spacing_km must be greater than 0")

    _, ny, nx = data.shape

    wx = tukey(nx, alpha=tukey_alpha)
    wy = tukey(ny, alpha=tukey_alpha)
    window = np.outer(wy, wx)
    window_power = np.mean(window**2)

    kx = np.fft.fftfreq(nx, d=grid_spacing_km)
    ky = np.fft.fftfreq(ny, d=grid_spacing_km)

    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")

    radial_k = np.hypot(kx_grid, ky_grid)

    dk = min(1.0 / (nx * grid_spacing_km), 1.0 / (ny * grid_spacing_km))
    k_max = 1.0 / (2.0 * grid_spacing_km)

    bin_edges = np.arange(0.0, k_max + dk, dk, dtype=np.float64)
    n_bins = len(bin_edges) - 1

    radial_k_flat = radial_k.ravel()
    bin_index = np.searchsorted(bin_edges, radial_k_flat, side="right") - 1

    valid_fourier = (bin_index >= 0) & (bin_index < n_bins) & (radial_k_flat > 0)
    valid_bin_index = bin_index[valid_fourier]
    radial_count = np.bincount(valid_bin_index, minlength=n_bins)

    spectrum_sum = np.zeros(n_bins)
    spectrum_count = np.zeros(n_bins)

    for field in data:
        finite = np.isfinite(field)

        if not finite.any():
            continue

        anomaly = np.where(finite, field - field[finite].mean(), 0.0)
        anomaly *= window

        fft = np.fft.fft2(anomaly)
        power = np.abs(fft) ** 2 / (ny * nx) / window_power

        power_flat = power.ravel()
        radial_energy_sum = np.bincount(
            valid_bin_index, weights=power_flat[valid_fourier], minlength=n_bins
        )

        case_energy = np.divide(
            radial_energy_sum,
            radial_count,
            out=np.full(n_bins, np.nan),
            where=radial_count > 0,
        )

        valid_case_bin = np.isfinite(case_energy)

        spectrum_sum[valid_case_bin] += case_energy[valid_case_bin]
        spectrum_count[valid_case_bin] += 1

    return spectrum_sum, spectrum_count


def get_spectrum_wavelength(
    shape: tuple[int, int], grid_spacing_km: float
) -> np.ndarray:

    ny, nx = shape

    dk = min(1.0 / (nx * grid_spacing_km), 1.0 / (ny * grid_spacing_km))
    k_max = 1.0 / (2.0 * grid_spacing_km)

    bin_edges = np.arange(0.0, k_max + dk, dk)
    k_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return 1.0 / k_centers


def smooth_spectrum(spectrum: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return spectrum

    if window % 2 == 0:
        raise ValueError("smooth window use odd")

    pad = window // 2
    padded = np.pad(spectrum, (pad, pad), mode="edge")
    kernel = np.ones(window) / window

    return np.convolve(padded, kernel, mode="valid")
