from collections.abc import Sequence

import numpy as np


def calculate_bin(bins: Sequence[float], data) -> np.ndarray:
    counts, _ = np.histogram(data, bins=bins)
    return counts
