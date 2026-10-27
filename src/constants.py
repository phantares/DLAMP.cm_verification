import numpy as np

GRAVITY = 9.80665
R = 6371.0088  # km
STANDARD_LEVEL = [
    1000,
    925,
    850,
    700,
    600,
    500,
    400,
    300,
    250,
    200,
    150,
    100,
    70,
    50,
    30,
    20,
]

q_bins = [
    0,
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
]

wp_bins = np.concatenate([[0.0], np.logspace(-7, 4, 110 + 1)])
WPS = ["iwp", "lwp"]
