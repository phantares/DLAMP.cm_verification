import numpy as np
from constants import R


def calculate_cross_section_points(start_point, end_point, res):
    lat1, lon1, lat2, lon2 = map(np.radians, start_point + end_point)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    total_dist = 2 * R * np.arcsin(np.sqrt(a))

    n = int(np.round(total_dist / res)) + 1

    lat_cs = np.linspace(start_point[0], end_point[0], n)
    lon_cs = np.linspace(start_point[1], end_point[1], n)

    return lat_cs, lon_cs
