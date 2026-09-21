import numpy as np
from scipy.interpolate import LinearNDInterpolator


def interpolate_xy(
    data,
    lon_source,
    lat_source,
    lon_target,
    lat_target,
):

    source_points = np.column_stack(
        [
            lon_source.ravel(),
            lat_source.ravel(),
        ]
    )

    target_points = np.column_stack(
        [
            lon_target.ravel(),
            lat_target.ravel(),
        ]
    )

    values = np.moveaxis(data, (-2, -1), (0, 1))
    values = values.reshape(
        -1,
        *data.shape[:-2],
    )

    interp = LinearNDInterpolator(
        source_points,
        values,
        fill_value=np.nan,
    )

    output = interp(target_points)

    output = output.reshape(
        *lon_target.shape,
        *data.shape[:-2],
    )

    output = np.moveaxis(
        output,
        (0, 1),
        (-2, -1),
    )

    return output
