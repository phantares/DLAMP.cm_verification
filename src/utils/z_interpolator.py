import numpy as np


def interpolate_z(data, z_source, z_target):
    if np.array_equal(z_source, z_target):
        return data

    zs = np.log(z_source)
    zt = np.log(z_target)

    if zs[-1] < zs[0]:
        zs = zs[::-1]
        data = np.flip(data, axis=-3)

    layer_below = np.searchsorted(zs, zt)
    layer_below = np.clip(layer_below, 1, len(z_source) - 1)
    layer_above = layer_below - 1

    z0 = zs[layer_above]
    z1 = zs[layer_below]

    ratio = (zt - z0) / (z1 - z0)

    data_below = data[..., layer_below, :, :]
    data_above = data[..., layer_above, :, :]

    ratio_shape = [1] * data.ndim
    ratio_shape[-3] = len(z_target)
    ratio = ratio.reshape(ratio_shape)

    output = ratio * data_below + (1.0 - ratio) * data_above

    return output
