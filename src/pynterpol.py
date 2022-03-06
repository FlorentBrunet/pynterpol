"""
Fast image interpolation.
"""
__author__ = 'Florent Brunet'
__email__ = 'florent.brunet@algostia.com'

from numba import jit, uint8, uint32, float64
import numpy as np
import math


def interp_bilinear_u8(image, x, y, default_value=0.):
    """Bilinear interpolation of an n-channels image encoded as np.uint8.

    :param image: Image to interpolate. Can be single channel (ndim==2) or n-channels (ndim==3).
    :param x: Horizontal coordinates where to interpolate.
        "Horizontal" means that it corresponds to the 2nd coordinates in the image array (image[y,x]).
        Must be the same size as y.
    :param y: Vertical coordinate where to interpolate.
        "Vertical" means that it corresponds to the 1st coordinates in the image array (image[y,x]).
        Must be the same size as x (can be of a different shape but this is discouraged).
    :param default_value: Default value if (x,y) is outside the image. Can either be a scalar or an array with as many
     elements as the number of channels. If it's a scalar and the image has n channels, the scalar value is repeated
     n times.
    :return: An array of the interpolated values of type np.float64. It has the same shape as x.
    """
    if image.dtype == np.uint8:
        if image.ndim == 2:
            default = np.array(default_value).reshape(-1).astype(np.float64)
            if default.size != 1:
                raise TypeError(f'The default value must have one value but is has {default.size}')

            xx = x if type(x) == np.ndarray else np.array(x)
            yy = y if type(y) == np.ndarray else np.array(y)

            if xx.size != yy.size:
                raise TypeError(f'x and y must have the same size but received size(x)={xx.size} and size(y)={yy.size}')

            xxx = xx if xx.ndim == 1 else xx.reshape(-1)
            yyy = yy if yy.ndim == 1 else yy.reshape(-1)

            xxxx = xxx if xxx.dtype == np.float64 else xxx.astype(np.float64)
            yyyy = yyy if yyy.dtype == np.float64 else yyy.astype(np.float64)

            val = interp_gray_u8_native(image, xxxx, yyyy, default[0])

            return val.reshape(xx.shape)
        elif image.ndim == 3:
            n_channels = image.shape[2]

            default = np.array(default_value).reshape(-1).astype(np.float64)
            if default.size == 1:
                default = default[0] * np.ones(n_channels, dtype=np.float64)
            elif default.size != n_channels:
                raise TypeError(f'The default value must have either 1 element or the same number of elements than'
                                f' the number of channels ({n_channels}) but it has {default.size}')

            vals = []
            for i in range(n_channels):
                val = interp_bilinear_u8(image[:, :, i], x, y, default[i])
                vals.append(val)
            return np.dstack(vals)
        else:
            raise TypeError(f'The input image must have 2 (single-channel) or 3 dimensions (n-channels)'
                            f' but it has {image.ndim}')
    else:
        raise TypeError(f'The input image must have uint8 elements but it has {image.dtype}')


# Explicitly gives the prototype to @jit() since it is required if we want AOT (Ahead-of-Time compilation).
# Besides, it seems that precisely controlling the prototype types drastically improves the performance.
@jit(float64(uint8[:, :], float64, uint32, uint32, float64, float64), nopython=True, cache=True)
def other_cases_gray_u8_native(img, default_value, rx, ry, cx, cy):
    height = img.shape[0]
    width = img.shape[1]

    if (rx == width - 1) and (ry < height - 1):
        # Right border
        if cx > 0:
            return default_value
        else:
            v00 = img[ry, rx]
            v01 = img[ry + 1, rx]
            return v00 * (1.0 - cy) + v01 * cy

    elif (rx < width - 1) and (ry == height - 1):
        # Bottom border
        if cy > 0:
            return default_value
        else:
            v00 = img[ry, rx]
            v10 = img[ry, rx + 1]
            return v00 * (1.0 - cx) + v10 * cx

    elif (rx == width - 1) and (ry == height - 1):
        # Bottom right pixel
        if (cx > 0) or (cy > 0):
            return default_value
        else:
            return img[ry, rx]

    else:
        # Default value out of the bounds
        return default_value


@jit(float64[:](uint8[:, :], float64[:], float64[:], float64), nopython=True, cache=True)
def interp_gray_u8_native(img, x, y, default_value):
    height = img.shape[0]
    width = img.shape[1]

    val = np.zeros(x.size)

    for i in range(x.size):
        cx = x[i]
        rx = int(math.floor(cx))
        cx -= rx
        omcx = 1.0 - cx

        cy = y[i]
        ry = int(math.floor(cy))
        cy -= ry

        if (rx >= 0) and (rx < width - 1) and (ry >= 0) and (ry < height - 1):
            # This is the general case (i.e. the 'middle' of the image)
            # The other cases are deported in the function 'other_cases'
            # (it makes the inner code of the loop smaller and consequently
            # reduces the number of 'instruction cache misses')
            v00 = img[ry, rx]
            v10 = img[ry, rx + 1]
            v01 = img[ry + 1, rx]
            v11 = img[ry + 1, rx + 1]

            val[i] = (v00 * omcx + v10 * cx) * (1.0 - cy) + (v01 * omcx + v11 * cx) * cy
        else:
            val[i] = other_cases_gray_u8_native(img, default_value, rx, ry, cx, cy)

    return val.reshape(x.shape)
