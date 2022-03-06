"""
Fast image interpolation.
"""
__author__ = 'Florent Brunet'
__email__ = 'florent.brunet@algostia.com'

from numba import jit, uint8, uint32, float64
from enum import Enum
import numpy as np
import math


class InterpolationMethod(Enum):
    BILINEAR = 1
    BICUBIC = 2


def interp_bilinear_u8(image, x, y, default_value=0.):
    """Bilinear interpolation of an image encoded as a NumPy ndarray (with dtype=uint8).

    :param image: Image to interpolate.
        Can be 2D single channel image (img.ndim=2 and img.shape=(height,width))
        or a 2D n-channels image (img.ndim=3 and shape=(height,width,n_channels)).
    :param x: Horizontal coordinates where to interpolate.
        "Horizontal" means that it corresponds to the 2nd coordinates in the image array (image[y,x]).
        Must be the same size as y.
    :param y: Vertical coordinate where to interpolate.
        "Vertical" means that it corresponds to the 1st coordinates in the image array (image[y,x]).
        Must be the same size as x (can be of a different shape but this is discouraged).
    :param default_value: Default value if (x,y) is outside the image.
        Can either be a scalar or an array with as many elements as the number of channels.
        If it's a scalar and the image has n channels, the scalar value is repeated n times.
    :return: An array of the interpolated values of type np.float64. It has the same shape as x.
    """
    return interp_u8(image, x, y, default_value, InterpolationMethod.BILINEAR)


def interp_bicubic_u8(image, x, y, default_value=0.):
    """Bicubic interpolation of an image encoded as a NumPy ndarray (with dtype=uint8).

    :param image: Image to interpolate.
        Can be 2D single channel image (img.ndim=2 and img.shape=(height,width))
        or a 2D n-channels image (img.ndim=3 and shape=(height,width,n_channels)).
    :param x: Horizontal coordinates where to interpolate.
        "Horizontal" means that it corresponds to the 2nd coordinates in the image array (image[y,x]).
        Must be the same size as y.
    :param y: Vertical coordinate where to interpolate.
        "Vertical" means that it corresponds to the 1st coordinates in the image array (image[y,x]).
        Must be the same size as x (can be of a different shape but this is discouraged).
    :param default_value: Default value if (x,y) is outside the image.
        Can either be a scalar or an array with as many elements as the number of channels.
        If it's a scalar and the image has n channels, the scalar value is repeated n times.
    :return: An array of the interpolated values of type np.float64. It has the same shape as x.
    """
    return interp_u8(image, x, y, default_value, InterpolationMethod.BICUBIC)


def interp_u8(image, x, y, default_value=0., method=InterpolationMethod.BILINEAR):
    """Interpolation of an image encoded as a NumPy ndarray (with dtype=uint8).

    :param image: Image to interpolate.
        Can be 2D single channel image (img.ndim=2 and img.shape=(height,width))
        or a 2D n-channels image (img.ndim=3 and shape=(height,width,n_channels)).
    :param x: Horizontal coordinates where to interpolate.
        "Horizontal" means that it corresponds to the 2nd coordinates in the image array (image[y,x]).
        Must be the same size as y.
    :param y: Vertical coordinate where to interpolate.
        "Vertical" means that it corresponds to the 1st coordinates in the image array (image[y,x]).
        Must be the same size as x (can be of a different shape but this is discouraged).
    :param default_value: Default value if (x,y) is outside the image.
        Can either be a scalar or an array with as many elements as the number of channels.
        If it's a scalar and the image has n channels, the scalar value is repeated n times.
    :param method: Interpolation method to be used.
    :return: An array of the interpolated values of type np.float64. It has the same shape as x.
    """
    if image.dtype == np.uint8:
        if image.ndim == 2:
            default = np.array(default_value).reshape(-1).astype(np.float64)
            if default.size != 1:
                raise ValueError(f'The default value must have one value but is has {default.size}')

            xx = x if type(x) == np.ndarray else np.array(x)
            yy = y if type(y) == np.ndarray else np.array(y)

            if xx.size != yy.size:
                raise ValueError(
                    f'x and y must have the same size but received size(x)={xx.size} and size(y)={yy.size}')

            xxx = xx if xx.ndim == 1 else xx.reshape(-1)
            yyy = yy if yy.ndim == 1 else yy.reshape(-1)

            xxxx = xxx if xxx.dtype == np.float64 else xxx.astype(np.float64)
            yyyy = yyy if yyy.dtype == np.float64 else yyy.astype(np.float64)

            if method == InterpolationMethod.BILINEAR:
                val = interp_bilinear_gray_u8_native(image, xxxx, yyyy, default[0])
            elif method == InterpolationMethod.BICUBIC:
                val = interp_bicubic_gray_u8_native(image, xxxx, yyyy, default[0])
            else:
                raise ValueError(f'Unsupported interpolation method {method}')

            return val.reshape(xx.shape)
        elif image.ndim == 3:
            n_channels = image.shape[2]

            default = np.array(default_value).reshape(-1).astype(np.float64)
            if default.size == 1:
                default = default[0] * np.ones(n_channels, dtype=np.float64)
            elif default.size != n_channels:
                raise ValueError(f'The default value must have either 1 element or the same number of elements as'
                                 f' the number of channels ({n_channels}) but it has {default.size}')

            vals = []
            for i in range(n_channels):
                val = interp_bilinear_u8(image[:, :, i], x, y, default[i])
                vals.append(val)
            return np.dstack(vals)
        else:
            raise ValueError(f'The input image must have 2 (single-channel) or 3 dimensions (n-channels)'
                             f' but it has {image.ndim}')
    else:
        raise TypeError(f'The input image must have uint8 elements but it has {image.dtype}')


# Explicitly gives the prototype to @jit() since it is required if we want AOT (Ahead-of-Time compilation).
# Besides, it seems that precisely controlling the prototype types drastically improves the performance.
@jit(float64(uint8[:, :], float64, uint32, uint32, float64, float64), nopython=True, cache=True)
def interp_bilinear_other_cases_gray_u8_native(img, default_value, rx, ry, cx, cy):
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
def interp_bilinear_gray_u8_native(img, x, y, default_value):
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
            val[i] = interp_bilinear_other_cases_gray_u8_native(img, default_value, rx, ry, cx, cy)

    return val.reshape(x.shape)


# noinspection DuplicatedCode
@jit(float64(uint8[:, :], float64, uint32, uint32, float64, float64, float64, float64, float64, float64),
     nopython=True, cache=True)
def interp_bicubic_other_cases_gray_u8_native(img, default_value, rx, ry, cx, cx2, cx3, cy, cy2, cy3):
    height = img.shape[0]
    width = img.shape[1]

    p0 = 2. * cy2 - cy3 - cy
    p1 = 3. * cy3 - 5. * cy2 + 2.
    p2 = 4. * cy2 - 3. * cy3 + cy
    p3 = cy3 - cy2

    # CASE 1 -----------------------------------------------------------------------------------------------------------
    if (rx == 0) and (ry == 0):

        v10 = 3. * img[ry, rx] - 3. * img[ry + 1, rx] + img[ry + 2, rx]
        v11 = img[ry, rx]
        v21 = img[ry, rx + 1]
        v31 = img[ry, rx + 2]

        v20 = 3. * img[ry, rx + 1] - 3 * img[ry + 1, rx + 1] + img[ry + 2, rx + 1]
        v12 = img[ry + 1, rx]
        v22 = img[ry + 1, rx + 1]
        v32 = img[ry + 1, rx + 2]

        v30 = 3. * img[ry, rx + 2] - 3. * img[ry + 1, rx + 2] + img[ry + 2, rx + 2]
        v13 = img[ry + 2, rx]
        v23 = img[ry + 2, rx + 1]
        v33 = img[ry + 2, rx + 2]

        v1 = v10 * p0 + v11 * p1 + v12 * p2 + v13 * p3
        v2 = v20 * p0 + v21 * p1 + v22 * p2 + v23 * p3
        v3 = v30 * p0 + v31 * p1 + v32 * p2 + v33 * p3
        v0 = 3. * v1 - 3. * v2 + v3

        return (v0 * (2. * cx2 - cx3 - cx)
                + v1 * (3. * cx3 - 5. * cx2 + 2.)
                + v2 * (4. * cx2 - 3. * cx3 + cx)
                + v3 * (cx3 - cx2)) / 4.

    # CASE 5 -----------------------------------------------------------------------------------------------------------
    elif (rx == 0) and (ry >= 1) and (ry < height - 2):

        v10 = img[ry - 1, rx]
        v20 = img[ry - 1, rx + 1]
        v30 = img[ry - 1, rx + 2]

        v11 = img[ry, rx]
        v21 = img[ry, rx + 1]
        v31 = img[ry, rx + 2]

        v12 = img[ry + 1, rx]
        v22 = img[ry + 1, rx + 1]
        v32 = img[ry + 1, rx + 2]

        v13 = img[ry + 2, rx]
        v23 = img[ry + 2, rx + 1]
        v33 = img[ry + 2, rx + 2]

        v1 = v10 * p0 + v11 * p1 + v12 * p2 + v13 * p3
        v2 = v20 * p0 + v21 * p1 + v22 * p2 + v23 * p3
        v3 = v30 * p0 + v31 * p1 + v32 * p2 + v33 * p3
        v0 = 3 * v1 - 3 * v2 + v3

        return (v0 * (2. * cx2 - cx3 - cx)
                + v1 * (3. * cx3 - 5. * cx2 + 2.)
                + v2 * (4. * cx2 - 3. * cx3 + cx)
                + v3 * (cx3 - cx2)) / 4.

    # CASE 9 -----------------------------------------------------------------------------------------------------------
    elif (rx == 0) and (ry == height - 2):

        v10 = img[ry - 1, rx]
        v20 = img[ry - 1, rx + 1]
        v30 = img[ry - 1, rx + 2]

        v11 = img[ry, rx]
        v21 = img[ry, rx + 1]
        v31 = img[ry, rx + 2]

        v12 = img[ry + 1, rx]
        v22 = img[ry + 1, rx + 1]
        v32 = img[ry + 1, rx + 2]

        v13 = img[ry - 1, rx] - 3. * img[ry, rx] + 3. * img[ry + 1, rx]
        v23 = img[ry - 1, rx + 1] - 3. * img[ry, rx + 1] + 3. * img[ry + 1, rx + 1]
        v33 = img[ry - 1, rx + 2] - 3. * img[ry, rx + 2] + 3. * img[ry + 1, rx + 2]

        v1 = v10 * p0 + v11 * p1 + v12 * p2 + v13 * p3
        v2 = v20 * p0 + v21 * p1 + v22 * p2 + v23 * p3
        v3 = v30 * p0 + v31 * p1 + v32 * p2 + v33 * p3
        v0 = 3. * v1 - 3. * v2 + v3

        return (v0 * (2. * cx2 - cx3 - cx)
                + v1 * (3. * cx3 - 5. * cx2 + 2.)
                + v2 * (4. * cx2 - 3. * cx3 + cx)
                + v3 * (cx3 - cx2)) / 4.

    # CASE 13 ----------------------------------------------------------------------------------------------------------
    elif (rx == 0) and (ry == height - 1) and (cy == 0):

        v1 = img[ry, rx]
        v2 = img[ry, rx + 1]
        v3 = img[ry, rx + 2]
        v0 = 3. * v1 - 3. * v2 + v3

        return (v0 * (2. * cx2 - cx3 - cx)
                + v1 * (3. * cx3 - 5. * cx2 + 2.)
                + v2 * (4. * cx2 - 3. * cx3 + cx)
                + v3 * (cx3 - cx2)) / 2.

    # CASE 2 -----------------------------------------------------------------------------------------------------------
    elif (rx >= 1) and (rx < width - 2) and (ry == 0):

        v01 = img[ry, rx - 1]
        v11 = img[ry, rx]
        v21 = img[ry, rx + 1]
        v31 = img[ry, rx + 2]

        v02 = img[ry + 1, rx - 1]
        v12 = img[ry + 1, rx]
        v22 = img[ry + 1, rx + 1]
        v32 = img[ry + 1, rx + 2]

        v03 = img[ry + 2, rx - 1]
        v13 = img[ry + 2, rx]
        v23 = img[ry + 2, rx + 1]
        v33 = img[ry + 2, rx + 2]

        v00 = 3. * img[ry, rx - 1] - 3. * img[ry + 1, rx - 1] + img[ry + 2, rx - 1]
        v10 = 3. * img[ry, rx] - 3. * img[ry + 1, rx] + img[ry + 2, rx]
        v20 = 3. * img[ry, rx + 1] - 3. * img[ry + 1, rx + 1] + img[ry + 2, rx + 1]
        v30 = 3. * img[ry, rx + 2] - 3. * img[ry + 1, rx + 2] + img[ry + 2, rx + 2]

        v0 = v00 * p0 + v01 * p1 + v02 * p2 + v03 * p3
        v1 = v10 * p0 + v11 * p1 + v12 * p2 + v13 * p3
        v2 = v20 * p0 + v21 * p1 + v22 * p2 + v23 * p3
        v3 = v30 * p0 + v31 * p1 + v32 * p2 + v33 * p3

        return (v0 * (2. * cx2 - cx3 - cx)
                + v1 * (3. * cx3 - 5. * cx2 + 2.)
                + v2 * (4. * cx2 - 3. * cx3 + cx)
                + v3 * (cx3 - cx2)) / 4.

    # CASE 10 ----------------------------------------------------------------------------------------------------------
    elif (rx >= 1) and (rx < width - 2) and (ry == height - 2):

        v00 = img[ry - 1, rx - 1]
        v10 = img[ry - 1, rx]
        v20 = img[ry - 1, rx + 1]
        v30 = img[ry - 1, rx + 2]

        v01 = img[ry, rx - 1]
        v11 = img[ry, rx]
        v21 = img[ry, rx + 1]
        v31 = img[ry, rx + 2]

        v02 = img[ry + 1, rx - 1]
        v12 = img[ry + 1, rx]
        v22 = img[ry + 1, rx + 1]
        v32 = img[ry + 1, rx + 2]

        v03 = img[ry - 1, rx - 1] - 3. * img[ry, rx - 1] + 3. * img[ry + 1, rx - 1]
        v13 = img[ry - 1, rx] - 3. * img[ry, rx] + 3. * img[ry + 1, rx]
        v23 = img[ry - 1, rx + 1] - 3. * img[ry, rx + 1] + 3. * img[ry + 1, rx + 1]
        v33 = img[ry - 1, rx + 2] - 3. * img[ry, rx + 2] + 3. * img[ry + 1, rx + 2]

        v0 = v00 * p0 + v01 * p1 + v02 * p2 + v03 * p3
        v1 = v10 * p0 + v11 * p1 + v12 * p2 + v13 * p3
        v2 = v20 * p0 + v21 * p1 + v22 * p2 + v23 * p3
        v3 = v30 * p0 + v31 * p1 + v32 * p2 + v33 * p3

        return (v0 * (2. * cx2 - cx3 - cx)
                + v1 * (3. * cx3 - 5. * cx2 + 2.)
                + v2 * (4. * cx2 - 3. * cx3 + cx)
                + v3 * (cx3 - cx2)) / 4.

    # CASE 14 ----------------------------------------------------------------------------------------------------------
    elif (rx >= 1) and (rx < width - 2) and (ry == height - 1) and (cy == 0):

        v0 = img[ry, rx - 1]
        v1 = img[ry, rx]
        v2 = img[ry, rx + 1]
        v3 = img[ry, rx + 2]

        return (v0 * (2. * cx2 - cx3 - cx)
                + v1 * (3. * cx3 - 5. * cx2 + 2.)
                + v2 * (4. * cx2 - 3. * cx3 + cx)
                + v3 * (cx3 - cx2)) / 2.

    # CASE 3 -----------------------------------------------------------------------------------------------------------
    elif (rx == width - 2) and (ry == 0):

        v01 = img[ry, rx - 1]
        v11 = img[ry, rx]
        v21 = img[ry, rx + 1]

        v02 = img[ry + 1, rx - 1]
        v12 = img[ry + 1, rx]
        v22 = img[ry + 1, rx + 1]

        v03 = img[ry + 2, rx - 1]
        v13 = img[ry + 2, rx]
        v23 = img[ry + 2, rx + 1]

        v00 = 3. * img[ry, rx - 1] - 3. * img[ry + 1, rx - 1] + img[ry + 2, rx - 1]
        v10 = 3. * img[ry, rx] - 3. * img[ry + 1, rx] + img[ry + 2, rx]
        v20 = 3. * img[ry, rx + 1] - 3. * img[ry + 1, rx + 1] + img[ry + 2, rx + 1]

        v0 = v00 * p0 + v01 * p1 + v02 * p2 + v03 * p3
        v1 = v10 * p0 + v11 * p1 + v12 * p2 + v13 * p3
        v2 = v20 * p0 + v21 * p1 + v22 * p2 + v23 * p3
        v3 = 3. * v2 - 3. * v1 + v0

        return (v0 * (2. * cx2 - cx3 - cx)
                + v1 * (3. * cx3 - 5. * cx2 + 2.)
                + v2 * (4. * cx2 - 3. * cx3 + cx)
                + v3 * (cx3 - cx2)) / 4.

    # CASE 7 -----------------------------------------------------------------------------------------------------------
    elif (rx == width - 2) and (ry >= 1) and (ry < height - 2):

        v00 = img[ry - 1, rx - 1]
        v10 = img[ry - 1, rx]
        v20 = img[ry - 1, rx + 1]

        v01 = img[ry, rx - 1]
        v11 = img[ry, rx]
        v21 = img[ry, rx + 1]

        v02 = img[ry + 1, rx - 1]
        v12 = img[ry + 1, rx]
        v22 = img[ry + 1, rx + 1]

        v03 = img[ry + 2, rx - 1]
        v13 = img[ry + 2, rx]
        v23 = img[ry + 2, rx + 1]

        v0 = v00 * p0 + v01 * p1 + v02 * p2 + v03 * p3
        v1 = v10 * p0 + v11 * p1 + v12 * p2 + v13 * p3
        v2 = v20 * p0 + v21 * p1 + v22 * p2 + v23 * p3
        v3 = 3. * v2 - 3. * v1 + v0

        return (v0 * (2. * cx2 - cx3 - cx)
                + v1 * (3. * cx3 - 5. * cx2 + 2.)
                + v2 * (4. * cx2 - 3. * cx3 + cx)
                + v3 * (cx3 - cx2)) / 4.

    # CASE 11 ----------------------------------------------------------------------------------------------------------
    elif (rx == width - 2) and (ry == height - 2):

        v00 = img[ry - 1, rx - 1]
        v10 = img[ry - 1, rx]
        v20 = img[ry - 1, rx + 1]

        v01 = img[ry, rx - 1]
        v11 = img[ry, rx]
        v21 = img[ry, rx + 1]

        v02 = img[ry + 1, rx - 1]
        v12 = img[ry + 1, rx]
        v22 = img[ry + 1, rx + 1]

        v03 = img[ry - 1, rx - 1] - 3. * img[ry, rx - 1] + 3. * img[ry + 1, rx - 1]
        v13 = img[ry - 1, rx] - 3. * img[ry, rx] + 3. * img[ry + 1, rx]
        v23 = img[ry - 1, rx + 1] - 3. * img[ry, rx + 1] + 3. * img[ry + 1, rx + 1]

        v0 = v00 * p0 + v01 * p1 + v02 * p2 + v03 * p3
        v1 = v10 * p0 + v11 * p1 + v12 * p2 + v13 * p3
        v2 = v20 * p0 + v21 * p1 + v22 * p2 + v23 * p3
        v3 = 3. * v2 - 3. * v1 + v0

        return (v0 * (2. * cx2 - cx3 - cx)
                + v1 * (3. * cx3 - 5. * cx2 + 2.)
                + v2 * (4. * cx2 - 3. * cx3 + cx)
                + v3 * (cx3 - cx2)) / 4.

    # CASE 15 ----------------------------------------------------------------------------------------------------------
    elif (rx == width - 2) and (ry == height - 1) and (cy == 0):

        v0 = img[ry, rx - 1]
        v1 = img[ry, rx]
        v2 = img[ry, rx + 1]
        v3 = 3. * v2 - 3. * v1 + v0

        return (v0 * (2. * cx2 - cx3 - cx)
                + v1 * (3. * cx3 - 5. * cx2 + 2.)
                + v2 * (4. * cx2 - 3. * cx3 + cx)
                + v3 * (cx3 - cx2)) / 2.

    # CASE 4 -----------------------------------------------------------------------------------------------------------
    elif (rx == width - 1) and (cx == 0) and (ry == 0):

        v10 = 3. * img[ry, rx] - 3. * img[ry + 1, rx] + img[ry + 2, rx]
        v11 = img[ry, rx]
        v12 = img[ry + 1, rx]
        v13 = img[ry + 2, rx]

        return (v10 * p0 + v11 * p1 + v12 * p2 + v13 * p3) / 2.

    # CASE 8 -----------------------------------------------------------------------------------------------------------
    elif (rx == width - 1) and (cx == 0) and (ry >= 1) and (ry < height - 2):

        v10 = img[ry - 1, rx]
        v11 = img[ry, rx]
        v12 = img[ry + 1, rx]
        v13 = img[ry + 2, rx]

        return (v10 * p0 + v11 * p1 + v12 * p2 + v13 * p3) / 2.

    # CASE 12 ----------------------------------------------------------------------------------------------------------
    elif (rx == width - 1) and (cx == 0) and (ry == height - 2):

        v10 = img[ry - 1, rx]
        v11 = img[ry, rx]
        v12 = img[ry + 1, rx]
        v13 = 3. * v12 - 3. * v11 + v10

        return (v10 * p0 + v11 * p1 + v12 * p2 + v13 * p3) / 2.

    # CASE 16 ----------------------------------------------------------------------------------------------------------
    elif (rx == width - 1) and (cx == 0) and (ry == height - 1) and (cy == 0):

        return img[ry, rx]

    else:
        return default_value


# noinspection DuplicatedCode
@jit(float64[:](uint8[:, :], float64[:], float64[:], float64), nopython=True, cache=True)
def interp_bicubic_gray_u8_native(img, x, y, default_value):
    height = img.shape[0]
    width = img.shape[1]

    val = np.zeros(x.size)

    for i in range(x.size):
        cx = x[i]
        rx = int(math.floor(cx))
        cx -= rx
        cx2 = cx * cx
        cx3 = cx * cx2
        rx -= 1

        cy = y[i]
        ry = int(math.floor(cy))
        cy -= ry
        cy2 = cy * cy
        cy3 = cy * cy2
        ry -= 1

        if (rx >= 1) and (rx < width - 2) and (ry >= 1) and (ry < height - 2):
            # This is the general cases (i.e. the 'middle' of the image)
            # The other cases are deported in the function 'inter_bicubic_other_cases_gray_u8'
            # (it makes the inner code of the loop smaller and consequently
            # reduces the number of 'instruction cache misses')
            v00 = img[ry, rx]
            v10 = img[ry, rx + 1]
            v20 = img[ry, rx + 2]
            v30 = img[ry, rx + 3]

            v01 = img[ry + 1, rx]
            v11 = img[ry + 1, rx + 1]
            v21 = img[ry + 1, rx + 2]
            v31 = img[ry + 1, rx + 3]

            v02 = img[ry + 2, rx]
            v12 = img[ry + 2, rx + 1]
            v22 = img[ry + 2, rx + 2]
            v32 = img[ry + 2, rx + 3]

            v03 = img[ry + 3, rx]
            v13 = img[ry + 3, rx + 1]
            v23 = img[ry + 3, rx + 2]
            v33 = img[ry + 3, rx + 3]

            p0 = 2. * cy2 - cy3 - cy
            p1 = 3. * cy3 - 5. * cy2 + 2
            p2 = 4. * cy2 - 3. * cy3 + cy
            p3 = cy3 - cy2

            v0 = v00 * p0 + v01 * p1 + v02 * p2 + v03 * p3
            v1 = v10 * p0 + v11 * p1 + v12 * p2 + v13 * p3
            v2 = v20 * p0 + v21 * p1 + v22 * p2 + v23 * p3
            v3 = v30 * p0 + v31 * p1 + v32 * p2 + v33 * p3

            val[i] = (v0 * (2. * cx2 - cx3 - cx)
                      + v1 * (3. * cx3 - 5. * cx2 + 2.)
                      + v2 * (4. * cx2 - 3. * cx3 + cx)
                      + v3 * (cx3 - cx2)) / 4.
        else:
            val[i] = interp_bicubic_other_cases_gray_u8_native(img, default_value, rx, ry, cx, cx2, cx3, cy, cy2, cy3)

    return val
