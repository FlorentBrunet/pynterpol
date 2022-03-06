# pynterpol

Fast image interpolation in Python.

## Installation

```shell
python -m pip install pynterpol
```

## Usage (example)

```python
from src.pynterpol import interp_bilinear_u8
import cv2

img = cv2.imread('image.jpg')  # or any other way of reading an image as a numpy ndarray

interpolated_values = interp_bilinear_u8(img, [100.1, 100.8], [50.5, 51.2])
```

## Performances

High performance is achieved:

- by considering that the image to interpolate is defined on an orthogonal equally-spaced regular grid, and
- by using [Numba](http://numba.pydata.org/) for the loop-intensive parts of the computation, and
- by optimizing the algorithm implementation (avoid code cache misses for special cases, etc.)

### Comparative Timings

Timings realized on an Apple MacBook Pro (Intel Core i9 8 cores, 2.3 GHz)
with an gray (i.e. single channel) image of size 1904x1081x1 interpolated on a regular equally-spaced grid of size
3807x2161 that covers the entire domain of the image
(that corresponds to a x2 upsampling of the input image).

| method | timings |
|--------|---------|
| SciPy `RegularGridInterpolator` | 1614 ms |
| `interp_bilinear_u8` (without Numba) | 82980 ms (LOL) |
| `interp_bilinear_u8` (with Numba) | **65 ms** (almost x25 speedup)|

## Reference

As of 2022-03-06, only images represented as a NumPy ndarray with uint8 as dtype are supported. Interpolated values are
stored as float64. All of this is implemented with a single function: `interp_bilinear_u8`.

Other cases may be considered in the future:

- bicubic interpolation or other interpolation schemes
- image represented with different types (float, bigger uints, ...)
- integer (or smaller floats) interpolated values

## Tests

```shell
cd $PROJECT_ROOT
PYTHONPATH=./src python -m unittest tests.test_pynterpol.TestPynterpol
```

Note: There may be a better way than providing the `PYTHONPATH` variable in the command...

## Distribution

```shell
python -m pip install --upgrade build twine
python -m pip build
python -m twine upload --repository testpypi dist/*
```
