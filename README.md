# pynterpol

Fast image interpolation in Python.

## Installation

```shell
python -m pip install pynterpol
```

## Usage

```python
from src.pynterpol import interp_bilinear_u8
import cv2

img = cv2.imread('image.jpg')

interpolated_values = interp_bilinear_u8(img, [100.1, 100.8], [50.5, 51.2])
```

## Distribution

```shell
python -m pip install --upgrade build twine
python -m pip build
python -m twine upload --repository testpypi dist/*
```
