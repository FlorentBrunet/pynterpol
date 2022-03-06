import unittest
import numpy as np
from parameterized import parameterized

from pynterpol import interp_bilinear_u8


class TestPynterpol(unittest.TestCase):

    def setUp(self) -> None:
        self.small_gray = np.arange(1, 49, dtype=np.uint8).reshape((6, 8))
        print(self.small_gray)

    def test_interp_bilinear_u8_type(self):
        values = interp_bilinear_u8(self.small_gray, np.array([0, 1, 2]), np.array([0, 1, 2]))
        self.assertEqual(np.float64, values.dtype)

    @parameterized.expand([
        [np.array([1, 2, 3]), np.array([0, 1, 2])],
        [np.array([1, 2, 3, 1, 2, 3]), np.array([0, 1, 2, 0, 1, 2])],
        [np.array([-1, 2, 3, 1, 2, 3]), np.array([[0, 1, -2], [0, 1, 2]])],
        [np.array(1), np.array(2)],
        [np.array([[1, 2], [3, 4]]), np.array([[0, 2], [4, 1]])],
        [np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]), np.array([[[0, 2], [4, 1]], [[0, 2], [4, 1]]])],
        [[1,2,3],[1,2,3]]
    ])
    def test_interp_bilinear_u8_shape(self, x, y):
        values = interp_bilinear_u8(self.small_gray, x, y)
        self.assertEqual(x.shape, values.shape)

    def test_interp_bilinear_u8_all_int(self):
        h = self.small_gray.shape[0]
        w = self.small_gray.shape[1]

        x, y = np.meshgrid(np.arange(w), np.arange(h))

        val = interp_bilinear_u8(self.small_gray, x, y)

        self.assertTrue(np.all(self.small_gray == val))


if __name__ == '__main__':
    unittest.main()
