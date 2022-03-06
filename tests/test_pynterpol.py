import unittest
import numpy as np
from scipy import interpolate
from parameterized import parameterized

from pynterpol import interp_bilinear_u8


class TestPynterpol(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.small_gray = np.arange(1, 49, dtype=np.uint8).reshape((6, 8))
        cls.small_color = np.dstack([
            np.arange(1, 49, dtype=np.uint8).reshape(6, 8),
            np.arange(101, 149, dtype=np.uint8).reshape(6, 8),
            np.arange(201, 249, dtype=np.uint8).reshape(6, 8)
        ])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.small_gray = None
        cls.small_color = None

    def test_interp_bilinear_u8_type_gray(self):
        values = interp_bilinear_u8(self.small_gray, np.array([0, 1, 2]), np.array([0, 1, 2]))
        self.assertEqual(np.float64, values.dtype)

    def test_interp_bilinear_u8_type_color(self):
        values = interp_bilinear_u8(self.small_color, np.array([0, 1, 2]), np.array([0, 1, 2]))
        self.assertEqual(np.float64, values.dtype)

    @parameterized.expand([
        [0, 0],
        [[1, 2, 3], [1, 2, 3]],
        [np.array([1, 2, 3]), np.array([0, 1, 2])],
        [np.array([1, 2, 3, 1, 2, 3]), np.array([0, 1, 2, 0, 1, 2])],
        [np.array([-1, 2, 3, 1, 2, 3]), np.array([[0, 1, -2], [0, 1, 2]])],
        [np.array(1), np.array(2)],
        [np.array([[1, 2], [3, 4]]), np.array([[0, 2], [4, 1]])],
        [np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]), np.array([[[0, 2], [4, 1]], [[0, 2], [4, 1]]])]
    ])
    def test_interp_bilinear_u8_shape_gray(self, x, y):
        values = interp_bilinear_u8(self.small_gray, x, y)
        self.assertEqual(np.array(x).shape, values.shape)

    @parameterized.expand([
        [(1, 1, 3), 2, 4],
        [(1, 2, 3), [1, 2], [2, 3]],
        [(2, 1, 3), [[1], [2]], [[2], [3]]],
    ])
    def test_interp_bilinear_u8_shape_color(self, expected, x, y):
        val = interp_bilinear_u8(self.small_color, x, y)
        self.assertEqual(expected, val.shape)

    def test_interp_bilinear_u8_all_int_gray(self):
        h = self.small_gray.shape[0]
        w = self.small_gray.shape[1]

        x, y = np.meshgrid(np.arange(w), np.arange(h))

        val = interp_bilinear_u8(self.small_gray, x, y)

        self.assertTrue(np.all(self.small_gray == val))

    def test_interp_bilinear_u8_all_int_color(self):
        h = self.small_gray.shape[0]
        w = self.small_gray.shape[1]

        x, y = np.meshgrid(np.arange(w), np.arange(h))

        val = interp_bilinear_u8(self.small_color, x, y)

        self.assertEqual((h, w, 3), val.shape)
        self.assertTrue(np.all(self.small_color == val))

    def test_interp_bilinear_u8_default_gray_01(self):
        self.assertEqual(0, interp_bilinear_u8(self.small_gray, -1, -10))

    def test_interp_bilinear_u8_default_gray_02(self):
        self.assertEqual(10, interp_bilinear_u8(self.small_gray, -1, -10, 10))

    def test_interp_bilinear_u8_default_color_01(self):
        val = interp_bilinear_u8(self.small_color, -2, -3)
        self.assertEqual(0, val[0, 0, 0])
        self.assertEqual(0, val[0, 0, 1])
        self.assertEqual(0, val[0, 0, 2])

    def test_interp_bilinear_u8_default_color_02(self):
        val = interp_bilinear_u8(self.small_color, -2, -3, [10, 20, 30])
        self.assertEqual(10, val[0, 0, 0])
        self.assertEqual(20, val[0, 0, 1])
        self.assertEqual(30, val[0, 0, 2])

    def test_interp_bilinear_u8_default_color_03(self):
        val = interp_bilinear_u8(self.small_color, [-2, 100], [-3, 200], [10, 20, 30])
        self.assertEqual(10, val[0, 0, 0])
        self.assertEqual(20, val[0, 0, 1])
        self.assertEqual(30, val[0, 0, 2])

        self.assertEqual(10, val[0, 1, 0])
        self.assertEqual(20, val[0, 1, 1])
        self.assertEqual(30, val[0, 1, 2])

    def test_interp_bilinear_u8_default_color_04(self):
        val = interp_bilinear_u8(
            self.small_color,
            np.array([-2, 100]).reshape((2, 1)),
            np.array([-3, 200]).reshape((2, 1)),
            [10, 20, 30]
        )
        self.assertEqual(10, val[0, 0, 0])
        self.assertEqual(20, val[0, 0, 1])
        self.assertEqual(30, val[0, 0, 2])

        self.assertEqual(10, val[1, 0, 0])
        self.assertEqual(20, val[1, 0, 1])
        self.assertEqual(30, val[1, 0, 2])

    @parameterized.expand([
        [1.5, 0.5, 0],
        [5, 0, 0.5],
        [5.5, 0.5, 0.5]
    ])
    def test_interp_bilinear_u8_frac_val_gray(self, expected, x, y):
        val = interp_bilinear_u8(self.small_gray, x, y)
        self.assertEqual(expected, val)

    @parameterized.expand([
        [[1.5, 101.5, 201.5], 0.5, 0],
        [[5, 105, 205], 0, 0.5],
        [[5.5, 105.5, 205.5], 0.5, 0.5]
    ])
    def test_interp_bilinear_u8_frac_val_color(self, expected, x, y):
        val = interp_bilinear_u8(self.small_color, x, y)
        val = val.reshape(-1)
        self.assertTrue(np.all(np.array(expected) == val))

    @parameterized.expand([
        [0, 0],
        [5, 0],
        [0.1, 2.9],
        [4.01, 3.99],
        [0.01, 0.01],
        [2.5, 3.5]
    ])
    def test_interp_bilinear_u8_scipy_gray(self, x, y):
        h = self.small_gray.shape[0]
        w = self.small_gray.shape[1]
        rgi = interpolate.RegularGridInterpolator((np.arange(h), np.arange(w)), self.small_gray)
        val_rgi = rgi((y, x))
        val_own = interp_bilinear_u8(self.small_gray, x, y)
        self.assertAlmostEqual(val_rgi, val_own, delta=1.0e-12)

    @parameterized.expand([
        [np.float64],
        [np.float32],
        [np.float16],
        [np.int64],
        [np.int32],
        [np.int16],
        [np.int8],
        [np.uint32],
        [np.uint16]
    ])
    def test_interp_bilinear_u8_error_not_u8(self, t):
        self.assertRaises(
            TypeError,
            lambda: interp_bilinear_u8(np.ones((6, 8), dtype=t), 0, 0)
        )

    @parameterized.expand([
        [0, [0, 1]],
        [[1, 2], 3],
        [np.array([1, 2]), np.array(3)],
        [np.arange(5), np.arange(6)],
        [np.arange(5).reshape((5, 1)), np.arange(6)]
    ])
    def test_interp_bilinear_u8_error_different_size(self, x, y):
        self.assertRaises(
            TypeError,
            lambda: interp_bilinear_u8(self.small_gray, x, y)
        )

    def test_interp_bilinear_u8_error_default_wrong_size_01(self):
        self.assertRaises(
            TypeError,
            lambda: interp_bilinear_u8(self.small_color, [1, 2], [3, 4], np.array([1, 2]))
        )

    def test_interp_bilinear_u8_error_default_wrong_size_02(self):
        self.assertRaises(
            TypeError,
            lambda: interp_bilinear_u8(self.small_color, [1, 2], [3, 4], np.array([1, 2, 3, 4]))
        )

    def test_interp_bilinear_u8_error_default_two_many_dimensions(self):
        self.assertRaises(
            TypeError,
            lambda: interp_bilinear_u8(np.ones([2, 3, 4, 5], dtype=np.uint8), [1, 2], [3, 4])
        )

    def test_interp_bilinear_u8_error_default_not_enough_dimensions(self):
        self.assertRaises(
            TypeError,
            lambda: interp_bilinear_u8(np.arange(10, dtype=np.uint8), [1, 2], [3, 4])
        )


if __name__ == '__main__':
    unittest.main()
