import unittest
from src.unconstrained_min import line_search
from src.utils import visualize_path
from tests.examples import quad_i, quad_ii, quad_iii, rosenbrock, linear
import numpy as np


class TestBFGSMinimization(unittest.TestCase):
    def test_quad_min(self):
        x_final, success, x_hist, f_hist = line_search(
            quad_i, [1, 1], 1e-12, 1e-8, 100, "bfgs", 1, 1e-4, 0.2
        )
        visualize_path(quad_i, x_hist, f_hist, "Quad i [BFGS]")
        self.assertTrue(success)
        assert np.allclose(x_final, [0, 0], atol=1e-5)

        x_final, success, x_hist, f_hist = line_search(
            quad_ii, [1, 1], 1e-12, 1e-8, 100, "bfgs", 1, 1e-4, 0.2
        )
        visualize_path(quad_ii, x_hist, f_hist, "Quad ii [BFGS]")
        self.assertTrue(success)
        assert np.allclose(x_final, [0, 0], atol=1e-5)

        x_final, success, x_hist, f_hist = line_search(
            quad_iii, [1, 1], 1e-12, 1e-8, 100, "bfgs", 1, 1e-4, 0.2
        )
        visualize_path(quad_iii, x_hist, f_hist, "Quad iii [BFGS]")
        self.assertTrue(success)
        assert np.allclose(x_final, [0, 0], atol=1e-5)

    def test_rosenbrock_min(self):
        x_final, success, x_hist, f_hist = line_search(
            rosenbrock, [2, 2], 1e-7, 1e-8, 10000, "bfgs", 1, 1e-4, 0.2
        )
        visualize_path(
            rosenbrock,
            x_hist,
            f_hist,
            "Rosenbrock [BFGS]",
            margin=1.5,
            levels=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200, 250, 300, 500, 750, 1000],
        )
        self.assertTrue(success)
        assert np.allclose(x_final, [1, 1], atol=1e-5)


class TestNewtonMinimization(unittest.TestCase):
    def test_quad_min(self):
        x_final, success, x_hist, f_hist = line_search(
            quad_i, [1, 1], 1e-12, 1e-8, 100, "nt", 1, 1e-4, 0.2
        )
        visualize_path(quad_i, x_hist, f_hist, "Quad i [NT]")
        self.assertTrue(success)
        assert np.allclose(x_final, [0, 0], atol=1e-5)

        x_final, success, x_hist, f_hist = line_search(
            quad_ii, [1, 1], 1e-12, 1e-8, 100, "nt", 1, 1e-4, 0.2
        )
        visualize_path(quad_ii, x_hist, f_hist, "Quad ii [NT]")
        self.assertTrue(success)
        assert np.allclose(x_final, [0, 0], atol=1e-5)

        x_final, success, x_hist, f_hist = line_search(
            quad_iii, [1, 1], 1e-12, 1e-8, 100, "nt", 1, 1e-4, 0.2
        )
        visualize_path(quad_iii, x_hist, f_hist, "Quad iii [NT]")
        self.assertTrue(success)
        assert np.allclose(x_final, [0, 0], atol=1e-5)

    def test_rosenbrock_min(self):
        x_final, success, x_hist, f_hist = line_search(
            rosenbrock, [2, 2], 1e-7, 1e-8, 10000, "nt", 1, 1e-4, 0.2
        )
        visualize_path(
            rosenbrock,
            x_hist,
            f_hist,
            "Rosenbrock [NT]",
            margin=1.5,
            levels=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200, 250, 300, 500, 750, 1000],
        )
        self.assertTrue(success)
        assert np.allclose(x_final, [1, 1], atol=1e-5)


class TestGDMinimization(unittest.TestCase):
    def test_quad_min(self):
        x_final, success, x_hist, f_hist = line_search(
            quad_i, [1, 1], 1e-12, 1e-8, 100, "gd", 0.1, 0, 0
        )
        visualize_path(quad_i, x_hist, f_hist, "Quad i [GD]")
        self.assertTrue(success)
        assert np.allclose(x_final, [0, 0], atol=1e-5)

        x_final, success, x_hist, f_hist = line_search(
            quad_ii, [1, 1], 1e-12, 1e-8, 100, "gd", 0.1, 0, 0
        )
        visualize_path(quad_ii, x_hist, f_hist, "Quad ii [GD]")
        self.assertTrue(success)
        assert np.allclose(x_final, [0, 0], atol=1e-5)

        x_final, success, x_hist, f_hist = line_search(
            quad_iii, [1, 1], 1e-12, 1e-8, 100, "gd", 0.1, 0, 0
        )
        visualize_path(quad_iii, x_hist, f_hist, "Quad iii [GD]")
        self.assertTrue(success)
        assert np.allclose(x_final, [0, 0], atol=1e-5)

    def test_rosenbrock_min(self):
        x_final, success, x_hist, f_hist = line_search(
            rosenbrock, [2, 2], 1e-7, 1e-8, 10000, "gd", 0.002, 0, 0.5
        )
        visualize_path(
            rosenbrock,
            x_hist,
            f_hist,
            "Rosenbrock [GD]",
            margin=1.5,
            levels=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200, 250, 300, 500, 750, 1000],
        )
        self.assertTrue(success)
        assert np.allclose(x_final, [1, 1], atol=2 * 1e-2)

    def test_lin_min(self):
        x_final, success, x_hist, f_hist = line_search(
            linear, [1, 1], 1e-7, 1e-8, 100, "gd", 0.1, 0, 0
        )
        visualize_path(linear, x_hist, f_hist, "Linear [GD]")
        self.assertFalse(success)


if __name__ == "__main__":
    unittest.main()
