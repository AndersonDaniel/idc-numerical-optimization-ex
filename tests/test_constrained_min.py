import unittest
import numpy as np
from src.constrained_min import interior_pt
from tests.examples import qp_example, lp_example
from src.utils import visualize_lp_path, visualize_qp_path


class TestConstrainedMinimization(unittest.TestCase):
    def test_qp(self):
        qp_problem = qp_example()
        x, x_hist, f_hist = interior_pt(
            qp_problem["f0"],
            qp_problem["ineq_constraints"],
            qp_problem["A"],
            qp_problem["b"],
            np.array([0.1, 0.2, 0.7]),
            100,
            1e-9,
            1,
            10,
            1e-9,
            1e-9,
            1,
            1e-4,
            0.2,
            1000,
        )

        objective_value = qp_problem["f0"](x)[0]
        inequality_constraint_values = [
            -np.exp(inequality_constraint(x)[0])
            for inequality_constraint in qp_problem["ineq_constraints"]
        ]

        print(f"Objective: {objective_value}, final candidate: {x_hist[-1].tolist()}")
        for i, v in enumerate(inequality_constraint_values):
            print(f"Inequality constraint #{i + 1} value: {v:.3f}")

        visualize_qp_path(x_hist)

        assert np.allclose(x, [0.5, 0.5, 0])

    def test_lp(self):
        lp_problem = lp_example()
        x, x_hist, f_hist = interior_pt(
            lp_problem["f0"],
            lp_problem["ineq_constraints"],
            lp_problem["A"],
            lp_problem["b"],
            np.array([0.5, 0.75]),
            100,
            1e-9,
            1,
            10,
            1e-9,
            1e-9,
            1,
            1e-4,
            0.2,
            1000,
        )

        # Turned a maximization problem to a minimization problem, recover original objective
        objective_value = -lp_problem["f0"](x)[0]
        inequality_constraint_values = [
            -np.exp(inequality_constraint(x)[0])
            for inequality_constraint in lp_problem["ineq_constraints"]
        ]

        print(f"Objective: {objective_value}, final candidate: {x_hist[-1].tolist()}")
        for i, v in enumerate(inequality_constraint_values):
            print(f"Inequality constraint #{i + 1} value: {v:.3f}")

        visualize_lp_path(x_hist)

        assert np.allclose(x, [2, 1])


if __name__ == "__main__":
    unittest.main()
