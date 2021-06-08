import numpy as np


def quad_symmetric(Q, x, return_hessian=False):
    hessian = None
    if return_hessian:
        hessian = 2 * Q

    return x.T @ Q @ x, 2 * Q @ x, hessian


def quad_i(x, return_hessian=False):
    Q = np.eye(2)
    return quad_symmetric(Q, x, return_hessian=return_hessian)


def quad_ii(x, return_hessian=False):
    Q = np.diag([5, 1])
    return quad_symmetric(Q, x, return_hessian=return_hessian)


def quad_iii(x, return_hessian=False):
    A = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
    B = np.diag([5, 1])
    Q = A.T @ B @ A
    return quad_symmetric(Q, x, return_hessian=return_hessian)


def rosenbrock(x, return_hessian=False):
    x1, x2 = x
    hessian = None
    if return_hessian:
        hessian = np.array([[1200 * x1 ** 2 - 400 * x2, -400 * x1], [-400 * x1, 200]])

    return (
        100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2,
        np.array(
            [400 * x1 ** 3 - 400 * x1 * x2 + 2 * x1 - 2, 200 * x2 - 200 * x1 ** 2]
        ),
        hessian,
    )


def linear(x, return_hessian=False):
    a = np.array([-2, 3])
    hessian = None
    if return_hessian:
        hessian = np.zeros((2, 2))

    return a @ x, a, hessian


def qp_example():
    def objective(x, return_hessian=False):
        hessian = None
        if return_hessian:
            hessian = np.eye(3) * 2

        return (
            ((x + np.array([0, 0, 1])) ** 2).sum(),
            2 * (x + np.array([0, 0, 1])),
            hessian,
        )

    def ineq1(x, return_hessian=False):
        hessian = None
        if return_hessian:
            hessian = np.zeros((3, 3))
            hessian[0, 0] = -1 / (x[0] ** 2)

        return np.log(x[0]), np.array([1 / x[0], 0, 0]), hessian

    def ineq2(x, return_hessian=False):
        hessian = None
        if return_hessian:
            hessian = np.zeros((3, 3))
            hessian[1, 1] = -1 / (x[1] ** 2)

        return np.log(x[1]), np.array([0, 1 / x[1], 0]), hessian

    def ineq3(x, return_hessian=False):
        hessian = None
        if return_hessian:
            hessian = np.zeros((3, 3))
            hessian[2, 2] = -1 / (x[2] ** 2)

        return np.log(x[2]), np.array([0, 0, 1 / x[2]]), hessian

    return {
        "f0": objective,
        "ineq_constraints": [ineq1, ineq2, ineq3],
        "A": np.array([[1, 1, 1]]),
        "b": np.array([1]),
    }


def lp_example():
    def objective(x, return_hessian=False):
        hessian = None
        if return_hessian:
            hessian = np.zeros((2, 2))

        return -x.sum(), -np.ones(2), hessian

    def ineq1(x, return_hessian=False):
        hessian = None
        intermediate_res = x.sum() - 1
        if return_hessian:
            hessian = -np.ones((2, 2)) / (intermediate_res ** 2)

        return np.log(intermediate_res), np.ones(2) / intermediate_res, hessian

    def ineq2(x, return_hessian=False):
        hessian = None
        if return_hessian:
            hessian = np.array([[0, 0], [0, -1 / ((1 - x[1]) ** 2)]])

        return np.log(1 - x[1]), np.array([0, 1 / (1 - x[1])]), hessian

    def ineq3(x, return_hessian=False):
        hessian = None
        if return_hessian:
            hessian = np.array([[-1 / ((x[0] - 2) ** 2), 0], [0, 0]])

        return np.log(2 - x[0]), np.array([1 / (x[0] - 2), 0]), hessian

    def ineq4(x, return_hessian=False):
        hessian = None
        if return_hessian:
            hessian = np.array([[0, 0], [0, -1 / (x[1] ** 2)]])

        return np.log(x[1]), np.array([0, 1 / x[1]]), hessian

    return {
        "f0": objective,
        "ineq_constraints": [ineq1, ineq2, ineq3, ineq4],
        "A": None,
        "b": None,
    }
