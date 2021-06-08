import numpy as np


class AbstractDirectionSelectionMethod:
    def __init__(self, f, x0):
        self.f = f
        self.x_curr = x0

    def get_direction(self):
        pass

    def update(self, x_new):
        pass


class GradientDescent(AbstractDirectionSelectionMethod):
    def __init__(self, f, x0):
        super().__init__(f, x0)
        self.f_curr = self.gradient_curr = None
        self.update(x0)

    def get_direction(self):
        return -self.gradient_curr

    def update(self, x_new):
        self.x_curr = x_new
        self.f_curr, self.gradient_curr, _ = self.f(self.x_curr)


class BFGS(AbstractDirectionSelectionMethod):
    def __init__(self, f, x0):
        super().__init__(f, x0)
        self.f_curr, self.gradient_curr, _ = self.f(self.x_curr)
        self.hessian_approx_curr = np.eye(
            x0.shape[0]
        )  # Initial Hessian approximation (identity matrix)

    def get_direction(self):
        return np.linalg.solve(self.hessian_approx_curr, -self.gradient_curr)

    def update(self, x_new):
        x_prev = self.x_curr
        gradient_prev = self.gradient_curr

        self.x_curr = x_new
        self.f_curr, self.gradient_curr, _ = self.f(self.x_curr)
        y = (self.gradient_curr - gradient_prev).reshape((-1, 1))
        s = (self.x_curr - x_prev).reshape((-1, 1))

        # Update Hessian approximation according to BFGS update rule
        self.hessian_approx_curr = (
            self.hessian_approx_curr
            - (
                (self.hessian_approx_curr @ s @ s.T @ self.hessian_approx_curr)
                / (s.T @ self.hessian_approx_curr @ s)
            )
            + (y @ y.T) / (y.T @ s)
        )


class NewtonsMethod(AbstractDirectionSelectionMethod):
    def __init__(self, f, x0):
        super().__init__(f, x0)
        self.f_curr = self.gradient_curr = self.hessian_curr = None
        self.update(x0)

    def get_direction(self):
        return np.linalg.solve(self.hessian_curr, -self.gradient_curr)

    def update(self, x_new):
        self.x_curr = x_new
        self.f_curr, self.gradient_curr, self.hessian_curr = self.f(
            self.x_curr, return_hessian=True
        )


class EqualityConstrainedNewtonsMethod(AbstractDirectionSelectionMethod):
    def __init__(self, f, x0, A):
        super().__init__(f, x0)
        self.A = A
        self.f_curr = self.gradient_curr = self.hessian_curr = None
        self.update(x0)

    def get_direction(self):
        linear_system_lhs = np.vstack(
            [
                np.hstack([self.hessian_curr, self.A.T]),
                np.hstack([self.A, np.zeros((1, self.A.shape[0]))]),
            ]
        )
        linear_system_rhs = np.concatenate(
            [-self.gradient_curr, np.zeros(self.A.shape[0])]
        )
        return np.linalg.solve(linear_system_lhs, linear_system_rhs)[:self.A.shape[1]]

    def update(self, x_new):
        self.x_curr = x_new
        self.f_curr, self.gradient_curr, self.hessian_curr = self.f(
            self.x_curr, return_hessian=True
        )


def get_direction_select_method(dir_selection_method):
    return {
        "gd": GradientDescent,
        "nt": NewtonsMethod,
        "bfgs": BFGS,
        "eq_nt": EqualityConstrainedNewtonsMethod,
    }[dir_selection_method]
