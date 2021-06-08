import numpy as np
from src.utils import report
from src.direction_selection_methods import get_direction_select_method


def line_search(
    f,
    x0,
    obj_tol,
    param_tol,
    max_iter,
    dir_selection_method,
    init_step_len,
    slope_ratio,
    back_track_factor,
    *solver_args,
    verbose=True
):
    x_hist = []
    f_hist = []
    f_prev = np.inf
    succeeded = False
    direction_selection_method = get_direction_select_method(dir_selection_method)(
        f, np.copy(x0), *solver_args
    )
    for i in range(1, max_iter + 1):
        x_curr = direction_selection_method.x_curr
        f_curr = direction_selection_method.f_curr
        x_hist.append(x_curr)
        f_hist.append(f_curr)
        step_direction = direction_selection_method.get_direction()
        step_size = get_step_size(
            f,
            f_curr,
            x_curr,
            direction_selection_method.gradient_curr,
            step_direction,
            slope_ratio,
            init_step_len,
            back_track_factor,
        )
        x_new = x_curr + step_size * step_direction
        direction_selection_method.update(x_new)

        if verbose:
            report(i, x_new, x_curr, f_curr, f_prev)

        if f_curr < f_prev and np.abs(f_prev - f_curr) < obj_tol:
            succeeded = True
            if verbose:
                print("Success - objective tolerance convergence")
            break
        if np.linalg.norm(x_new - x_curr) < param_tol:
            succeeded = True
            if verbose:
                print("Success - parameter tolerance convergence")
            break

        f_prev = f_curr

    if not succeeded and verbose:
        print("Failure - optimization didn't converge")

    return direction_selection_method.x_curr, succeeded, x_hist, f_hist


def get_step_size(
    f,
    f_curr,
    x_curr,
    gradient_curr,
    step_direction,
    slope_ratio,
    init_step_len,
    back_track_factor,
):
    curr_step_size = init_step_len
    # "Infinite loop" but first Wolfe condition is guaranteed to be eventually satisfied by a small enough step size
    while True:
        x_temp = x_curr + curr_step_size * step_direction
        f_temp, _, _ = f(x_temp)
        if f_temp <= f_curr + slope_ratio * curr_step_size * (
            gradient_curr @ step_direction
        ):
            # First Wolfe condition satisfied
            return curr_step_size

        curr_step_size *= back_track_factor
