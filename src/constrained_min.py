from src.unconstrained_min import line_search


def interior_pt(
    func,
    ineq_constraints,
    eq_constraints_mat,
    eq_constraints_rhs,
    x0,
    max_outer_iter,
    eps,
    init_t,
    mu,
    obj_tol,
    param_tol,
    init_step_len,
    slope_ratio,
    back_track_factor,
    max_inner_iter,
):
    t = init_t
    x_curr = x0.copy()
    x_hist = []
    f_hist = []
    for i in range(1, max_outer_iter + 1):

        def curr_unconstrained_problem(x, **kwargs):
            f_curr, gradient_curr, hessian = func(x, return_hessian=True)
            f_curr *= t
            gradient_curr *= t
            hessian *= t
            for ineq_constraint in ineq_constraints:
                ineq_f, ineq_gradient, ineq_hessian = ineq_constraint(
                    x, return_hessian=True
                )
                f_curr -= ineq_f
                gradient_curr -= ineq_gradient
                hessian -= ineq_hessian

            return f_curr, gradient_curr, hessian

        solver_args = []
        dir_selection_method = "nt"
        if eq_constraints_mat is not None:
            solver_args.append(eq_constraints_mat)
            dir_selection_method = "eq_nt"

        x_curr, success, temp_x_hist, temp_f_hist = line_search(
            curr_unconstrained_problem,
            x_curr,
            obj_tol,
            param_tol,
            max_inner_iter,
            dir_selection_method,
            init_step_len,
            slope_ratio,
            back_track_factor,
            *solver_args,
            verbose=False
        )

        x_hist += temp_x_hist
        f_hist += temp_f_hist

        if len(ineq_constraints) / t < eps:
            break

        t *= mu

    return x_curr, x_hist, f_hist
