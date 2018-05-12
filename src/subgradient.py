# -*- coding: utf-8 -*-
# subgradient.py: Subgradient algorithm for solving lagrangian relaxation
# authors: Antoine Passemiers, Cedric Simar

from decomposition import decompose_problem

import pulp
import time
import numpy as np


def solve_subproblem(subproblem):
    """ Solve sub-problem of the lagrangian decomposition of SUC.

    Args:
        subproblem (SUCLpProblem): Sub-problem to be solved
    """
    start = time.time()
    status = subproblem.solve()
    #assert(subproblem.status == pulp.constants.LpStatusOptimal)
    exec_time = time.time() - start
    #print("\tSub-problem %s - Solve time: %f s (status: %i)" % \
    #    (subproblem.name, exec_time, status))


def solve_with_subgradient(instance, _lambda=0.1):
    PI = instance.PI
    Gs = instance.Gs
    K = instance.K
    S = instance.S
    C = instance.C

    # Lagrangian multipliers
    mu = np.zeros((instance.n_generators, instance.n_scenarios, instance.n_periods))
    nu = np.zeros((instance.n_generators, instance.n_scenarios, instance.n_periods))

    for k in range(20):
        P1, P2, ED, u, v, w, z, p = decompose_problem(instance, mu, nu)
        for s in range(instance.n_scenarios):
            # TODO: paralléliser
            solve_subproblem(P1[s])
        solve_subproblem(P2)

        L_k = P2.objective.value() if P2.objective.value() else 0
        for subproblem in P1:
            L_k += subproblem.objective.value()

        u_k = np.swapaxes(u[Gs, :, :].get_var_values(), 2, 1)
        v_k = np.swapaxes(v[Gs, :, :].get_var_values(), 2, 1)
        w_k = w[Gs, :].get_var_values()[..., np.newaxis]
        z_k = z[Gs, :].get_var_values()[..., np.newaxis]

        for s in range(instance.n_scenarios):
            for g in Gs:
                for t in range(instance.n_periods):
                    w[g, t].lowBound = w[g, t].upBound = w[g, t].varValue
                    z[g, t].lowBound = z[g, t].upBound = z[g, t].varValue
            solve_subproblem(ED[s])

        obj = np.sum(PI * np.swapaxes(
            K * np.swapaxes(u.get_var_values(), 0, 2) + \
            S * np.swapaxes(v.get_var_values(), 0, 2) + \
            C * np.swapaxes(p.get_var_values(), 0, 2), 1, 2))

        print("L_hat: %f, L_k: %f" % (obj, L_k))
        L_hat = obj
        alpha_k = _lambda * (L_hat - L_k) / np.sum(PI**2 * (u_k - w_k)**2 + \
            PI**2 * (v_k - z_k)**2)

        mu[Gs, :, :] = mu[Gs, :, :] + np.swapaxes(alpha_k * PI * (w_k - u_k), 2, 1)
        nu[Gs, :, :] = nu[Gs, :, :] + np.swapaxes(alpha_k * PI * (z_k - v_k), 2, 1)
