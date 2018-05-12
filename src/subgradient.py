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
    print(status)
    #assert(subproblem.status == pulp.constants.LpStatusOptimal)
    exec_time = time.time() - start
    print("\tSub-problem %s - Solve time: %f s" % (subproblem.name, exec_time))


def solve_with_subgradient(instance, lambda_=1000.0):
    PI = instance.PI
    Gs = instance.Gs
    G = instance.n_generators
    S = instance.n_scenarios
    T = instance.n_periods

    # Lagrangian multipliers
    mu = np.zeros((G, S, T))
    nu = np.zeros((G, S, T))

    for k in range(10):
        P1, P2, u, v, w, z = decompose_problem(instance, mu, nu)
        for i in range(len(P1)):
            solve_subproblem(P1[i])
        solve_subproblem(P2)

        L_k = P2.objective.value() if P2.objective.value() else 0
        for subproblem in P1:
            if subproblem.objective.value():
                L_k += subproblem.objective.value()

        print("Lk: %f" % L_k)
        u_k = np.swapaxes(u[Gs, :, :].get_var_values(), 2, 1)
        v_k = np.swapaxes(v[Gs, :, :].get_var_values(), 2, 1)
        w_k = w[Gs, :].get_var_values()[..., np.newaxis]
        z_k = z[Gs, :].get_var_values()[..., np.newaxis]
        alpha_k = 1000. # TODO

        print(mu.shape, w_k.shape,u_k.shape)
        mu[Gs, :, :] += np.swapaxes(alpha_k * PI * (w_k - u_k), 2, 1)
        nu[Gs, :, :] += np.swapaxes(alpha_k * PI * (z_k - v_k), 2, 1)
