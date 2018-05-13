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
    assert(status == pulp.constants.LpStatusOptimal)
    #assert(subproblem.status == pulp.constants.LpStatusOptimal)
    exec_time = time.time() - start
    #print("\tSub-problem %s - Solve time: %f s (status: %i)" % \
    #    (subproblem.name, exec_time, status))


def solve_with_subgradient(instance, _lambda=2.):
    PI = instance.PI
    Gs = instance.Gs
    K = instance.K
    S = instance.S
    C = instance.C

    # Lagrangian multipliers
    mu = np.zeros((instance.n_generators, instance.n_scenarios, instance.n_periods))
    nu = np.zeros((instance.n_generators, instance.n_scenarios, instance.n_periods))

    for k in range(20):
        PP, P1, P2, ED, variables = decompose_problem(instance, mu, nu)
        (u, v, p, theta, w, z, e) = variables
        for s in range(instance.n_scenarios):
            # TODO: paralléliser
            solve_subproblem(P1[s])
        solve_subproblem(P2)

        L_k = P2.objective.value() if P2.objective.value() else 0
        for s in range(instance.n_scenarios):
            L_k += P1[s].objective.value()

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

        L_hat = PP.objective.value()

        """
        n_violated, groups_n_violated = PP.constraints_violated()
        print("Number of violated constraints: %i" % n_violated)
        print(PP.is_integer_solution(), n_violated)
        for group in groups_n_violated.keys():
            if groups_n_violated[group][0] > 0:
                print("Number of violated constraints of group %s: %i / %i" % (
                    group, groups_n_violated[group][0], groups_n_violated[group][1]))
        """

        print("L_hat: %f, L_k: %f" % (L_hat, L_k))
        alpha_k = -0.5*_lambda * (L_hat - L_k) / np.sum(PI**2 * (u_k - w_k)**2 + \
            PI**2 * (v_k - z_k)**2)

        mu[Gs, :, :] += alpha_k * np.swapaxes(PI * (w_k - u_k), 2, 1)
        nu[Gs, :, :] += alpha_k * np.swapaxes(PI * (z_k - v_k), 2, 1)