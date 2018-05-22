# -*- coding: utf-8 -*-
# subgradient.py: Subgradient algorithm for solving lagrangian relaxation
# authors: Antoine Passemiers, Cedric Simar

from decomposition import decompose_problem

import pulp
import time
import numpy as np

import matplotlib.pyplot as plt


def solve_subproblem(subproblem):
    """ Solve sub-problem of the lagrangian decomposition of SUC.

    Args:
        subproblem (SUCLpProblem): Sub-problem to be solved
    """
    start = time.time()
    # print(subproblem.name)
    status = subproblem.solve()

    if subproblem.status != pulp.constants.LpStatusOptimal:
        n_violated, groups_n_violated = subproblem.constraints_violated()
        print("Number of violated constraints: %i" % n_violated)
        print(subproblem.is_integer_solution(), n_violated)
        for group in groups_n_violated.keys():
            if groups_n_violated[group][0] > 0:
                print("Number of violated constraints of group %s: %i / %i" % (
                    group, groups_n_violated[group][0], groups_n_violated[group][1]))
        print("Problem status: %s" % pulp.LpStatus[subproblem.status])
        assert(subproblem.status == pulp.constants.LpStatusOptimal)
    exec_time = time.time() - start
    #print("\tSub-problem %s - Solve time: %f s (status: %i)" % \
    #    (subproblem.name, exec_time, status))


def solve_with_subgradient(instance, _lambda=2., _epsilon=1.0):
    PI, Gs, K, S, C = instance.get_attributes(["PI", "Gs", "K", "S", "C"])

    # Lagrangian multipliers
    mu = np.zeros((instance.n_generators, instance.n_scenarios, instance.n_periods))
    nu = np.zeros((instance.n_generators, instance.n_scenarios, instance.n_periods))

    aa, ab = list(), list()
    LB, UB = -np.inf, np.inf
    for k in range(100):
        PP, P1, P2, ED, variables = decompose_problem(instance, mu, nu)
        (u, v, p, theta, w, z, e) = variables

        for s in range(instance.n_scenarios):
            # TODO: paralléliser
            solve_subproblem(P1[s])
        solve_subproblem(P2)

        L_k = P2.objective.value() if P2.objective.value() else 0
        for s in range(instance.n_scenarios):
            L_k += P1[s].objective.value()

        n_violated, groups_n_violated = PP.constraints_violated()
        print("Number of violated constraints: %i" % n_violated)
        for group in groups_n_violated.keys():
            if groups_n_violated[group][0] > 0:
                print("Number of violated constraints of group %s: %i / %i" % (
                    group, groups_n_violated[group][0], groups_n_violated[group][1]))

        if L_k == LB:
            _lambda /= 2
        elif L_k > LB:
            LB = L_k

        u_k = np.swapaxes(u[Gs, :, :].get_var_values(), 2, 1)
        v_k = np.swapaxes(v[Gs, :, :].get_var_values(), 2, 1)
        w_k = w[Gs, :].get_var_values()[..., np.newaxis]
        z_k = z[Gs, :].get_var_values()[..., np.newaxis]

        for g in Gs:
            for t in range(instance.n_periods):
                w[g, t].lowBound = w[g, t].upBound = w[g, t].varValue
                z[g, t].lowBound = z[g, t].upBound = z[g, t].varValue

        for s in range(instance.n_scenarios):
            solve_subproblem(ED[s])

        L_hat = PP.objective.value()
        if L_hat < UB:
            UB = L_hat

        if UB - LB <= _epsilon:
            break

        print("UB: %f, LB: %f" % (UB, LB))
        alpha_k = _lambda * (L_hat - L_k) / np.sum((PI**2) * (u_k - w_k)**2 + \
            (PI**2) * (v_k - z_k)**2)
        
        aa.append(LB)
        ab.append(UB)
        
        # print(np.swapaxes(PI * (z_k - v_k), 2, 1))


        mu[Gs, :, :] -= alpha_k * np.swapaxes(PI * (w_k - u_k), 2, 1)
        nu[Gs, :, :] -= alpha_k * np.swapaxes(PI * (z_k - v_k), 2, 1)

    plt.step(np.arange(1, len(aa)+1), aa)
    plt.step(np.arange(1, len(ab)+1), ab)
    plt.show()