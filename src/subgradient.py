# -*- coding: utf-8 -*-
# subgradient.py: Subgradient algorithm for solving lagrangian relaxation
# authors: Antoine Passemiers, Cedric Simar

from decomposition import decompose_problem
from heuristics import evolve_and_fix

import pulp
import time
import numpy as np

import matplotlib.pyplot as plt


def solve_subproblem(subproblem):
    """ Solve sub-problem of the lagrangian decomposition of SUC.

    Args:
        subproblem (SUCLpProblem): Sub-problem to be solved
    """
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


def solve_with_subgradient(instance, _lambda=0.01, _epsilon=0.01, _alpha0=5000.0, _rho=0.96, _nar=25):
    """ Solve sub-problems of the lagrangian decomposition using subgradient method.

    Returns the PuLP instance of the original problem and the last known value
    of the lagrangian dual.

    Args:
        instance (SUPInstance):
            stores constants and indices that are part of the problem instance
        _lambda (float, optional):
            Constant control parameter for the dynamic steplength
        _epsilon (float, optional):
            Convergence threshold of subgradient method
        _alpha0 (float, optional):
            Initial subgradient steplength
        _nar (float, int):
            Number of subgraient iterations without primal recovery
    """
    # Retrieve useful constants from problem instance
    PI, Gs, K, S, C, P_plus = instance.get_attributes(["PI", "Gs", "K", "S", "C", "P_plus"])
    n_periods = instance.n_periods
    n_scenarios = instance.n_scenarios
    n_generators = instance.n_generators

    # Compute the worst upper bound possible
    L_hat = np.sum(PI * np.swapaxes(
        K * np.ones((n_periods, n_scenarios, n_generators)) + \
        S * np.ones((n_periods, n_scenarios, n_generators)) + \
        C * P_plus.T, 1, 2))

    # Initialize Lagrange multipliers to zeroes
    mu = np.zeros((instance.n_generators, instance.n_scenarios, instance.n_periods))
    nu = np.zeros((instance.n_generators, instance.n_scenarios, instance.n_periods))

    # Initialize empty history
    lb_history, ub_history, dual_history = list(), list(), list()
    primal_solutions = list()
    LB, UB = -np.inf, np.inf
    k = 0
    found_integer_solution = False
    while True:
        print("Iteration %03d" % (k + 1))
        # Create instance of the original problem, the subproblems of
        # the lagrangian decomposition, and the economic dispatch
        # associated to each scenario
        PP, P1, P2, ED, variables = decompose_problem(instance, mu, nu)
        (u, v, p, theta, w, z, e) = variables

        # Solve the lagrangian subproblems
        # This part of the algorithm should be distributed in order
        # to benefit from the decomposition
        for s in range(instance.n_scenarios):
            solve_subproblem(P1[s])
        solve_subproblem(P2)

        # Compute the value of the lagrangian dual
        L_k = P2.objective.value() if P2.objective.value() else 0
        for s in range(instance.n_scenarios):
            L_k += P1[s].objective.value()

        # Update steplength control parameter and lower bound
        if L_k == LB:
            _lambda /= 2
        elif L_k > LB:
            LB = L_k

        # Retrieve current values of variables U, V, W and Z
        # from PuLP variables
        u_k = np.swapaxes(u[Gs, :, :].get_var_values(), 2, 1)
        v_k = np.swapaxes(v[Gs, :, :].get_var_values(), 2, 1)
        w_k = w[Gs, :].get_var_values()[..., np.newaxis]
        z_k = z[Gs, :].get_var_values()[..., np.newaxis]
        
        if k > _nar:
            # Compute a new unfeasible solution as the ergodic sum of
            # previous unfeasible primal solutions
            betas = np.flip(1. / np.arange(1, len(primal_solutions)+1), 0)
            betas /= betas.sum()
            convex_comb = (betas * np.asarray(primal_solutions).T).sum(axis=1)
            PP.set_var_values(convex_comb)

            # Apply heuristic in order to recover a primal feasible solution
            evolve_and_fix(PP, verbose=False)
            n_violated, _ = PP.constraints_violated()
            if n_violated == 0:
                L_hat = PP.objective.value()
                found_integer_solution = True

        # Update upper bound if a new primal feasible solution is found
        if L_hat < UB:
            UB = L_hat
        if (UB - LB) / UB <= _epsilon:
            break

        print("\tL_k : {:10.4f} <= LB : {:10.4f} <= L_hat : {:10.4f} <= UB : {:10.4f}".format(L_k, LB, L_hat, UB))

        # Compute the new steplength
        squared_cons = np.sum((PI**2) * (u_k - w_k)**2 + (PI**2) * (v_k - z_k)**2)
        if squared_cons == 0 or L_hat == L_k:
            alpha_k = 0
        else:
            if k > _nar and found_integer_solution:
                alpha_k = _lambda * (L_hat - L_k) / squared_cons
            else:
                alpha_k = _alpha0 * (_rho ** k)
        
        # Update solution history
        lb_history.append(LB)
        ub_history.append(UB)
        dual_history.append(L_k)
        primal_solutions.append(PP.get_var_values())
        
        # Update Lagrange multipliers
        mu[Gs, :, :] -= alpha_k * np.swapaxes(PI * (w_k - u_k), 2, 1)
        nu[Gs, :, :] -= alpha_k * np.swapaxes(PI * (z_k - v_k), 2, 1)

        if n_violated == 0 or alpha_k == 0:
            print("Feasible primal solution found")
            break
        k += 1

    # If a feasible solution has not been found at last iteration,
    # compute a new unfeasible solution as the ergodic sum of
    # previous unfeasible primal solutions and apply heuristic on it
    n_violated, _ = PP.constraints_violated()
    if n_violated > 0:    
        primal_solutions = np.asarray(primal_solutions)
        betas = np.flip(1. / np.arange(1, len(primal_solutions)+1), 0)
        betas /= betas.sum()
        convex_comb = (betas * primal_solutions.T).sum(axis=1)
        PP.set_var_values(convex_comb)
        evolve_and_fix(PP, verbose=False)
    return PP, L_k