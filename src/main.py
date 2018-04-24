# -*- coding: utf-8 -*-
# main.py
# authors: Antoine Passemiers, Cedric Simar

from instance import SUPInstance
from utils import lp_array, ArrayCompatibleLpProblem

import numpy as np
import pulp

try:
    import pyximport
    pyximport.install(setup_args={'include_dirs': np.get_include()})
    import heuristics
except ImportError:
    print("You definitely should install Cython.")


def init_variables(Gs, Gf, S, T, N, L, I, var_type="Continuous"):
    G = len(Gs) + len(Gf) # Number of generators

    # u[g, s, t] = Commitment of generator g in scenario s, period t
    u = lp_array("U", (G, S, T), "Integer", 0, 1)

    # v[g, s, t] = Startup of generator g in scenario s, period t
    v = lp_array("V", (G, n_scenarios, T), var_type, up_bound=1)

    # p[g, s, t] = Production of generator g in scenario s, period t
    p = lp_array("P", (G, n_scenarios, T), var_type, low_bound=0)

    # theta[n, s, t] = Phase angle at bus n in scenario s, period t
    theta = lp_array("THETA", (N, n_scenarios, T), var_type)

    # w[g, t] = Commitment of slow generator g in period t
    w = lp_array("W", (len(Gs), T), "Integer", 0, 1)

    # z[g, t] = Startup of slow generator g in period t
    z = lp_array("Z", (len(Gs), T), var_type, 0, 1)

    # s[g, t] = Slow reserve provided by generator
    # TODO

    # f[g, t] = Fast reserve provided by generator g in period t
    # TODO

    # e[l, s, t] = Power flow on line l in scenario s, period t
    e = lp_array("E", (L, n_scenarios, T), var_type)

    return u, v, p, theta, w, z, e



if __name__ == "__main__":

    instance = SUPInstance.from_file("../instances/inst-20-24-10-0.txt")
    
    (G, n_scenarios, T, L, N, n_import_groups) = instance.get_sizes()
    (Gs, Gf, Gn, LIn, LOn, IG) = instance.get_indices()
    (PI, K, S, C, D, P_plus, P_minus, R_plus, R_minus, \
        UT, DT, T_req, F_req, B, TC, FR, IC, GAMMA) = instance.get_constants()

    problem = ArrayCompatibleLpProblem("SUC", pulp.LpMinimize)

    u, v, p, theta, w, z, e = init_variables(Gs, Gf, n_scenarios, T, N, L, n_import_groups)

    print("Defining problem...")

    # Define objective function: 
    #    sum_g sum_s sum_t PI[s] * (K[g]*u[g, s, t] + S[g]*v[g, s, t] + C[g]*p[g, s, t])
    problem += np.sum(PI * np.swapaxes(
        K * np.swapaxes(u, 0, 2) + \
        S * np.swapaxes(v, 0, 2) + \
        C * np.swapaxes(p, 0, 2), 1, 2))


    # Define constraints group 3.21
    #    Market-clearing constraint: uncertainty in demand 
    #    and production of renewable resources
    #    sum_LIn e[l, s, t] + sum_g p[g, s, t] == D[n, s, t] + sum_LOn e[l, s, t]
    #for n in range(N):
    #    problem += (np.sum(e[LIn, :, :], axis=0) + np.sum(p[Gn[n], :, :], axis=0) == \
    #        D[n, :, :] + np.sum(e[LOn, :, :], axis=0))
    
    # Define constraints group 3.22
    #    e[l, s, t] == B[l, s] * (theta[n, s, t] - theta[m, s, t])
    #for l, (m, n) in enumerate(LIn + LOn):
    #    problem += (e[l, :, :] == B[l, :] * (theta[n, s, t] - theta[m, s, t]))

    # Define constraints group 3.23
    #    e[l, s, t] <= TC[l]
    problem += (np.swapaxes(e, 0, 2) <= TC)

    # Define constraints group 3.24
    #    -TC[l] <= e[l, s, t]
    problem += (-TC <= np.swapaxes(e, 0, 2))

    # Define constraints group 3.25
    # Generator contingencies: Maximum generator capicity limits
    #    p[g, s, t] <= P_plus[g, s] * u[g, s, t]
    problem += (np.transpose(p, (2, 0, 1)) <= P_plus * np.transpose(u, (2, 0, 1)))

    # Define constraints group 3.26
    # Generator contingencies: Minimum generator capicity limits
    #    P_minus[g, s]* u[g, s, t] <= p[g, s, t]
    problem += (P_minus * np.transpose(u, (2, 0, 1)) <= np.transpose(p, (2, 0, 1)))

    # Define constraints group 3.27
    #    p[g, s, t] - p[g, s, t-1] <= R_plus[g]
    problem += (np.swapaxes(p[:, :, 1:] - p[:, :, :-1], 0, 2) <= R_plus)

    # Define constraints group 3.28
    #    p[g, s, t-1] - p[g, s, t] <= R_minus[g]
    problem += (np.swapaxes(p[:, :, :-1] - p[:, :, 1:], 0, 2) <= R_minus)

    # Define constraint group 3.29
    #    sum_{t-UT[g]+1}^t z[g, q] <= w[g, t]
    #    t >= UT[g]
    for g in range(len(Gs)):
        Utg = int(UT[Gs[g]])
        for t in range(Utg, T):
            problem += (np.sum(z[g, t-Utg+1:t+1]) <= w[g, t])

    # TODO: constraints groups 3.30 to 3.32

    # Define constraints group 3.35
    #    s[g, t] >= w[g, t] - w[g, t-1]
    problem += z[:, 1:] >= w[:, 1:] - w[:, :-1]

    # Define constraints group 3.36
    #    v[g, s, t] >= u[g, s, t] - u[g, s, t-1]
    problem += v[Gf, :, 1:] >= u[Gf, :, 1:] - u[Gf, :, :-1]

    # Define constraints group 3.37
    #    PI[s] * u[g, s, t] == PI[s] * w[g, t]
    problem += (np.swapaxes(u, 0, 1)[:, Gs, :] == w)

    # Define constraints group 3.38
    #    PI[s] * v[g, s, t] == PI[s] * z[g, t]
    problem += (np.swapaxes(v, 0, 1)[:, Gs, :] == z)

    print("Solving problem...")
    problem.solve()
    print("Problem status: %i" % problem.status)
    if problem.status == pulp.constants.LpStatusOptimal:
        print("Solution is optimal.")
        print("Value of the objective: %f" % problem.objective.value())
    elif problem.status == pulp.constants.LpStatusNotSolved:
        print("Problem not solved.")
    elif problem.status == pulp.constants.LpStatusInfeasible:
        print("Problem is infeasible.")
    elif problem.status == pulp.constants.LpStatusUnbounded:
        print("Problem is unbounded.")
    else:
        print("Problem is undefined.")

    with open("solution.txt", "w") as f:
        f.write("Problem status: %i\n" % problem.status)
        f.write("Value of the objective: %f\n" % problem.objective.value())
        for variable in problem.variables():
            f.write("%s = %s\n" % (str(variable.name), str(variable.varValue)))

    solution = problem.get_variables().get_var_values()
    constraints = problem.get_constraints_as_tuples()
