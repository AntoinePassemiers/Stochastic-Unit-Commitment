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
    G = Gs + Gf

    # u[g, s, t] = Commitment of generator g in scenario s, period t
    u = lp_array("U", (G, S, T), "Integer", 0, 1)

    # v[g, s, t] = Startup of generator g in scenario s, period t
    v = lp_array("V", (G, n_scenarios, T), var_type, up_bound=1)

    # p[g, s, t] = Production of generator g in scenario s, period t
    p = lp_array("P", (G, n_scenarios, T), var_type, low_bound=0)

    # theta[n, s, t] = Phase angle at bus n in scenario s, period t
    theta = lp_array("THETA", (N, n_scenarios, T), var_type)

    # w[g, t] = Commitment of slow generator g in period t
    w = lp_array("W", (G, T), "Integer", 0, 1)

    # z[g, t] = Startup of slow generator g in period t
    z = lp_array("Z", (Gs, T), var_type, 0, 1)

    # s[g, t] = Slow reserve provided by generator
    # TODO

    # f[g, t] = Fast reserve provided by generator g in period t
    # TODO

    # e[l, s, t] = Power flow on line l in scenario s, period t
    e = lp_array("E", (L, n_scenarios, T), var_type)

    return u, v, p, theta, w, z, e



if __name__ == "__main__":

    Gs = n_slow_generators = 4
    Gf = n_fast_generators = 6
    G = n_generators = Gs + Gf
    n_scenarios = 2
    T = n_periods = 15
    N = n_buses = 5
    L = n_lines = 3
    I = n_import_groups = 4
    
    instance = SUPInstance(n_generators, n_scenarios, n_periods, n_buses, n_lines, n_import_groups)
    (PI, K, S, C, D, P_plus, P_minus, R_plus, R_minus, \
        UT, DT, T_req, F_req, B, TC, FR, IC, GAMMA) = instance.get_constants()

    problem = ArrayCompatibleLpProblem("SUC", pulp.LpMinimize)

    u, v, p, theta, w, z, e = init_variables(Gs, Gf, n_scenarios, T, N, L, I)

    # Define objective function: 
    #    sum_g sum_s sum_t PI[s] * (K[g]*u[g, s, t] + S[g]*v[g, s, t] + C[g]*p[g, s, t])
    problem += np.sum(PI * np.swapaxes(
        K * np.swapaxes(u, 0, 2) + \
        S * np.swapaxes(v, 0, 2) + \
        C * np.swapaxes(p, 0, 2), 1, 2))


    # TODO: constraints groups 3.21 to 3.22

    # Define constraints group 3.23
    #    e[l, s, t] <= TC[l]
    problem += (np.swapaxes(e, 0, 2) <= TC)

    # Define constraints group 3.24
    #    -TC[l] <= e[l, s, t]
    problem += (-TC <= np.swapaxes(e, 0, 2))

    # Define constraints group 2.25
    #    p[g, s, t] <= P_plus[g, s] * u[g, s, t]
    problem += (np.transpose(p, (2, 0, 1)) <= P_plus * np.transpose(u, (2, 0, 1)))

    # Define constraints group 2.26
    #    P_minus[g, s]* u[g, s, t] <= p[g, s, t]
    problem += (P_minus * np.transpose(u, (2, 0, 1)) <= np.transpose(p, (2, 0, 1)))

    # Define constraints group 2.27
    #    p[g, s, t] - p[g, s, t-1] <= R_plus[g]
    problem += (np.swapaxes(p[:, :, 1:] - p[:, :, :-1], 0, 2) <= R_plus)

    # Define constraints group 2.28
    #    p[g, s, t-1] - p[g, s, t] <= R_minus[g]
    problem += (np.swapaxes(p[:, :, :-1] - p[:, :, 1:], 0, 2) <= R_minus)

    # TODO: constraints groups 2.29 to 3.38

    problem.solve()
    print("Problem status: %i" % problem.status)
    print("Value of the objective: %f" % problem.objective.value())

    with open("solution.txt", "w") as f:
        f.write("Problem status: %i\n" % problem.status)
        f.write("Value of the objective: %f\n" % problem.objective.value())
        for variable in problem.variables():
            f.write("%s = %s\n" % (str(variable.name), str(variable.varValue)))

    solution = problem.get_variables().get_var_values()
    constraints = problem.get_constraints_as_tuples()
