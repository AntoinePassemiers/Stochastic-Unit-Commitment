# -*- coding: utf-8 -*-
# decomposition.py: Problem decomposition using Lagrangian Duality
# authors: Antoine Passemiers, Cedric Simar

from instance import SUPInstance
from lp_relaxation import create_formulation
from utils import SUCLpProblem
from variables import init_variables

import numpy as np
import pulp


def decompose_problem(instance, mu, nu):
    (G, n_scenarios, T, L, N, n_import_groups) = instance.get_sizes()
    n_generators, n_periods, n_lines, n_nodes = G, T, L, N

    (Gs, Gf, Gn, LIn, LOn, IG, LI_indices, LO_indices, \
        L_node_indices) = instance.get_indices()

    (PI, K, S, C, D, P_plus, P_minus, R_plus, R_minus, \
        UT, DT, T_req, F_req, B, TC, FR, IC, GAMMA) = instance.get_constants()
        
    (u, v, p, theta, w, z, e) = variables = init_variables(
        Gs, Gf, n_scenarios, T, N, L, n_import_groups, relax=False)

    # Original problem
    PP, _ = create_formulation(instance, variables=variables, relax=False)

    P1 = list() # P1 subproblems
    for s in range(n_scenarios):
        problem = SUCLpProblem("P1_%i" % (s+1), pulp.LpMinimize)
        P1.append(problem)

        # Define objective function for scenario s:
        #    sum_g sum_t PI[s] * (K[g]*u[g, s, t] + S[g]*v[g, s, t] + C[g]*p[g, s, t])
        #       + sum_gs sum_t PI[s] * (mu[g, s, t] * u[g, s, t] + nu[g, s, t] * v[g, s, t])
        problem += np.sum(PI[s] * (K * u[:, s, :].T + S * v[:, s, :].T + C * p[:, s, :].T)) + \
            np.sum(PI[s] * (mu[Gs, s, :] * u[Gs, s, :] + nu[Gs, s, :] * v[Gs, s, :]))

        # Define constraints group 3.21
        #    Market-clearing constraint: uncertainty in demand 
        #    and production of renewable resources for each node
        #    sum_LIn e[l, s, t] + sum_g p[g, s, t] == D[n, s, t] + sum_LOn e[l, s, t]
        problem.set_constraint_group("3.21")
        for n in range(N):
            LIn_ids = LI_indices[n][LI_indices[n] != -1]
            LOn_ids = LO_indices[n][LO_indices[n] != -1]
            sum_g = np.sum(p[Gn[n], s, :], axis=0) if len(Gn[n]) > 1 else p[Gn[n][0], s, :]
            problem += (np.sum(e[LIn_ids, s, :], axis=0) + sum_g == \
                D[n, s, :] + np.sum(e[LOn_ids, s, :], axis=0))
        
        # Define constraints group 3.22
        #    e[l, s, t] == B[l, s] * (theta[n, s, t] - theta[m, s, t])
        problem.set_constraint_group("3.22")
        for l in range(L):
            m, n = L_node_indices[l]
            problem += (e[l, s, :] == B[l, s] * (theta[n, s, :] - theta[m, s, :]))

        # Define constraints group 3.23
        #    e[l, s, t] <= TC[l]
        problem.set_constraint_group("3.23")
        problem += (e[:, s, :].T <= TC)

        # Define constraints group 3.24
        #    -TC[l] <= e[l, s, t]
        problem.set_constraint_group("3.24")
        problem += (-TC <= e[:, s, :].T)

        # Define constraints group 3.25
        # Generator contingencies: Maximum generator capacity limits
        #    p[g, s, t] <= P_plus[g, s] * u[g, s, t]
        problem.set_constraint_group("3.25")
        problem += (p[:, s, :].T <= P_plus[:, s] * u[:, s, :].T)

        # Define constraints group 3.26
        # Generator contingencies: Minimum generator capicity limits
        #    P_minus[g, s]* u[g, s, t] <= p[g, s, t]
        problem.set_constraint_group("3.26")
        problem += (P_minus[:, s] * u[:, s, :].T <= p[:, s, :].T)

        # Define constraints group 3.27
        #    p[g, s, t] - p[g, s, t-1] <= R_plus[g]
        problem.set_constraint_group("3.27")
        problem += ((p[:, s, 1:] - p[:, s, :-1]).T <= R_plus)

        # Define constraints group 3.28
        #    p[g, s, t-1] - p[g, s, t] <= R_minus[g]
        problem.set_constraint_group("3.28")
        problem += ((p[:, s, :-1] - p[:, s, 1:]).T <= R_minus)
        
        # Define contraints group 3.31
        #    sum_{t-UT[g]+1}^t v[g, s, q] <= u[g, s, t]
        #    t >= UT[g]
        problem.set_constraint_group("3.31")
        for g in Gf:
            UTg = int(UT[g])
            for t in range(UTg, T):
                problem += (np.sum(v[g, s, t-UTg+1:t+1]) <= u[g, s, t])

        # Define constraints group 3.32
        #    sum_{t+1}^{t+DT[g]} v[g, s, q] <= 1 - u[g, s, t]
        #    t <= N - DT[g]
        problem.set_constraint_group("3.32")
        for g in Gf:
            DTg = int(DT[g])
            # Number of periods in horizon = T
            for t in range(0, T-DTg-1):
                if t + 1 < T:
                    problem += (np.sum(v[g, s, t+1:t+DTg+1]) <= 1 - u[g, s, t])

        # Define constraints group 3.34
        #    v[g, s, t] <= 1 for slow generators
        #    Those constraints have been added during variables initialization

        # Define constraints group 3.36
        #    v[g, s, t] >= u[g, s, t] - u[g, s, t-1] for fast generators
        problem.set_constraint_group("3.36")
        problem += v[Gf, s, 1:] >= u[Gf, s, 1:] - u[Gf, s, :-1]

        # Define constraints group 3.39
        #    For all generators:
        #    p[g, s, t] >= 0
        #    v[g, s, t] >= 0
        #    u[g, s, t] in {0, 1}
        #    Those constraints have been added during variables initialization


    P2 = problem = SUCLpProblem("P2", pulp.LpMaximize)

    # Define objective function for each s:
    #    - sum_Gs sum_s sum_t PI[s] * (mu[g, s, t] * w[g, t] + nu[g, s, t] * z[g, t])
    problem += np.sum(PI * np.transpose(
        np.transpose(mu[Gs, :, :], (1, 0, 2)) * w[Gs, :] + \
        np.transpose(nu[Gs, :, :], (1, 0, 2)) * z[Gs, :], (1, 2, 0)))

    # Define constraints group 3.29
    #    sum_{t-UT[g]+1}^t z[g, q] <= w[g, t]
    #    t >= UT[g]
    problem.set_constraint_group("3.29")
    for g in Gs:
        UTg = int(UT[g])
        for t in range(UTg, T):
            problem += (np.sum(z[g, t-UTg+1:t+1]) <= w[g, t])

    # Define constraints group 3.30
    #    sum_{t+1}^{t+DT[g]} z[g, q] <= 1 - w[g, t]
    #    t <= N - DT[g]
    problem.set_constraint_group("3.30")
    for g in Gs:
        DTg = int(DT[g])
        for t in range(0, T-DTg+1):
            if t + 1 < T:
                problem += (np.sum(z[g, t+1:t+DTg+1]) <= 1 - w[g, t])

    # Define constraints group 3.33
    #    z[g, t] <= 1 for slow generators
    #    Those constraints have been added during variables initialization

    # Define constraints group 3.35
    #    z[g, t] >= w[g, t] - w[g, t-1] for slow generators
    problem.set_constraint_group("3.35")
    problem += z[Gs, 1:] >= w[Gs, 1:] - w[Gs, :-1]

    # Define constraints group 3.40
    #    For slow generators:
    #    z[g, t] >= 0
    #    w[g, t] in {0, 1}
    #    Those constraints have been added during variables initialization


    ED = list() # Economic dispatches
    for s in range(n_scenarios):
        problem = SUCLpProblem("ED_%i" % (s+1), pulp.LpMinimize)
        ED.append(problem)

        # Define objective function for scenario s:
        #    sum_g sum_t K[g]*w[g, t] + S[g]*z[g, t] + C[g]*p[g, s, t]
        problem += np.sum(K * w[:, :].T + S * z[:, :].T + C * p[:, s, :].T)

        # Define constraints group 3.42
        #    Market-clearing constraint: uncertainty in demand 
        #    and production of renewable resources for each node
        #    sum_LIn e[l, s, t] + sum_g p[g, s, t] == D[n, s, t] + sum_LOn e[l, s, t]
        problem.set_constraint_group("3.42")
        for n in range(N):
            LIn_ids = LI_indices[n][LI_indices[n] != -1]
            LOn_ids = LO_indices[n][LO_indices[n] != -1]
            sum_g = np.sum(p[Gn[n], s, :], axis=0) if len(Gn[n]) > 1 else p[Gn[n][0], s, :]
            problem += (np.sum(e[LIn_ids, s, :], axis=0) + sum_g == \
                D[n, s, :] + np.sum(e[LOn_ids, s, :], axis=0))

        # Define constraints group 3.43
        #    e[l, s, t] == B[l, s] * (theta[n, s, t] - theta[m, s, t])
        problem.set_constraint_group("3.43")
        for l in range(L):
            m, n = L_node_indices[l]
            problem += (e[l, s, :] == B[l, s] * (theta[n, s, :] - theta[m, s, :]))

        # Define constraints group 3.44
        # Generator contingencies: Maximum generator capacity limits
        #    p[g, s, t] <= P_plus[g, s] * u[g, s, t]
        problem.set_constraint_group("3.44")
        problem += (p[:, s, :].T <= P_plus[:, s] * u[:, s, :].T)

        # Define constraints group 3.45
        # Generator contingencies: Minimum generator capicity limits
        #    P_minus[g, s]* u[g, s, t] <= p[g, s, t]
        problem.set_constraint_group("3.45")
        problem += (P_minus[:, s] * u[:, s, :].T <= p[:, s, :].T)

        # Define constraints group 3.46
        #    p[g, s, t] - p[g, s, t-1] <= R_plus[g]
        problem.set_constraint_group("3.46")
        problem += ((p[:, s, 1:] - p[:, s, :-1]).T <= R_plus)

        # Define constraints group 3.28
        #    p[g, s, t-1] - p[g, s, t] <= R_minus[g]
        problem.set_constraint_group("3.28")
        problem += ((p[:, s, :-1] - p[:, s, 1:]).T <= R_minus)

        # Define constraints group 3.23
        #    e[l, s, t] <= TC[l]
        problem.set_constraint_group("3.23")
        problem += (e[:, s, :].T <= TC)

        # Define constraints group 3.24
        #    -TC[l] <= e[l, s, t]
        problem.set_constraint_group("3.24")
        problem += (-TC <= e[:, s, :].T)

        # Define constraints group 3.49
        #    For slow generators:
        #    p[g, s, t] >= 0
        #    z[g, t] >= 0
        #    w[g, t] in {0, 1}
        #    Those constraints have been added during variables initialization


    return PP, P1, P2, ED, variables