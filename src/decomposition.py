# -*- coding: utf-8 -*-
# decomposition.py: Problem decomposition using Lagrangian Duality
# authors: Antoine Passemiers, Cedric Simar

from instance import SUPInstance
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
    u, v, p, theta, w, z, e = init_variables(
        Gs, Gf, n_scenarios, T, N, L, n_import_groups, var_type="Integer")


    P1s = list() # P1s subproblems
    for s in range(n_scenarios):
        problem = SUCLpProblem("P1_%i" % (s+1), pulp.LpMinimize)
        P1s.append(problem)

        print("Defining sub-problem P1_%i..." % (s+1))

        # Define objective function for each s:
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
            problem += (np.sum(e[LIn_ids, s, :], axis=0) + np.sum(p[Gn[n], s, :], axis=0) == \
                D[n, s, :] + np.sum(e[LOn_ids, s, :], axis=0))
        
        # Define constraints group 3.22
        #    e[l, s, t] == B[l, s] * (theta[n, s, t] - theta[m, s, t])
        problem.set_constraint_group("3.22")
        for l in range(L):
            n, m = L_node_indices[l][0], L_node_indices[l][1]
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
        # Generator contingencies: Maximum generator capicity limits
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
        for g in range(len(Gf)):
            UTg = int(UT[Gf[g]])
            for t in range(UTg, T):
                problem += (np.sum(v[Gf[g], s, t-UTg+1:t+1]) <= u[Gf[g], s, t])

        # Define constraints group 3.32
        #    sum_{t+1}^{t+DT[g]} v[g, s, q] <= 1 - u[g, s, t]
        #    t <= N - DT[g]
        problem.set_constraint_group("3.32")
        for g in range(len(Gf)):
            DTg = int(DT[Gf[g]])
            for t in range(0, N-DTg-1):
                problem += (np.sum(v[Gf[g], s, t+1:t+DTg+1]) <= 1 - u[Gf[g], s, t])

        # Define constraints group 3.34
        #    v[g, s, t] <= 1 for slow generators
        problem.set_constraint_group("3.34")
        problem += (v[Gs, s, :] <= 1)

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

        # Define constraints group 3.40
        #    For slow generators:
        #    z[g, t] >= 0
        #    w[g, t] in {0, 1}
        #    Those constraints have been added during variables initialization

    
    print("Defining sub-problem P2...")

    P2 = problem = SUCLpProblem("P2", pulp.LpMinimize)

    # Define objective function for each s:
    #    - sum_Gs sum_s sum_t PI[s] * (mu[g, s, t] * w[g, t] + nu[g, s, t] * z[g, t])
    problem += -np.sum(PI * np.transpose(
        np.transpose(mu[Gs, :, :], (1, 0, 2)) * w + \
        np.transpose(nu[Gs, :, :], (1, 0, 2)) * z, (1, 2, 0)))


    # Define constraints group 3.29
    #    sum_{t-UT[g]+1}^t z[g, q] <= w[g, t]
    #    t >= UT[g]
    problem.set_constraint_group("3.29")
    for g in range(len(Gs)):
        UTg = int(UT[Gs[g]])
        for t in range(UTg, T):
            problem += (np.sum(z[g, t-UTg+1:t+1]) <= w[g, t])

    # Define constraints group 3.30
    #    sum_{t+1}^{t+DT[g]} z[g, q] <= 1 - w[g, t]
    #    t <= N - DT[g]
    problem.set_constraint_group("3.30")
    for g in range(len(Gs)):
        DTg = int(DT[Gs[g]])
        for t in range(0, N-DTg-1):
            problem += (np.sum(z[g, t+1:t+DTg+1]) <= 1 + w[g, t])

    # Define constraints group 3.33
    #    z[g, t] <= 1 for slow generators
    problem.set_constraint_group("3.33")
    problem += (z <= 1)

    # Define constraints group 3.35
    #    s[g, t] >= w[g, t] - w[g, t-1]
    problem.set_constraint_group("3.35")
    problem += z[:, 1:] >= w[:, 1:] - w[:, :-1]

    return P1s, P2, u, v, w, z