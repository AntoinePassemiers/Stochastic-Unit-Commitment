# -*- coding: utf-8 -*-
# lp_relaxation.py: Linear Programmaing relaxation for SUC
# authors: Antoine Passemiers, Cedric Simar

from instance import SUPInstance
from utils import SUCLpProblem
from variables import init_variables

import numpy as np
import pulp


def create_formulation(instance, lower_bound=None, relax=True):
    (G, n_scenarios, T, L, N, n_import_groups) = instance.get_sizes()
    n_generators, n_periods, n_lines, n_nodes = G, T, L, N

    (Gs, Gf, Gn, LIn, LOn, IG, LI_indices, LO_indices, \
        L_node_indices) = instance.get_indices()
    

    (PI, K, S, C, D, P_plus, P_minus, R_plus, R_minus, \
        UT, DT, T_req, F_req, B, TC, FR, IC, GAMMA) = instance.get_constants()


    (u, v, p, theta, w, z, e) = variables = init_variables(
        Gs, Gf, n_scenarios, T, N, L, n_import_groups, relax=relax)

    print("Defining problem...")
    problem = SUCLpProblem("SUC", pulp.LpMinimize)

    # Define objective function: 
    #    sum_g sum_s sum_t PI[s] * (K[g]*u[g, s, t] + S[g]*v[g, s, t] + C[g]*p[g, s, t])
    obj = np.sum(PI * np.swapaxes(
            K * np.swapaxes(u, 0, 2) + \
            S * np.swapaxes(v, 0, 2) + \
            C * np.swapaxes(p, 0, 2), 1, 2))
    problem += obj

    if lower_bound:
        problem.set_constraint_group("obj")
        problem += (obj >= lower_bound)


    # Define constraints group 3.21
    #    Market-clearing constraint: uncertainty in demand 
    #    and production of renewable resources for each node
    #    sum_LIn e[l, s, t] + sum_g p[g, s, t] == D[n, s, t] + sum_LOn e[l, s, t]
    problem.set_constraint_group("3.21")
    for n in range(N):
        LIn_ids = LI_indices[n][LI_indices[n] != SUPInstance.NO_LINE]
        LOn_ids = LO_indices[n][LO_indices[n] != SUPInstance.NO_LINE]
        sum_g = np.sum(p[Gn[n], :, :], axis=0) if len(Gn[n]) > 1 else p[Gn[n][0], :, :]
        problem += (np.sum(e[LIn_ids, :, :], axis=0) + sum_g == \
            D[n, :, :] + np.sum(e[LOn_ids, :, :], axis=0))
    
    # Define constraints group 3.22
    #    e[l, s, t] == B[l, s] * (theta[n, s, t] - theta[m, s, t])
    problem.set_constraint_group("3.22")
    for l in range(L):
        m, n = L_node_indices[l]
        problem += (e[l, :, :] == B[l, :][..., np.newaxis] * \
            (theta[n, :, :] - theta[m, :, :]))

    # Define constraints group 3.23
    #    e[l, s, t] <= TC[l]
    problem.set_constraint_group("3.23")
    problem += (np.swapaxes(e, 0, 2) <= TC)

    # Define constraints group 3.24
    #    -TC[l] <= e[l, s, t]
    problem.set_constraint_group("3.24")
    problem += (-TC <= np.swapaxes(e, 0, 2))

    # Define constraints group 3.25
    # Generator contingencies: Maximum generator capicity limits
    #    p[g, s, t] <= P_plus[g, s] * u[g, s, t]
    problem.set_constraint_group("3.25")
    problem += (np.transpose(p, (2, 0, 1)) <= P_plus * np.transpose(u, (2, 0, 1)))

    # Define constraints group 3.26
    # Generator contingencies: Minimum generator capacity limits
    #    P_minus[g, s]* u[g, s, t] <= p[g, s, t]
    problem.set_constraint_group("3.26")
    problem += (P_minus * np.transpose(u, (2, 0, 1)) <= np.transpose(p, (2, 0, 1)))

    # Define constraints group 3.27
    #    p[g, s, t] - p[g, s, t-1] <= R_plus[g]
    problem.set_constraint_group("3.27")
    problem += (np.swapaxes(p[:, :, 1:] - p[:, :, :-1], 0, 2) <= R_plus)

    # Define constraints group 3.28
    #    p[g, s, t-1] - p[g, s, t] <= R_minus[g]
    problem.set_constraint_group("3.28")
    problem += (np.swapaxes(p[:, :, :-1] - p[:, :, 1:], 0, 2) <= R_minus)

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
        for t in range(0, N-DTg-1):
            problem += (np.sum(z[g, t+1:t+DTg+1]) <= 1 + w[g, t])
    
    # Define contraints group 3.31
    #    sum_{t-UT[g]+1}^t v[g, s, q] <= u[g, s, t]
    #    t >= UT[g]
    problem.set_constraint_group("3.31")
    for g in Gf:
        UTg = int(UT[g])
        for t in range(UTg, T):
            problem += (np.sum(v[g, :, t-UTg+1:t+1], axis=1) <= u[g, :, t])

    # Define constraints group 3.32
    #    sum_{t+1}^{t+DT[g]} v[g, s, q] <= 1 - u[g, s, t]
    #    t <= N - DT[g]
    problem.set_constraint_group("3.32")
    for g in Gf:
        DTg = int(DT[g])
        for t in range(0, N-DTg-1):
            problem += (np.sum(v[g, :, t+1:t+DTg+1], axis=1) <= 1 - u[g, :, t])
    
    # Define constraints group 3.33
    #    z[g, t] <= 1 for slow generators
    problem.set_constraint_group("3.33")
    problem += (z[Gs, :] <= 1)

    # Define constraints group 3.34
    #    v[g, s, t] <= 1 for slow generators
    problem.set_constraint_group("3.34")
    problem += (v[Gs, :, :] <= 1)

    # Define constraints group 3.35
    #    z[g, t] >= w[g, t] - w[g, t-1] for slow generators
    problem.set_constraint_group("3.35")
    problem += z[Gs, 1:] >= w[Gs, 1:] - w[Gs, :-1]

    # Define constraints group 3.36
    #    v[g, s, t] >= u[g, s, t] - u[g, s, t-1]
    problem.set_constraint_group("3.36")
    problem += v[Gf, :, 1:] >= u[Gf, :, 1:] - u[Gf, :, :-1]

    # Define constraints group 3.37
    #    PI[s] * u[g, s, t] == PI[s] * w[g, t]
    problem.set_constraint_group("3.37")
    problem += (np.swapaxes(u[Gs, :, :], 0, 1) == w[Gs, :])

    # Define constraints group 3.38
    #    PI[s] * v[g, s, t] == PI[s] * z[g, t]
    problem.set_constraint_group("3.38")
    problem += (np.swapaxes(v[Gs, :, :], 0, 1) == z[Gs, :])

    # Define constraints group 3.39
    #    For all generators:
    #    p[g, s, t] >= 0
    #    v[g, s, t] >= 0
    #    0 <= u[g, s, t] <= 1
    #    Those constraints have been added during variables initialization

    # Define constraints group 3.40
    #    For slow generators:
    #    z[g, t] >= 0
    #    0 <= w[g, t] <= 1
    #    Those constraints have been added during variables initialization

    return problem, variables