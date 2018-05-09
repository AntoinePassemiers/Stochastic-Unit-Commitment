# -*- coding: utf-8 -*-
# main.py
# authors: Antoine Passemiers, Cedric Simar

from decomposition import decompose_problem
from instance import SUPInstance
from lp_relaxation import init_variables, create_formulation
from utils import LpVarArray, SUCLpProblem

import os
import sys
import time
import argparse
import random
import pulp
import numpy as np

try:
    import pyximport
    pyximport.install(setup_args={'include_dirs': np.get_include()})
    import heuristics
except ImportError:
    print("You definitely should install Cython.")
    sys.exit(0)


def dive_and_fix(problem, u, w):
    variables = problem.get_variables()
    int_mask = [var.name[0] in ["U", "W"] for var in variables]

    for t in range(u.shape[2]):
        while True:
            variables = problem.get_variables()
            solution = variables.get_var_values()
            print(problem.status, np.sum(np.abs(solution[int_mask] - \
                np.round(solution[int_mask])) >= 1e-04), problem.objective.value())
            if problem.status == -1:
                break

            """
            indices = np.where(np.logical_and(
                int_mask,
                np.abs(solution - np.round(solution)) >= 1e-04))[0]
            if len(indices) == 0:
                break

            less_fractional = np.argmin(
                np.abs(np.round(solution[indices]) - solution[indices]))
            h = indices[less_fractional]
            value = 0 if variables[h].varValue < 0.5 else 1
            """

            variables = LpVarArray(list(u[:, :, t].flatten()) + list(w[:, t]), info={"var_type" : "Mixed"})
            values = variables.get_var_values()
            indices = np.where(np.abs(values - np.round(values)) >= 1e-04)[0]
            if len(indices) == 0:
                break
            less_fractional = np.argmin(np.abs(np.round(values[indices]) - values[indices]))
            h = indices[less_fractional]
            value = 0 if variables[h].varValue < 0.5 else 1
            print(variables[h].varValue)
            variables[h].varValue = value
            variables[h].lowBound = value
            variables[h].upBound = value
            problem.solve()


def create_admissible_solution(problem, variables, instance):
    (Gs, Gf, Gn, LIn, LOn, IG, LI_indices, LO_indices, \
        L_node_indices) = instance.get_indices()
    (n_generators, n_scenarios, n_periods, n_lines, \
        n_nodes, n_import_groups) = instance.get_sizes()
    (PI, K, S, C, D, P_plus, P_minus, R_plus, R_minus, \
        UT, DT, T_req, F_req, B, TC, FR, IC, GAMMA) = instance.get_constants()
    (u, v, p, theta, w, z, e) = variables
    variables = problem.get_variables()
    int_mask = [var.name[0] in ["U", "W"] for var in variables]
    solution = variables.get_var_values()
    solution[int_mask] = np.round(solution[int_mask])
    problem.set_var_values(solution)

    n_violated, groups_n_violated = problem.constraints_violated()
    print(n_violated)

    for g in range(n_generators):
        for s in range(n_scenarios):
            for t in range(n_periods):
                ll = P_plus[g, s] * u[g, s, t].varValue if random.random() < 0.5 else P_minus[g, s] * u[g, s, t].varValue
                if p[g, s, t].varValue > P_plus[g, s] * u[g, s, t].varValue:
                    p[g, s, t].varValue = P_plus[g, s] * u[g, s, t].varValue
                elif p[g, s, t].varValue < P_minus[g, s] * u[g, s, t].varValue:
                    p[g, s, t].varValue = P_minus[g, s] * u[g, s, t].varValue


    subprob = SUCLpProblem("-", pulp.LpMinimize)
    obj = C * np.swapaxes(p, 0, 2)
    subprob += obj

    subprob.set_constraint_group("3.21")
    for n in range(n_nodes):
        for s in range(n_scenarios):
            for t in range(n_periods):
                p_values = p[Gn[n], s, t].get_var_values() if isinstance(p[Gn[n], s, t], LpVarArray) else p[Gn[n], s, t].varValue
                subprob += (np.sum(p[Gn[n], s, t], axis=0) == np.sum(p_values))
            
    for g in range(n_generators):
        subprob += (np.transpose(p, (2, 0, 1)) <= P_plus * np.transpose(u.get_var_values(), (2, 0, 1)))
        subprob += (P_minus * np.transpose(u.get_var_values(), (2, 0, 1)) <= np.transpose(p, (2, 0, 1)))

    subprob.solve()
    print(subprob.status)

    n_violated, groups_n_violated = subprob.constraints_violated()
    print(n_violated)


def solve_lp_relaxation(instance):
    # Formulate the LP relaxation for the given instance
    problem, variables = create_formulation(instance, relax=True)

    # Solve LP relaxation
    print("Solving problem...")
    problem.writeLP("prob.lp")
    start = time.time()
    problem.solve()
    exec_time = time.time() - start
    obj = problem.objective.value()
    print("Solve time: %f s" % exec_time)
    print("Problem status: %i" % problem.status)
    if problem.status == pulp.constants.LpStatusOptimal:
        print("Solution is optimal.")
        print("Value of the objective: %f" % obj)
    elif problem.status == pulp.constants.LpStatusNotSolved:
        print("Problem not solved.")
    elif problem.status == pulp.constants.LpStatusInfeasible:
        print("Problem is infeasible.")
    elif problem.status == pulp.constants.LpStatusUnbounded:
        print("Problem is unbounded.")
    else:
        print("Problem is undefined.")


    print(problem.is_integer_solution())
    if args.round == "dive-and-fix": # TODO
        #cp = heuristics.CyProblem(constraints)
        #rounded = cp.round(solution, int_mask)
        #problem.set_var_values(rounded)
        
        #problem, u, w = create_formulation(instance, relax=False, lower_bound=1.1*obj)
        #problem.solve()
        create_admissible_solution(problem, variables, instance)

    n_violated, groups_n_violated = problem.constraints_violated()
    print("Number of violated constraints (python): %i" % n_violated)
    for group in groups_n_violated.keys():
        if groups_n_violated[group][0] > 0:
            print("Number of violated constraints of group %s: %i / %i" % (
                group, groups_n_violated[group][0], groups_n_violated[group][1]))
    if problem.is_integer_solution() and n_violated == 0:
        print("Found integer MIP solution.")
        print("Value of the objective: %f" % problem.objective.value())


    with open("solution.txt", "w") as f:
        f.write("Problem status: %i\n" % problem.status)
        f.write("Value of the objective: %f\n" % problem.objective.value())
        for variable in problem.variables():
            f.write("%s = %s\n" % (str(variable.name), str(variable.varValue)))


def solve_subproblems(instance):
    PI = instance.PI
    Gs = instance.Gs
    G = instance.n_generators
    S = instance.n_scenarios
    T = instance.n_periods

    # Lagrangian multipliers
    mu = np.zeros((G, S, T))
    nu = np.zeros((G, S, T))

    for i in range(10):
        P1, P2, u, v, w, z = decompose_problem(instance, mu, nu)
        start = time.time()
        print(P1[0].solve())
        print(P2.solve())
        exec_time = time.time() - start
        print("Solve time: %f s" % exec_time)




if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("instance_file_path")
    parser.add_argument(
        "--round",
        choices=['dive-and-fix'],
        help="Round solution using heuristic algorithms")
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="Decompose problem using lagrangian duality")
    args = parser.parse_args()

    # Parse instance file
    instance = SUPInstance.from_file(args.instance_file_path)

    if args.decompose:
        solve_subproblems(instance)
    else:
        solve_lp_relaxation(instance)
