# -*- coding: utf-8 -*-
# main.py
# authors: Antoine Passemiers, Cedric Simar

from decomposition import decompose_problem
from instance import SUPInstance
from lp_relaxation import init_variables, create_lp_relaxation

import sys
import time
import argparse
import pulp
import numpy as np

try:
    import pyximport
    pyximport.install(setup_args={'include_dirs': np.get_include()})
    import heuristics
except ImportError:
    print("You definitely should install Cython.")
    sys.exit(0)


def solve_lp_relaxation(instance):
    # Formulate the LP relaxation for the given instance
    problem = create_lp_relaxation(instance)

    # Solve LP relaxation
    print("Solving problem...")
    problem.writeLP("prob.lp")
    start = time.time()
    problem.solve()
    exec_time = time.time() - start
    print("Solve time: %f s" % exec_time)
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

    variables = problem.get_variables()
    solution = variables.get_var_values()
    var_names = [var.name for var in variables]
    int_mask = [var.name[0] in ["U", "W"] for var in variables]
    constraints = problem.get_constraints_as_tuples()

    print(problem.is_integer_solution())
    if args.round == "dive-and-fix":
        print("\nApplying Dive-And-Fix...")

        fixed = list()
        step = 10

        for k in range(100):
            variables = problem.get_variables()
            solution = variables.get_var_values()

            print(problem.status, np.sum(np.abs(solution[int_mask] - np.round(solution[int_mask])) >= 1e-04), problem.objective.value())

            for i in range(step):
                indices = np.where(np.logical_and(int_mask, np.abs(solution - np.round(solution)) >= 1e-04))[0]
                less_fractional = np.argmin(np.abs(np.round(solution[indices]) - solution[indices]))
                h = indices[less_fractional]
                value = 0 if variables[h].varValue < 0.5 else 1
                fixed.append((h, value))
                solution[h] = value

            for h, value in fixed:
                variables[h].lowBound = value
                variables[h].upBound = value
            problem.solve()

        n_violated, groups_n_violated = problem.constraints_violated()
        print("Number of violated constraints (python): %i" % n_violated)
        for group in groups_n_violated.keys():
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
    G = instance.n_generators
    S = instance.n_scenarios
    T = instance.n_periods

    # Lagrangian multipliers
    mu = np.zeros((G, S, T))
    nu = np.zeros((G, S, T))

    P1, P2 = decompose_problem(instance, mu, nu)

    start = time.time()
    print(P1[0].solve())
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
