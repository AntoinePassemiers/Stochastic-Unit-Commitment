# -*- coding: utf-8 -*-
# main.py
# authors: Antoine Passemiers, Cedric Simar

from decomposition import decompose_problem
from instance import SUPInstance
from lp_relaxation import init_variables, create_formulation
from dive_and_fix import dive_and_fix
from heuristics import evolve_and_fix
from subgradient import solve_with_subgradient

import os
import time
import argparse
import pulp


def solve_problem(instance, relax=True, _round=False, decompose=False, 
        _epsilon=0.01, _alpha0=5000.0, _rho=0.92, _nar=10):
    """ Solve a SUC problem instance given instance constants.

    Returns the value of the objective function, the total execution
    time, the number of violated constraints (if there are any),
    and the value of the lagrangian dual (if lagrangian decomposition
    is used).

    Args:
        instance (SUPInstance):
            Constants of the problem instance
        relax (bool, optional):
            Whether to solve the linear relaxation of the problem
        _round (bool, optional):
            Whether to apply rounding heuristic at the end of the process
        _decompose (bool, optional):
            Whether to solve the lagrangian decomposition of the problem
            using subgradient method
        _lambda (float, optional):
            Constant control parameter for the dynamic steplength
        _epsilon (float, optional):
            Convergence threshold of subgradient method
        _alpha0 (float, optional):
            Initial subgradient steplength
        _nar (float, int):
            Number of subgradient iterations without primal recovery
    """
    print("Solving problem...")
    start = time.time()
    if not decompose:
        # Formulate the LP relaxation for the given instance
        problem, (u, v, p, theta, w, z, e) = create_formulation(instance, relax=relax)
        # Solve LP relaxation
        problem.solve()
        l_k = None
    else:
        problem, l_k = solve_with_subgradient(instance,
            _epsilon=_epsilon, _alpha0=_alpha0, _rho=_rho, _nar=_nar)
        evolve_and_fix(problem)
    exec_time = time.time() - start
    obj = problem.objective.value()
    print("Solve time: %f s" % exec_time)
    if problem.status == pulp.constants.LpStatusOptimal:
        print("Value of the objective: %f" % obj)

    if _round:
        # Use 'evolve-an-fix' heuristic
        heuristic_start = time.time()
        evolve_and_fix(problem)
        print("Rounding time: %f s" % (time.time() - heuristic_start))
        obj = problem.objective.value()
        print("Value of the objective: %f" % obj)

    n_violated, groups_n_violated = problem.constraints_violated()
    print("Number of violated constraints: %i" % n_violated)
    for group in groups_n_violated.keys():
        if groups_n_violated[group][0] > 0:
            print("Number of violated constraints of group %s: %i / %i" % (
                group, groups_n_violated[group][0], groups_n_violated[group][1]))
    if problem.is_integer_solution() and n_violated == 0:
        print("Found feasible solution to the original primal problem.")
        print("Value of the objective: %f" % problem.objective.value())
    total_time = time.time() - start

    with open("solution.txt", "w") as f:
        f.write("Problem status: %i\n" % problem.status)
        f.write("Value of the objective: %f\n" % problem.objective.value())
        for variable in problem.variables():
            f.write("%s = %s\n" % (str(variable.name), str(variable.varValue)))
    return obj, total_time, n_violated, l_k


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    parser.add_argument("instance_file_path")
    group.add_argument(
        "--relax",
        action="store_true",
        help="Solve linear relaxation")
    group.add_argument(
        "--decompose",
        action="store_true",
        help="Decompose problem using lagrangian duality")
    parser.add_argument(
        "--round",
        action="store_true",
        help="Round solution using heuristic algorithms")
    parser.add_argument(
        "--alpha0",
        default=2000.0,
        help="Initial subgradient steplength")
    parser.add_argument(
        "--nar",
        default=50,
        help="Number of iterations without heuristic")
    parser.add_argument(
        "--rho",
        default=0.96,
        help="Round solution using heuristic algorithms")
    parser.add_argument(
        "--epsilon",
        default=0.01,
        help="Convergence threshold")
    args = parser.parse_args()

    # Parse instance file
    instance = SUPInstance.from_file(args.instance_file_path)

    obj, total_time, n_violated, l_k = solve_problem(
        instance, relax=args.relax, _round=args.round, decompose=args.decompose,
        _epsilon=float(args.epsilon), _alpha0=float(args.alpha0),
        _rho=float(args.rho), _nar=int(args.nar))


if __name__ == "__main__":
    main()