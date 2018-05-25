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


def solve_problem(instance, relax=True, _round=False, decompose=False):
    print("Solving problem...")
    start = time.time()
    if not decompose:
        # Formulate the LP relaxation for the given instance
        problem, (u, v, p, theta, w, z, e) = create_formulation(instance, relax=relax)
        # Solve LP relaxation
        problem.solve()
        l_k = None
    else:
        problem, l_k = solve_with_subgradient(instance)
        evolve_and_fix(problem)
    exec_time = time.time() - start
    obj = problem.objective.value()
    print("Solve time: %f s" % exec_time)
    print("Problem status: %s" % pulp.LpStatus[problem.status])
    if problem.status == pulp.constants.LpStatusOptimal:
        print("Value of the objective: %f" % obj)

    if _round:
        heuristic_start = time.time()
        if _round == "evolve-and-fix":
            evolve_and_fix(problem)
        elif _round == "dive-and-fix":
            dive_and_fix(problem, variables)
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
        choices=['evolve-and-fix', 'dive-and-fix'],
        help="Round solution using heuristic algorithms")
    args = parser.parse_args()

    # Parse instance file
    instance = SUPInstance.from_file(args.instance_file_path)

    obj, total_time, n_violated = solve_problem(
        instance, relax=args.relax, _round=args.round, decompose=args.decompose)


if __name__ == "__main__":
    
    folder = "../instances"
    for filename in os.listdir(folder):
        if "inst-10-12-5" in filename:
            print(filename)
            with open("results.txt", "r") as f:
                filenames = [line.split(",")[0] for line in f.readlines()]
                
            if(filename not in filenames):

                filepath = os.path.join(folder, filename)
                instance = SUPInstance.from_file(filepath)

                obj, total_time, n_violated, l_k = solve_problem(instance, relax=True, _round="evolve-and-fix", decompose=True)

                with open("results.txt", "a") as f:
                    if l_k is None:
                        f.write("%s, %f, %f, %d\n" % (filename, obj, total_time, n_violated))
                    else:
                        f.write("%s, %f, %f, %d, %f\n" % (filename, obj, total_time, n_violated, l_k))
            else:
                print("File : " + filename + " already computed")
    
    # main()