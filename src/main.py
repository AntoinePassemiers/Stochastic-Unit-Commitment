# -*- coding: utf-8 -*-
# main.py
# authors: Antoine Passemiers, Cedric Simar

from decomposition import decompose_problem
from instance import SUPInstance
from lp_relaxation import init_variables, create_formulation
from dive_and_fix import dive_and_fix
from subgradient import solve_with_subgradient
from utils import LpArray, SUCLpProblem, RELAXED_VARIABLES

import os
import sys
import time
import argparse
import random
import pulp
import numpy as np
import matplotlib.pyplot as plt

try:
    import pyximport
    pyximport.install(setup_args={'include_dirs': np.get_include()})
    import genetic
except ImportError:
    print("You definitely should install Cython.")
    sys.exit(0)


def solve_problem(instance, relax=True):
    # Formulate the LP relaxation for the given instance
    problem, (u, v, p, theta, w, z, e) = create_formulation(instance, relax=relax)

    # Solve LP relaxation
    print("Solving problem...")
    problem.writeLP("prob.lp")
    start = time.time()
    problem.solve()
    exec_time = time.time() - start
    obj = problem.objective.value()
    print("Solve time: %f s" % exec_time)
    print("Problem status: %s" % pulp.LpStatus[problem.status])
    if problem.status == pulp.constants.LpStatusOptimal:
        print("Value of the objective: %f" % obj)

    start = time.time()
    if args.round == "evolve-and-fix":

        variables = problem.get_variables()
        int_mask = [var.name[0] in RELAXED_VARIABLES for var in variables]
        solution = problem.get_var_values()
        cy_constraints = problem.get_constraints_as_tuples(
            groups=["3.25", "3.26", "3.29", "3.30", "3.31", "3.32", "3.35", "3.36", "3.37"])
        cp = genetic.CyProblem(cy_constraints)

        print("n int:", )

        max_n_iter = 10000
        part_size = 4
        n_mutations = int(np.sum(int_mask) / .05)
        pop_size = 200

        rounded, fitness = cp.round(solution, int_mask, max_n_iter=max_n_iter,
            part_size=part_size, n_mutations=n_mutations, pop_size=pop_size)
        problem.set_var_values(rounded)

        n_violated, _ = problem.constraints_violated()
        last = n_violated
        stage = 0
        max_n_iter = 50
        n_iter = 0
        fitness_history = list()
        only_3_32_constraints = False
        while n_violated > 0 and not only_3_32_constraints:
            n_iter += 1

            only_3_32_constraints = True
            for i, name in enumerate(problem.constraints):
                group = problem.groups[i]
                c = problem.constraints[name]
                if not (c.valid() or abs(c.value()) < 1e-04):
                    for var in c.keys():
                        if group in ["3.26", "3.25", "3.36", "3.35"]:
                            only_3_32_constraints = False
                        if group == "3.26" and stage >= 1:
                            if "P" in var.name:
                                var.upBound = var.lowBound = var_value = var.varValue + abs(c.value())
                        if group == "3.25" and stage >= 0:
                            if "U" in var.name:
                                if random.random() < 1 - abs(c.value()):
                                    var.upBound = var.lowBound = var_value = 1
                        if group == "3.36" and stage >= 2:
                            if "V" in var.name:
                                if random.random() < (1 - abs(c.value())) / 2.:
                                    var.upBound = var.lowBound = var_value = 1
                        if group == "3.35" and stage >= 2:
                            if "Z" in var.name:
                                if random.random() < (1 - abs(c.value())) / 2.:
                                    var.upBound = var.lowBound = var_value = 1
            problem.solve()

            solution = problem.get_var_values()
            rounded, fitness = cp.round(solution, int_mask, max_n_iter=max_n_iter,
                part_size=part_size, n_mutations=n_mutations, pop_size=pop_size)
            problem.set_var_values(rounded)
            fitness_history.append(fitness)

            print("Iteration %i - Fitness: %f" % (n_iter, fitness))

            n_violated, _ = problem.constraints_violated()
            if fitness == last:
                stage += 1
            last = fitness
        
        if n_violated == -1:
            print("[Warning] Evolve-and-fix heuristic failed")
        else:
            plt.plot(fitness_history)
            plt.show()

        obj = problem.objective.value()
        print("Value of the objective: %f" % obj)
        print("Is integer: ", problem.is_integer_solution())

        #problem, u, w = create_formulation(instance, relax=False, lower_bound=1.1*obj)
        #problem.solve()
        #dive_and_fix(problem, variables)
    
    print("Rounding time: %f s" % (time.time() - start))

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


if __name__ == "__main__":

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
        choices=['evolve-and-fix'],
        help="Round solution using heuristic algorithms")
    args = parser.parse_args()

    # Parse instance file
    instance = SUPInstance.from_file(args.instance_file_path)

    if args.decompose:
        solve_with_subgradient(instance)
    else:
        solve_problem(instance, relax=args.relax)
