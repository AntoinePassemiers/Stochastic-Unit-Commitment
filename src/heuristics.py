# -*- coding: utf-8 -*-
# heuristics.py
# authors: Antoine Passemiers, Cedric Simar

from utils import LpArray, SUCLpProblem, RELAXED_VARIABLES

import random
import numpy as np
import matplotlib.pyplot as plt


EF_AVAILABLE = False
try:
    import pyximport
    try:
        pyximport.install(setup_args={'include_dirs': np.get_include()})
        import genetic
    except ImportError:
        print("[Warning] Compilation failed with Cython")
    EF_AVAILABLE = True
except ImportError:
    print("[Warning] Cython is not installed on your computer")
if not EF_AVAILABLE:
    print("[Warning] Evolve-and-fix is not available")

def evolve_and_fix(problem):
    assert(EF_AVAILABLE)
    variables = problem.get_variables()
    int_mask = [var.name[0] in RELAXED_VARIABLES for var in variables]
    solution = problem.get_var_values()
    cy_constraints = problem.get_constraints_as_tuples(
        groups=["3.25", "3.26", "3.29", "3.30", "3.31", "3.32", "3.35", "3.36", "3.37"])
    cp = genetic.CyProblem(cy_constraints)

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

        print("Iteration %03d - Fitness: %f" % (n_iter, fitness))

        n_violated, _ = problem.constraints_violated()
        if fitness == last:
            stage += 1
        last = fitness
    
    if n_violated == -1:
        print("[Warning] Evolve-and-fix heuristic failed")
    else:
        plt.plot(fitness_history)
        plt.show()
