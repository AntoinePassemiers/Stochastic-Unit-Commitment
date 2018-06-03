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


def evolve_and_fix(problem, max_n_iter=100000, part_size=2, n_mutations=2, pop_size=100, verbose=True, n_max=50):
    """ Apply custom heuristic 'evolve-and-fix' on the current problem instance with its
    current variable values.

    Args:
        problem (SUCLpProblem):
            Instance of the original problem
        max_n_iter (int, optional):
            Maximum number of iterations of the genetic algorithm (GA)
        part_size (int, optional):
            Partition size of GA (must be < pop_size // 2)
        n_mutations (int, optional):
            Number of mutations to apply to each newly created solution/individual
        pop_size (int, optional):
            Population size
        verbose (bool, optional):
            Whether to display all outputs
        n_max (int, optional):
            Maximum number of evolve-and-fix iterations
    """
    assert(EF_AVAILABLE)

    # Get information that is necessary to reconstruct the problem instance
    # from the cython side of the code
    variables = problem.get_variables()
    int_mask = [var.name[0] in RELAXED_VARIABLES for var in variables]
    solution = problem.get_var_values()
    cy_constraints = problem.get_constraints_as_tuples(
        groups=["3.25", "3.26", "3.29", "3.30", "3.31", "3.32", "3.35", "3.36", "3.37"])
    cp = genetic.CyProblem(cy_constraints)

    # Relax the integrality constraints
    for var in variables[int_mask]:
        var.cat = "Continuous"

    # Round solution using genetic algorithm
    rounded, fitness = cp.round(solution, int_mask, max_n_iter=max_n_iter,
        part_size=part_size, n_mutations=n_mutations, pop_size=pop_size)
    problem.set_var_values(rounded)

    n_violated, _ = problem.constraints_violated()
    last = n_violated
    stage = 0
    n_iter = 0
    fitness_history = list()
    only_3_32_constraints = False
    history = list()
    print("\tApplying evolve-and-fix heuristic...")
    while n_violated > 0 and not only_3_32_constraints and n_iter < n_max:
        n_iter += 1

        # only_3_32_constraints is True, then the heuristic fails because
        # there is at least one constraint that is violated by
        # fixed variables
        only_3_32_constraints = True
        # Iterate over the constraints
        for i, name in enumerate(problem.constraints):
            group = problem.groups[i]
            c = problem.constraints[name]
            # Numerically stable validity check
            if not (c.valid() or abs(c.value()) < 1e-04):
                for var in c.keys():
                    if group in ["3.26", "3.25", "3.36", "3.35"]:
                        # At least one more variable can be fixed
                        only_3_32_constraints = False
                    if group == "3.26" and stage >= 1:
                        if "P" in var.name:
                            # Constraint of type: P_minus[g, s]* u[g, s, t] <= p[g, s, t]
                            # p[g, s, t] is now fixed to P_plus[g, s]
                            var.upBound = var.lowBound = var_value = var.varValue + abs(c.value())
                    if group == "3.25" and stage >= 0:
                        if "U" in var.name:
                            # Constraint of type: p[g, s, t] <= P_plus[g, s] * u[g, s, t]
                            # u[g, s, t] is now fixed to 1
                            if random.random() < 1 - abs(c.value()):
                                var.upBound = var.lowBound = var_value = 1
                    if group == "3.36" and stage >= 2:
                        if "V" in var.name:
                            # Constraint of type: v[g, s, t] >= u[g, s, t] - u[g, s, t-1]
                            # v[g, s, t] is now fixed to 1 with reasonably low probability
                            if random.random() < (1 - abs(c.value())) / 2.:
                                var.upBound = var.lowBound = var_value = 1
                    if group == "3.35" and stage >= 2:
                        if "Z" in var.name:
                            # Constraint of type: z[g, t] >= w[g, t] - w[g, t-1] 
                            # z[g, t] is now fixed to 1 with reasonably low probability
                            if random.random() < (1 - abs(c.value())) / 2.:
                                var.upBound = var.lowBound = var_value = 1
        # Solve the linear relaxation
        problem.solve()

        # Get integral (possibly infeasible) solution from the optimal relaxed problem
        solution = problem.get_var_values()
        rounded, fitness = cp.round(solution, int_mask, max_n_iter=max_n_iter,
            part_size=part_size, n_mutations=n_mutations, pop_size=pop_size)
        problem.set_var_values(rounded)
        fitness_history.append(fitness)

        if verbose:
            print("\tIteration %03d - Fitness: %f" % (n_iter, fitness))
        history.append(fitness)

        # Increase the difficulty and allow more variables to be fixed
        n_violated, _ = problem.constraints_violated()
        if fitness == last:
            stage += 1
        last = fitness
    
    # Cancel the relaxation of the integrality constraints
    for var in variables[int_mask]:
        var.cat = "Integer"