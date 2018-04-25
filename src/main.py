# -*- coding: utf-8 -*-
# main.py
# authors: Antoine Passemiers, Cedric Simar

from instance import SUPInstance
from lp_relaxation import init_variables, create_lp_relaxation

import argparse
import pulp
import numpy as np

try:
    import pyximport
    pyximport.install(setup_args={'include_dirs': np.get_include()})
    import heuristics
except ImportError:
    print("You definitely should install Cython.")

if not pulp.solvers.GLPK_CMD().available():
    print("You definitely should install GLPK.")


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("instance_file_path")
    parser.add_argument(
        "--round",
        choices=['custom'],
        help="Round solution using heuristic algorithms")
    args = parser.parse_args()

    # Parse instance file
    instance = SUPInstance.from_file(args.instance_file_path)

    # Formulate the LP relaxation for the given instance
    problem = create_lp_relaxation(instance)

    #print(constraint.valid())

    # Solve LP relaxation
    print("Solving problem...")
    time_limit = 30 * 60 # 30 minutes
    problem.writeLP("prob.lp")
    solver = pulp.solvers.GLPK_CMD(options=["--tmlim", str(time_limit)])
    problem.solve() # TODO
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

    print(len(variables))
    print(len(solution))

    print(problem.is_integer_solution())

    cy_problem = heuristics.CyProblem(constraints)
    print("Number of violated constraints (python): %i" % problem.constraints_violated())
    print("Number of violated constraints (cython): %i" % cy_problem.constraints_violated(solution))

    rounded = cy_problem.round(solution, int_mask)


    print(len(rounded))

    problem.set_var_values(rounded)

    print(problem.is_integer_solution())

    if args.round:
        pass # TODO: Use a stochastic rounding algorithm


    with open("solution.txt", "w") as f:
        f.write("Problem status: %i\n" % problem.status)
        f.write("Value of the objective: %f\n" % problem.objective.value())
        for variable in problem.variables():
            f.write("%s = %s\n" % (str(variable.name), str(variable.varValue)))