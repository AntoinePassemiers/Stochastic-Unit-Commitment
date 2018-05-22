# -*- coding: utf-8 -*-
# dive_and_fix.py: Rounding algorithm for SUC
# authors: Antoine Passemiers, Cedric Simar

from utils import LpArray

import numpy as np


def dive_and_fix(problem, variables):
    (u, v, p, theta, w, z, e) = variables

    variables = problem.get_variables()
    int_mask = [var.name[0] in ["U", "W"] for var in variables]

    while True:
        variables = problem.get_variables()
        solution = variables.get_var_values()
        print(problem.status, np.sum(np.abs(solution[int_mask] - \
            np.round(solution[int_mask])) >= 1e-04), problem.objective.value())
        if problem.status == -1:
            break

        indices = np.where(np.logical_and(
            int_mask,
            np.abs(solution - np.round(solution)) >= 1e-04))[0]
        if len(indices) == 0:
            break

        less_fractional = np.argmin(
            np.abs(np.round(solution[indices]) - solution[indices]))
        h = indices[less_fractional]
        value = 0 if variables[h].varValue < 0.5 else 1

        print(variables[h].varValue)
        variables[h].varValue = value
        variables[h].lowBound = value
        variables[h].upBound = value
        problem.solve()