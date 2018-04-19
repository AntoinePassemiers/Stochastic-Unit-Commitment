# -*- coding: utf-8 -*-
# utils.py: Extended Numpy-friendly PuLP API
# authors: Antoine Passemiers, Cedric Simar

import copy
import numpy as np
import pulp


def lp_array(name, shape, var_type, low_bound=None, up_bound=None):
    """ Creates a Numpy array of PuLP variables

    Parameters
    ----------
    name: str
        Base name for the variables.
        Example: if "X" is provided, the first element of a 5x5 array
        will be defined as "X(0,0)"
    shape: tuple
        Dimensionality of the new array
    var_type: str
        Type of PuLP variables (either "Mixed", "Continuous" or "Integer")
    low_bound: float, int (optional)
        Lower bound for all variables in the new array
    up_bound: float, int (optional)
        Upper bound for all variables in the new array
    """
    variables = np.empty(shape, dtype=np.object)
    for index in np.ndindex(*shape):
        var_name = name + str(tuple(index)).replace(" ", "")
        variables[index] = pulp.LpVariable(
            var_name, lowBound=low_bound, upBound=up_bound, cat=var_type)
    return LpVarArray(variables, info={"var_type" : var_type})
    

class LpVarArray(np.ndarray):

    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        obj.var_type = info["var_type"]
        assert(obj.var_type in ["Mixed", "Continuous", "Integer"])
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    @np.vectorize
    def __vectorized_eq__(a, b):
        return a == b

    def __eq__(self, other):
        return LpVarArray.__vectorized_eq__(self, other)

    @np.vectorize
    def __vectorized_le__(a, b):
        return a <= b

    def __le__(self, other):
        return LpVarArray.__vectorized_le__(self, other)

    @np.vectorize
    def __vectorized_ge__(a, b):
        return a >= b

    def __ge__(self, other):
        return LpVarArray.__vectorized_ge__(self, other)

    def __lt__(self, other):
        class OperationNotSupportedError(Exception): pass
        raise OperationNotSupportedError("Operation not supported for pulp.LpVariable")

    def __gt__(self, other):
        class OperationNotSupportedError(Exception): pass
        raise OperationNotSupportedError("Operation not supported for pulp.LpVariable")

    def set_var_values(self, values):
        values = np.asarray(values)
        for index, variable in np.ndenumerate(self):
            variable.varValue = values[index]
    
    def get_var_values(self):
        dtype = np.float if (self.var_type in ["Continuous", "Mixed"]) else np.int
        values = np.empty(self.shape, dtype=dtype)
        for index, variable in np.ndenumerate(self):
            values[index] = variable.varValue
        return values


class ArrayCompatibleLpProblem(pulp.LpProblem):

    def __init__(self, *args, **kwargs):
        pulp.LpProblem.__init__(self, *args, **kwargs)
    
    def __add__(self, other):
        return self.__iadd__(other)
    
    def __iadd__(self, other):
        if isinstance(other, np.ndarray):
            result = self
            for index in np.ndindex(*other.shape):
                result = super(ArrayCompatibleLpProblem, self).__iadd__(other[index])
        else:
            result = super(ArrayCompatibleLpProblem, self).__iadd__(other)
        return result
    
    def constraints_violated(self):
        n_violated = 0
        for name in self.constraints:
            constraint = self.constraints[name]
            if not constraint.valid():
                n_violated += 1
        return n_violated
    
    def get_constraints_as_tuples(self):
        """ Get list of constraints in a non-PuLP format. Each constraint
        is a tuple containing:

        var_ids: list
            The (integer) identifiers of the variables involved in the constraint
        values: list
            The values corresponding to the variables in var_ids
        sense: int
            Inequality sense, where the inequality is of the form Sum_i coef_i*var_i sense intercept.
            1 stands for ">=" and -1 stands for "<=".
        intercept: float, int
            Constant right-hand side of the inequality
        """
        all_var_ids = {var.name: i for i, var in enumerate(self.variables())}
        constraints = list()
        for name in self.constraints:
            c = self.constraints[name]
            sense = c.sense
            intercept = c.getLb() if sense == 1 else c.getUb()
            var_ids = [all_var_ids[var.name] for var in c.keys()]
            values = list(c.values())
            constraints.append((var_ids, values, sense, intercept))
            assert(sense in [1, -1] and intercept is not None)
        return constraints
    
    def get_variables(self):
        return LpVarArray(self.variables(), info={"var_type" : "Mixed"})
    

if __name__ == "__main__":
    """
    prob = ArrayCompatibleLpProblem("The fancy indexing problem", pulp.LpMinimize)

    X = lp_array("X", (5, 5), "Continuous", up_bound=80)
    Y = lp_array("Y", (7, 5, 5), "Continuous", up_bound=500)


    prob += (1.4*X[:, 2] <= [8, 7, 6, 5, 4])
    prob += (X[0, :] >= 8)
    prob += (0 <= X[1, :] + 2.5)
    prob += (X[2, 1] <= X[3, 4] - X[4, 4]*4.5)
    prob += (X[:, 0] * [1, 2, 3, 4, 5] <= 1)

    prob += (4*X[:, :] + 9*Y[3, :, :] <= 2*Y[2, :, :] + 7)
    prob += (X.sum() == 8)

    for name in prob.constraints:
        print(prob.constraints[name])

    #print(sum(X))
    """

    X = lp_array("X", (2, 2), "Continuous")
    X.set_var_values([[0, 1], [9, 3]])
    Y = lp_array("Y", (2, 2), "Continuous")
    Y.set_var_values([[7, -8], [2, -3]])

    problem = ArrayCompatibleLpProblem()
    problem += (Y + X >= 5)
    print(problem.constraints_violated())

    print(problem.get_constraints_as_tuples())
    print(problem.variables())