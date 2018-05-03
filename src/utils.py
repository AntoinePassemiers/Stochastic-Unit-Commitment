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


class SUCLpProblem(pulp.LpProblem):

    def __init__(self, *args, **kwargs):
        pulp.LpProblem.__init__(self, *args, **kwargs)
        self.last_constraint_mat_shape = None
        self.ordered_variables = None
        self.all_var_ids = None
        self.groups = list()
        self.current_group_name = "-"
    
    def __iadd__(self, other):
        if isinstance(other, np.ndarray):
            result = self
            for index in np.ndindex(*other.shape):
                self.groups.append(self.current_group_name)
                result = super(SUCLpProblem, self).__iadd__(other[index])
            if len(tuple(other.shape)) == 0:
                self.last_constraint_mat_shape = (1,)
            else:
                self.last_constraint_mat_shape = tuple(other.shape)
        else:
            self.groups.append(self.current_group_name)
            result = super(SUCLpProblem, self).__iadd__(other)
            self.last_constraint_mat_shape = (1,)
        return result
    
    def is_integer_solution(self, eps=1e-04):
        for variable in self.variables():
            if variable.name[0] in ["U", "W"]:
                if abs(round(variable.varValue) - variable.varValue) > eps:
                    print(variable.name, variable.varValue)
                    return False
        return True
    
    def set_constraint_group(self, name):
        self.current_group_name = name

    def assert_shape(self, *args):
        assert(tuple(args) == self.last_constraint_mat_shape)
    
    def constraints_violated(self, eps=1e-04):
        groups_n_violated = dict()
        n_violated = 0
        for i, name in enumerate(self.constraints):
            if self.groups[i] not in groups_n_violated.keys():
                groups_n_violated[self.groups[i]] = [0, 0]
            constraint = self.constraints[name]
            if not (constraint.valid() or abs(constraint.value()) < eps):
                n_violated += 1
                if self.groups[i] in groups_n_violated.keys():
                    groups_n_violated[self.groups[i]][0] += 1
            groups_n_violated[self.groups[i]][1] += 1
        return n_violated, groups_n_violated
    
    def get_constraints_as_tuples(self):
        """ Get list of constraints in a non-PuLP format. Each constraint
        is a tuple containing:

        var_ids: list
            The (integer) identifiers of the variables involved in the constraint
        coefs: list
            The values corresponding to the variables in var_ids
        sense: int
            Inequality sense, where the inequality is of the form Sum_i coef_i*var_i sense intercept.
            1 stands for ">=", -1 for "<=", and 0 for"==".
        intercept: float, int
            Constant right-hand side of the inequality
        """
        all_var_ids = {var.name: i for i, var in enumerate(list(self.variables()))}
        constraints = list()
        for name in self.constraints:
            c = self.constraints[name]
            sense = c.sense
            intercept = c.getLb() if sense == 1 else c.getUb()
            var_ids = [all_var_ids[var.name] for var in c.keys()]
            coefs = list(c.values())
            constraints.append((var_ids, coefs, sense, intercept))
            assert(sense in [1, 0, -1] and intercept is not None)
        return constraints
    
    def get_variables(self):
        return LpVarArray(list(self.variables()), info={"var_type" : "Mixed"})
    
    def set_var_values(self, values):
        for i, variable in enumerate(list(self.variables())):
            variable.varValue = values[i]
    

if __name__ == "__main__":
    """
    prob = SUCLpProblem("The fancy indexing problem", pulp.LpMinimize)

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

    x = pulp.LpVariable("x", lowBound=0, upBound=0.9, cat="Continuous")

    problem = pulp.LpProblem("qskjd", pulp.LpMaximize)

    problem += 2*x

    constraint = (0.5*x <= 0.1)
    problem += constraint

    problem.solve()

    print(dir(constraint))

    print(constraint.valid())

    print(constraint.values())


