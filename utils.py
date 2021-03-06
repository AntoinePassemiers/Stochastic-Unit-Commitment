# -*- coding: utf-8 -*-
# utils.py: Extended Numpy-friendly PuLP API
# authors: Antoine Passemiers, Cedric Simar

import copy
import numpy as np
import pulp


# List of variables that are part of the linear relaxation of SUC
RELAXED_VARIABLES = ["U", "W"]


def lp_array(name, shape, var_type, low_bound=None, up_bound=None):
    """ Create a Numpy array of PuLP variables.

    Args:
        name (str): Base name for the variables.
            Example: if "X" is provided, the first element of a 5x5 array
            will be defined as "X(0,0)"
        shape (tuple): Dimensionality of the new array
        var_type (str): Type of PuLP variables
            (either "Mixed", "Continuous" or "Integer")
        low_bound (:obj:`float`,`int`, optional):
            Lower bound for all variables in the new array
        up_bound (:obj:`float`,`int`, optional):
            Upper bound for all variables in the new array
    """
    variables = np.empty(shape, dtype=np.object)
    for index in np.ndindex(*shape):
        var_name = name + str(tuple(index)).replace(" ", "")
        variables[index] = pulp.LpVariable(
            var_name, lowBound=low_bound, upBound=up_bound, cat=var_type)
    return LpArray(variables, info={"var_type" : var_type})
    

class LpArray(np.ndarray):
    """ PuLP-compatible NumPy array. This extended array
    is able to contain either PuLP variables or constraints.
    Vectorized arithmetical and logical operations are supported
    and produce new arrays of variables/constraints.

    Args:
        input_array (np.ndarray):
            Regular NumPy array containing references to PuLP variables
    """

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
        return LpArray.__vectorized_eq__(self, other)

    @np.vectorize
    def __vectorized_le__(a, b):
        return a <= b

    def __le__(self, other):
        return LpArray.__vectorized_le__(self, other)

    @np.vectorize
    def __vectorized_ge__(a, b):
        return a >= b

    def __ge__(self, other):
        return LpArray.__vectorized_ge__(self, other)

    def __lt__(self, other):
        class OperationNotSupportedError(Exception): pass
        raise OperationNotSupportedError("Operation not supported for pulp.LpVariable")

    def __gt__(self, other):
        class OperationNotSupportedError(Exception): pass
        raise OperationNotSupportedError("Operation not supported for pulp.LpVariable")
    
    def fix_variables(self):
        for index, variable in np.ndenumerate(self):
            variable.lowBound = variable.varValue
            variable.upBound = variable.varValue

    def set_var_values(self, values):
        values = np.asarray(values)
        for index, variable in np.ndenumerate(self):
            variable.varValue = values[index]
    
    def get_var_values(self):
        values = np.empty(self.shape, dtype=np.float)
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
        assert(other is not False)
        if other is True:
            return True
        if isinstance(other, np.ndarray):
            result = self
            for index in np.ndindex(*other.shape):
                self.groups.append(self.current_group_name)
                result = self.add_single_constraint(other[index])
            if len(tuple(other.shape)) == 0:
                self.last_constraint_mat_shape = (1,)
            else:
                self.last_constraint_mat_shape = tuple(other.shape)
        else:
            self.groups.append(self.current_group_name)
            result = self.add_single_constraint(other)
            self.last_constraint_mat_shape = (1,)
        return result
    
    def add_single_constraint(self, constraint):
        assert(not isinstance(constraint, np.bool_))
        return super(SUCLpProblem, self).__iadd__(constraint)
    
    def is_integer_solution(self, eps=1e-04):
        for variable in self.variables():
            if variable.name[0] in RELAXED_VARIABLES:
                if abs(round(variable.varValue) - variable.varValue) > eps:
                    return False
        return True
    
    def set_constraint_group(self, name):
        self.current_group_name = name

    def assert_shape(self, *args):
        assert(tuple(args) == self.last_constraint_mat_shape)
    
    def constraints_violated(self, eps=1e-03):
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
    
    def get_fitness(self, eps=1e-04):
        fitness = 0
        for i, name in enumerate(self.constraints):
            constraint = self.constraints[name]
            if not (constraint.valid() or abs(constraint.value()) < eps):
                fitness += abs(constraint.value())
        return fitness
    
    def get_constraints_as_tuples(self, groups=None):
        """ Get list of constraints in a non-PuLP format.

        Each constraint is represented as a tuple containing:
            var_ids: list
                The (integer) identifiers of the variables involved in the constraint
            coefs: list
                The values corresponding to the variables in var_ids
            sense: int
                Inequality sense, where the inequality is of the form 
                Sum_i coef_i*var_i sense intercept.
                1 stands for ">=", -1 for "<=", and 0 for"==".
            intercept: float, int
                Constant right-hand side of the inequality
        """
        all_var_ids = {var.name: i for i, var in enumerate(list(self.variables()))}
        constraints = list()
        for i, name in enumerate(self.constraints):
            group = self.groups[i]
            if groups is None or group in groups:
                c = self.constraints[name]
                sense = c.sense
                intercept = (c.getLb() if sense == 1 else c.getUb())
                var_ids = [all_var_ids[var.name] for var in c.keys()]
                coefs = np.asarray(list(c.values()))
                constraints.append((var_ids, coefs, sense, intercept))
                assert(sense in [1, 0, -1] and intercept is not None)
        return constraints
    
    def get_variables(self):
        return LpArray(list(self.variables()), info={"var_type" : "Mixed"})
    
    def get_var_values(self):
        return self.get_variables().get_var_values()

    def set_var_values(self, values):
        LpArray(self.variables(), info={"var_type" : "Mixed"}).set_var_values(values)
