# -*- coding: utf-8 -*-
# utils.py
# authors: Antoine Passemiers, Cedric Simar

import copy
import numpy as np
import pulp


class LpVarArray:

    def __init__(self, name, shape, var_type, low_bound=None, up_bound=None):
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
            Type of PuLP variables (either "Continuous" or "Integer")
        low_bound: float, int (optional)
            Lower bound for all variables in the new array
        up_bound: float, int (optional)
            Upper bound for all variables in the new array
        """
        self.variables = np.empty(shape, dtype=np.object)
        for index in np.ndindex(*shape):
            var_name = name + str(tuple(index)).replace(" ", "")
            self.variables[index] = pulp.LpVariable(
                var_name, lowBound=low_bound, upBound=up_bound, cat=var_type)
    
    def __getitem__(self, index):
        item = self.variables[index]
        if isinstance(item, np.ndarray):
            result = copy.copy(self)
            result.variables = item
        else:
            result = item
        return result

    def __add__(self, other):
        print(type(other))
        result = copy.copy(self)
        other = other if not isinstance(other, LpVarArray) else other.variables
        result.variables = self.variables + other
        print(result.variables)
        return result
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        result = copy.copy(self)
        other = other if not isinstance(other, LpVarArray) else other.variables
        result.variables = self.variables - other
        return result

    def __rsub__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        result = copy.copy(self)
        result.variables = self.variables * other
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    @np.vectorize
    def __vectorized_eq__(a, b):
        return a == b

    def __eq__(self, other):
        other = other if not isinstance(other, LpVarArray) else other.variables
        return LpVarArray.__vectorized_eq__(self.variables, other)

    @np.vectorize
    def __vectorized_le__(a, b):
        return a <= b

    def __le__(self, other):
        other = other if not isinstance(other, LpVarArray) else other.variables
        return LpVarArray.__vectorized_le__(self.variables, other)

    @np.vectorize
    def __vectorized_ge__(a, b):
        return a >= b

    def __ge__(self, other):
        other = other if not isinstance(other, LpVarArray) else other.variables
        return LpVarArray.__vectorized_ge__(self.variables, other)

    def __lt__(self, other):
        class OperationNotSupportedError(Exception): pass
        raise OperationNotSupportedError("Operation not supported for pulp.LpVariable")

    def __gt__(self, other):
        class OperationNotSupportedError(Exception): pass
        raise OperationNotSupportedError("Operation not supported for pulp.LpVariable")


class ArrayFriendlyLpProblem(pulp.LpProblem):

    def __init__(self, *args, **kwargs):
        pulp.LpProblem.__init__(self, *args, **kwargs)
    
    def __iadd__(self, other):
        if isinstance(other, np.ndarray):
            result = self
            for index in np.ndindex(*other.shape):
                result = super(ArrayFriendlyLpProblem, self).__iadd__(other[index])
        else:
            result = super(ArrayFriendlyLpProblem, self).__iadd__(other)
        return result
    

if __name__ == "__main__":

    prob = ArrayFriendlyLpProblem("The fancy indexing problem", pulp.LpMinimize)

    X = LpVarArray("X", (5, 5), "Continuous", up_bound=80)
    Y = LpVarArray("Y", (7, 5, 5), "Continuous", up_bound=500)

    prob += (1.4*X[:, 2] <= [8, 7, 6, 5, 4])
    prob += (X[0, :] >= 8)
    prob += (0 <= X[1, :] + 2.5)
    prob += (X[2, 1] <= X[3, 4] - X[4, 4]*4.5)
    prob += (X[:, 0] * [1, 2, 3, 4, 5] <= 1)

    print("A")
    prob += (4*X[:, :] + 9*Y[3, :, :] <= 2*Y[2, :, :] + 7)
    print("B")

    print(prob.constraints)