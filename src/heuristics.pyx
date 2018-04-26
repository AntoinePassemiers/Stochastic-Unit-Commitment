# -*- coding: utf-8 -*-
# heuristics.pyx
# authors: Antoine Passemiers, Cedric Simar

import numpy as np
cimport numpy as cnp
cnp.import_array()

from libc.stdio cimport printf
from libc.stdlib cimport malloc, calloc, rand, RAND_MAX
from libc.string cimport memcpy, memset
cimport libc.math


np_data_t = np.float
ctypedef cnp.float_t data_t

cdef struct constraint_t:
    int        n_values  # Size of var_ids and values
    cnp.int_t* var_ids   # Identifiers of the variables involved
    data_t*    values    # Coefficients of the variables
    data_t     rhs       # Right-hand side of the constraint (intercept)
    int        sense     # Either 1, 0 or -1
    int        satisfied # Whether it is currently satisfied


cdef constraint_t __create_constraint(cnp.int_t[::1] var_ids,
                                      data_t[::1] values,
                                      int sense,
                                      data_t rhs):
    """
    Create a C struct constraint from python constraint.
    
    Parameters
    ----------
    var_ids: cnp.int_t[::1]
        Identifiers of the variables involved in the constraint
    values: data_t[::1]
        Coefficients of the variables involved
    sense: int
        Sense is "lhs >= rhs" if 1 else "lhs <= rhs"
    rhs: data_t
        Right-hand side of the constraint (intercept)
    """
    cdef constraint_t constraint
    cdef int n_values = var_ids.shape[0]
    constraint.n_values = n_values
    constraint.var_ids = <cnp.int_t*>malloc(n_values * sizeof(cnp.int_t))
    memcpy(constraint.var_ids, &var_ids[0], n_values * sizeof(cnp.int_t))
    constraint.values = <data_t*>malloc(n_values * sizeof(data_t))
    memcpy(constraint.values, &values[0], n_values * sizeof(data_t))
    constraint.rhs = rhs
    constraint.sense = sense
    constraint.satisfied = 1
    return constraint


cdef inline bint __is_satisfied(constraint_t* constraint, data_t* solution, data_t eps) nogil:
    """
    Given a solution, check whether a constraint is satisfied.

    Parameters
    ----------
    constraint: constraint_t
        Constraint to be evaluated
    solution: data_t*
        Current solution to the problem. Its size is equal to the number
        of variables in the problem instance.
    eps: data_t
        Numerical precision of the solution
    """
    cdef data_t lhs = 0
    cdef int i
    for i in range(constraint.n_values):
        lhs += constraint.values[i] * \
            solution[constraint.var_ids[i]]
    if constraint.sense < 0:
        constraint.satisfied = (lhs <= constraint.rhs + eps)
    elif constraint.sense > 0:
        constraint.satisfied = (lhs >= constraint.rhs - eps)
    else:
        constraint.satisfied = ((lhs == constraint.rhs) or \
            (libc.math.fabs(lhs - constraint.rhs) < eps))
    return constraint.satisfied


cdef int __constraints_violated(data_t* solution,
                                constraint_t* constraints,
                                int n_constraints,
                                data_t eps) nogil:
    """
    Given a solution, check whether a set of constraints is satisfied.

    Parameters
    ----------
    solution: data_t*
        Current solution to the problem. Its size is equal to the number
        of variables in the problem instance.
    constraints: constraint_t
        Constraints to be evaluated
    n_constraints: int
        Number of constraints
    eps: data_t
        Numerical precision of the solution
    """
    cdef int n_violated = 0
    cdef int i
    with nogil:
        for i in range(n_constraints):
            n_violated += not __is_satisfied(&constraints[i], solution, eps)
    return n_violated


cdef void __round_solution(data_t[::1] solution,
                           data_t[::1] rounded,
                           cnp.uint8_t[::1] int_mask,
                           constraint_t* constraints,
                           int n_constraints,
                           data_t eps) nogil:
    cdef int n_violated
    cdef int i, j, k
    for i in range(solution.shape[0]):
        rounded[i] = libc.math.round(solution[i]) if int_mask[i] else solution[i]

    n_violated = __constraints_violated(&rounded[0], constraints, n_constraints, eps)
    for k in range(20):
        printf("%d\n", n_violated)

        for i in range(solution.shape[0]):
            if int_mask[i]:
                rounded[i] = (<double>rand() / <double>RAND_MAX < 0.5)
        
        n_violated = __constraints_violated(&rounded[0], constraints, n_constraints, eps)


cdef class CyProblem:

    cdef int n_constraints
    cdef constraint_t* constraints
    cdef data_t eps
    cdef int n_variables

    def __cinit__(self, constraints, eps=1e-04):
        self.eps = eps
        self.n_constraints = len(constraints)
        self.constraints = <constraint_t*>malloc(
            self.n_constraints * sizeof(constraint_t))
        for i in range(self.n_constraints):
            self.constraints[i] = __create_constraint(
                np.ascontiguousarray(constraints[i][0], dtype=np.int),
                np.ascontiguousarray(constraints[i][1], dtype=np_data_t),
                constraints[i][2],
                constraints[i][3])
    
    def constraints_violated(self, solution):
        cdef data_t[::1] solution_buf = np.ascontiguousarray(solution, dtype=np_data_t)
        return __constraints_violated(&solution_buf[0], self.constraints,
            self.n_constraints, self.eps)
    
    def round(self, solution, int_mask, eps=1e-04):
        rounded = np.empty_like(solution)
        __round_solution(solution, rounded, np.asarray(int_mask, dtype=np.uint8),
            self.constraints, self.n_constraints, eps=eps)
        return rounded