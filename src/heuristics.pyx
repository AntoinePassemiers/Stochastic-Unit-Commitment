# -*- coding: utf-8 -*-
# heuristics.pyx: Genetic algorithm for increasing the number
#                 of satisfied constraints from an initial solution
# authors: Antoine Passemiers, Cedric Simar
# distutils: language=c
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False

import numpy as np
cimport numpy as cnp
cnp.import_array()

import random

from libc.stdio cimport printf
from libc.stdlib cimport malloc, calloc, rand, RAND_MAX
from libc.string cimport memcpy, memset
cimport libc.math
from cython.parallel import parallel, prange

cdef double INF = <double>np.inf
np_data_t = np.float
ctypedef cnp.float_t data_t
ctypedef data_t* data_ptr_t
ctypedef fused shuffable:
    data_ptr_t
    cnp.int_t

cdef struct constraint_t:
    int        n_values  # Size of var_ids and values
    cnp.int_t* var_ids   # Identifiers of the variables involved
    data_t*    values    # Coefficients of the variables
    data_t     rhs       # Right-hand side of the constraint (intercept)
    int        sense     # Either 1, 0 or -1


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
    return constraint


cdef inline data_t __compute_value(constraint_t* constraint,
                                   data_t* solution,
                                   data_t eps) nogil:
    """
    Given a solution, compute a constraint's value.
    A constraint's value is non-negative if and only if the constraint is satisfied.

    Parameters
    ----------
    constraint: constraint_t
        Constraint to be evaluated
    solution: data_t*
        Current solution to the problem. Its size is equal to the number
        of variables in the problem instance.
    eps: data_t
        Numerical precision of the solution
    oriented: bint
        Whether to return a signed value in case of an equality constraint
    """
    cdef data_t lhs = 0
    cdef data_t value
    cdef int i
    for i in range(constraint.n_values):
        lhs += constraint.values[i] * solution[constraint.var_ids[i]]
    if constraint.sense < 0:
        value = constraint.rhs + eps - lhs
    elif constraint.sense > 0:
        value = lhs + eps - constraint.rhs
    else: # Equality constraint
        value = -libc.math.fabs(lhs - constraint.rhs)
        if value > -eps:
            value = 0
    return value


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
    return __compute_value(constraint, solution, eps) >= 0


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


cdef double __fitness(data_t* solution,
                     constraint_t* constraints,
                     int n_constraints,
                     data_t eps) nogil:
    cdef double fitness = 0.0
    cdef int i
    for i in range(n_constraints):
        if not __is_satisfied(&constraints[i], solution, eps):
            """
            fitness -= libc.math.fabs(
                __compute_value(&constraints[i], solution, eps, False))
            """
            fitness -= 1
    return fitness


cdef void __round_closest(data_t[::1] solution,
                          cnp.uint8_t[::1] int_mask):
    for i in range(solution.shape[0]):
        if int_mask[i]:
            solution[i] = libc.math.round(solution[i])


cdef void __round_solution(data_t[::1] solution,
                           data_t[::1] rounded,
                           cnp.uint8_t[::1] int_mask,
                           constraint_t* constraints,
                           int n_constraints,
                           data_t eps):
    cdef int n_violated, best
    cdef int i, j, k, l, h
    cdef cnp.float_t[:] values

    __round_closest(rounded, int_mask)

    n_violated = __constraints_violated(&rounded[0], constraints, n_constraints, eps)
    for k in range(20):
        proba = np.zeros(n_constraints, dtype=np.float)
        values = np.zeros(n_constraints, dtype=np.float)
        for j in range(n_constraints):
            values[j] = __compute_value(&constraints[j], &rounded[0], eps)
            values[j] = 0 if values[j] >= 0 else values[j]
            proba[j] = np.abs(values[j])
        proba /= proba.sum()
        h = np.random.choice(np.arange(n_constraints), p=proba)

        with nogil:
            for j in range(constraints[h].n_values):
                l = constraints[h].var_ids[j]
                if constraints[h].sense == 1:
                    rounded[l] -= values[h] * constraints[h].values[j]
                elif constraints[h].sense == 0:
                    rounded[l] -= values[h] * constraints[h].values[j]
                else:
                    rounded[l] += values[h] * constraints[h].values[j]
        __round_closest(rounded, int_mask)
        n_violated = __constraints_violated(&rounded[0], constraints, n_constraints, eps)


cdef void shuffle(shuffable* population, size_t pop_size) nogil:
    """

    References:
        Shuffling algorithm inspired by Ben Pfaff's
        general purpose shuffling algorithm.
        https://benpfaff.org/writings/clc/shuffle.html
    """
    cdef int i, j
    cdef shuffable tmp
    if pop_size > 1:
        for i in range(pop_size - 1):
            j = i + rand() / (RAND_MAX / (pop_size - i) + 1)
            tmp = population[j]
            population[j] = population[i]
            population[i] = tmp


cdef inline cnp.int_t find_worst_member(double* F, int pop_size) nogil:
    cdef double worst_value = INF
    cdef int worst_index = 0
    cdef int i
    for i in range(pop_size):
        if F[i] < worst_value:
            worst_value = F[i]
            worst_index = i
    return worst_index


cdef int tournament(cnp.double_t* F,
                    cnp.int_t* indices,
                    int partition_start,
                    int partition_end) nogil:
    cdef cnp.double_t best_value = -INF
    cdef int best_index = 0
    for i in range(partition_start, partition_end):
        if F[i] > best_value:
            best_value = F[indices[i]]
            best_index = indices[i]
    return best_index


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
    
    def round_ga(self, solution, int_mask,
                 int max_n_iter=700, int part_size=30, int n_mutations=50, int pop_size=50):
        assert(2 * part_size <= pop_size)
        cdef int a, b, c, i, j, k, d, alpha, pp
        cdef int n_variables = len(solution)
        cdef data_t[::1] _solution = np.ascontiguousarray(np.copy(solution), dtype=np_data_t)
        cdef cnp.int_t[::1] int_indices = np.ascontiguousarray(np.where(int_mask)[0], dtype=np.int)
        cdef int n_relaxed = int_indices.shape[0]
        cdef bint feasible = False

        cdef cnp.int_t[::1] indices = np.ascontiguousarray(np.arange(pop_size), dtype=np.int)
        cdef cnp.double_t[::1] F = np.ascontiguousarray(np.empty(pop_size), dtype=np.double)

        cdef data_t** population = <data_t**>malloc(pop_size * sizeof(data_t*))
        for i in range(pop_size):
            population[i] = <data_t*>malloc(n_variables * sizeof(data_t))
            memcpy(population[i], &_solution[0], n_variables * sizeof(data_t))
            for j in range(n_relaxed):
                population[i][int_indices[j]] = (_solution[int_indices[j]] > (rand() / RAND_MAX))
        cdef data_t* child

        with nogil, parallel():
            for k in range(max_n_iter):
                shuffle(population, pop_size)
                for i in prange(pop_size):
                    F[i] = __fitness(
                        population[i],
                        self.constraints,
                        self.n_constraints,
                        self.eps)
                    if F[i] >= 0.0:
                        feasible = True
                if feasible:
                    break

                shuffle(&indices[0], pop_size)

                a = tournament(&F[0], &indices[0], 0, part_size)
                b = tournament(&F[0], &indices[0], part_size, 2*part_size)

                c = find_worst_member(&F[0], pop_size) # Child replaces worst member

                for j in prange(n_relaxed):
                    alpha = ((rand() / RAND_MAX) < 0.5)
                    population[c][int_indices[j]] = alpha * population[a][int_indices[j]] + \
                        (1 - alpha) * population[b][int_indices[j]]

                for d in range(n_mutations):
                    pp = rand() % n_relaxed
                    population[c][int_indices[pp]] = 1 - population[c][int_indices[pp]]
            
        print(np.max(F))

        memcpy(&_solution[0], population[np.argmax(F)], n_variables * sizeof(data_t))
        return np.asarray(_solution)