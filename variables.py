# -*- coding: utf-8 -*-
# variables.py: SUC variables
# authors: Antoine Passemiers, Cedric Simar

from utils import lp_array, RELAXED_VARIABLES


def init_variables(Gs, Gf, S, T, N, L, I, relax=True):
    """ Initialize NumPy arrays of PuLP variables required
    to solve the SUC problem.

    Args:
        Gs (np.ndarray):
            Identifiers of slow generators
        Gf (np.ndarray):
            Identifiers of fast generators
        S (int):
            Number of scenarios
        T (int):
            Number of periods
        N (int):
            Number of nodes
        L (int):
            Number of lines
        I (int):
            Number of import groups
        relax (:obj:`bool`, optional):
            Either to relax the integrality constraints on U and W
    """
    G = len(Gs) + len(Gf) # Number of generators

    # u[g, s, t] = Commitment of generator g in scenario s, period t
    var_type = "Continuous" if (relax and "U" in RELAXED_VARIABLES) else "Integer"
    u = lp_array("U", (G, S, T), var_type, 0, 1)

    # v[g, s, t] = Startup of generator g in scenario s, period t
    v = lp_array("V", (G, S, T), "Continuous", 0, 1)

    # p[g, s, t] = Production of generator g in scenario s, period t
    p = lp_array("P", (G, S, T), "Continuous", low_bound=0)

    # theta[n, s, t] = Phase angle at bus n in scenario s, period t
    theta = lp_array("THETA", (N, S, T), "Continuous")

    # w[g, t] = Commitment of slow generator g in period t
    var_type = "Continuous" if (relax and "W" in RELAXED_VARIABLES) else "Integer"
    w = lp_array("W", (G, T), var_type, 0, 1)

    # z[g, t] = Startup of slow generator g in period t
    z = lp_array("Z", (G, T), "Continuous", 0, 1)

    # e[l, s, t] = Power flow on line l in scenario s, period t
    e = lp_array("E", (L, S, T), "Continuous")

    return u, v, p, theta, w, z, e