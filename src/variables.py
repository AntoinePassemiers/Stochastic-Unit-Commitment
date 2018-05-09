# -*- coding: utf-8 -*-
# variables.py: SUC variables
# authors: Antoine Passemiers, Cedric Simar

from utils import lp_array


def init_variables(Gs, Gf, S, T, N, L, I, relax=True):
    var_type = "Continuous" if relax else "Integer"
    G = len(Gs) + len(Gf) # Number of generators

    # u[g, s, t] = Commitment of generator g in scenario s, period t
    u = lp_array("U", (G, S, T), var_type, 0, 1)

    # v[g, s, t] = Startup of generator g in scenario s, period t
    v = lp_array("V", (G, S, T), "Continuous", low_bound=0)

    # p[g, s, t] = Production of generator g in scenario s, period t
    p = lp_array("P", (G, S, T), "Continuous", low_bound=0)

    # theta[n, s, t] = Phase angle at bus n in scenario s, period t
    theta = lp_array("THETA", (N, S, T), "Continuous")

    # w[g, t] = Commitment of slow generator g in period t
    w = lp_array("W", (len(Gs), T), var_type, 0, 1)

    # z[g, t] = Startup of slow generator g in period t
    z = lp_array("Z", (len(Gs), T), "Continuous", low_bound=0)

    # e[l, s, t] = Power flow on line l in scenario s, period t
    e = lp_array("E", (L, S, T), "Continuous")

    return u, v, p, theta, w, z, e