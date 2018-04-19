# -*- coding: utf-8 -*-
# instance.py
# authors: Antoine Passemiers, Cedric Simar

import numpy as np


class SUPInstance:

    def __init__(self, n_generators, n_scenarios, n_periods, n_buses, n_lines, n_import_groups):
        self.n_generators = n_generators
        self.n_scenarios = n_scenarios
        self.n_periods = n_periods
        self.n_buses = n_buses
        self.n_lines = n_lines
        self.n_import_groups = n_import_groups
    
        # PI[s] = Probability of scenario s
        self.PI = np.empty(self.n_scenarios, dtype=np.float)

        # K[g] = Minimum load cost of generator g
        self.K = np.empty(self.n_generators, dtype=np.float)

        # S[g] = Startup cost of generator g
        self.S = np.empty(self.n_generators, dtype=np.float)
        
        # C[g] = Marginal cost of generator g
        self.C = np.empty(self.n_generators, dtype=np.float)

        # D[n, s, t] = Demand in bus n, scenario s, period t
        self.D = np.empty((self.n_buses, self.n_scenarios, self.n_periods), dtype=np.float)

        # P_plus[g, s] = Maximum capacity of generator g in scenario s
        self.P_plus = np.empty((self.n_generators, self.n_scenarios), dtype=np.float)

        # P_minus[g, s] = Minimum capacity of generator g in scenario s
        self.P_minus = np.empty((self.n_generators, self.n_scenarios), dtype=np.float)

        # R_plus[g] = Maximum ramping of generator g
        self.R_plus = np.empty(self.n_generators, dtype=np.float)

        # R_minus[g] = Minimum ramping of generator g
        self.R_minus = np.empty(self.n_generators, dtype=np.float)

        # UT[g] = Minimum up time of generator g
        self.UT = np.empty(self.n_generators, dtype=np.float)

        # DT[g] = Minimum down time of generator g
        self.DT = np.empty(self.n_generators, dtype=np.float)

        # TODO: N = Number of periods in horizon?

        # T_req[t] = Total reserve requirement in period t
        self.T_req = np.empty(self.n_periods, dtype=np.float)

        # F_req[t] = Fast reserve requirement in period t
        self.F_req = np.empty(self.n_periods, dtype=np.float)

        # B[l, s] = Susceptance of line l in scenario s
        self.B = np.empty((self.n_lines, self.n_scenarios), dtype=np.float)

        # TC[l] = Maximum capacity of line l
        self.TC = np.empty(self.n_lines, dtype=np.float)

        # FR[g] = Fast reserve limit of generator g
        self.FR = np.empty(self.n_generators, dtype=np.float)

        # IC[j] = Maximum capacity of import group j
        self.IC = np.empty(n_import_groups, dtype=np.float)

        # GAMMA[j, l] = Polarity of line l in import group j
        self.GAMMA = np.empty((self.n_import_groups, self.n_lines), dtype=np.float)

    def get_constants(self):
        return (self.PI, self.K, self.S, self.C, self.D, self.P_plus, self.P_minus,
            self.R_plus, self.R_minus, self.UT, self.DT, self.T_req, self.F_req, 
            self.B, self.TC, self.FR, self.IC, self.GAMMA)

