def create_admissible_solution(problem, variables, instance):
    (Gs, Gf, Gn, LIn, LOn, IG, LI_indices, LO_indices, \
        L_node_indices) = instance.get_indices()
    (n_generators, n_scenarios, n_periods, n_lines, \
        n_nodes, n_import_groups) = instance.get_sizes()
    (PI, K, S, C, D, P_plus, P_minus, R_plus, R_minus, \
        UT, DT, T_req, F_req, B, TC, FR, IC, GAMMA) = instance.get_constants()
    (u, v, p, theta, w, z, e) = variables
    variables = problem.get_variables()
    int_mask = [var.name[0] in ["U", "W"] for var in variables]
    solution = variables.get_var_values()
    solution[int_mask] = np.round(solution[int_mask])
    problem.set_var_values(solution)

    n_violated, groups_n_violated = problem.constraints_violated()
    print(n_violated)

    for g in range(n_generators):
        for s in range(n_scenarios):
            for t in range(n_periods):
                ll = P_plus[g, s] * u[g, s, t].varValue if random.random() < 0.5 else P_minus[g, s] * u[g, s, t].varValue
                if p[g, s, t].varValue > P_plus[g, s] * u[g, s, t].varValue:
                    p[g, s, t].varValue = P_plus[g, s] * u[g, s, t].varValue
                elif p[g, s, t].varValue < P_minus[g, s] * u[g, s, t].varValue:
                    p[g, s, t].varValue = P_minus[g, s] * u[g, s, t].varValue


    subprob = SUCLpProblem("-", pulp.LpMinimize)
    obj = C * np.swapaxes(p, 0, 2)
    subprob += obj

    subprob.set_constraint_group("3.21")
    for n in range(n_nodes):
        for s in range(n_scenarios):
            for t in range(n_periods):
                p_values = p[Gn[n], s, t].get_var_values() if isinstance(p[Gn[n], s, t], LpVarArray) else p[Gn[n], s, t].varValue
                subprob += (np.sum(p[Gn[n], s, t], axis=0) == np.sum(p_values))
            
    for g in range(n_generators):
        subprob += (np.transpose(p, (2, 0, 1)) <= P_plus * np.transpose(u.get_var_values(), (2, 0, 1)))
        subprob += (P_minus * np.transpose(u.get_var_values(), (2, 0, 1)) <= np.transpose(p, (2, 0, 1)))

    subprob.solve()
    print(subprob.status)

    n_violated, groups_n_violated = subprob.constraints_violated()
    print(n_violated)