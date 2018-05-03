from utils import *
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import heuristics


X = lp_array("X", (5, 5), "Continuous", up_bound=80)
Y = lp_array("Y", (5, 5), "Continuous", up_bound=500)

prob = SUCLpProblem("prob", pulp.LpMaximize)
prob += 3 * X + 4 * Y

prob += (X - 2 * Y <= 7)
prob += (-8*X + 3*Y == 15)

prob.solve()

variables = prob.get_variables()
int_mask = [1] * len(variables)
solution = variables.get_var_values()
constraints = prob.get_constraints_as_tuples()



rounded = np.round(solution)
while True:
    prob.set_var_values(rounded)
    n_violated, _ = prob.constraints_violated()
    print(len(n_violated))
    if n_violated == 0:
        break
    




prob.set_var_values(rounded)
n_violated, groups_n_violated = prob.constraints_violated()
print("Number of violated constraints: %i" % n_violated)
for group in groups_n_violated.keys():
    print("Number of violated constraints of group %s: %i / %i" % (
        group, groups_n_violated[group][0], groups_n_violated[group][1]))
if prob.is_integer_solution() and not prob.constraints_violated():
    print("Found integer MIP solution.")
    print("Value of the objective: %f" % prob.objective.value())