# Stochastic Unit Commitment

The formulations and dual optimization algorithm mainly draw on the work of Anthony Papavasiliou.


## Solve primal problem
  
  $ python main.py <path_to_instance>
  
## Solve linear relaxation

  $ python main.py <path_to_instance> --relax

## Solve linear relaxation + rounding algorithm
  
  $ python main.py <path_to_instance> --relax --round

## Lagrangian decomposition and subgradient optimization

  $ python main.py <path_to_instance> --decompose
  $ python main.py <path_to_instance> --decompose --nar 6 --epsilon 0.01 --alpha0 2000 --rho 0.96


The parameters for the subgradient algorithm are the following:
    - nar: number of iterations of subgradient algorithm to make before to start applying heuristics to recover primal solutions and upper bounds.
    - epsilon: convergence threshold / duality gap under which the subgradient algorithm is considered to have converged. When $(UB - LB) / UB  < \epsilon$, the algorithm stops.
    - alpha0: Initial steplength for updating Lagrange multipliers.
    - rho: discount factor of the steplength (if no feasible primal solution has been found yet).
