import gurobipy as gp
import numpy as np
from solver_for_validation import *
from gurobipy import GRB

params = {
"WLSACCESSID": '730eb8a4-9f5d-46f4-a00a-5a5a124b5cfc',
"WLSSECRET": 'fb659f16-0ba2-4a5d-8740-c85a5ba53c04',
"LICENSEID": 2482509,
}

env = gp.Env(params=params)

I = 2
J = 3
S = 2
L = 1

unit_penalty = unitary_penalty(I, J, L)

shipment_cost = unit_shipment_cost(I, J, L, unit_penalty)

handling_cost = fixed_handling_cost(I, L)

setup_cost = set_up_cost(I, J, L)

demand = demander(J, S, L)

supply_cap = max_supply(I, J, L, S, demand)

def generate_binary_lists(n,I,J, current_list=[], all_lists=[]):
    if n == 0:
        all_lists.append(np.append(np.array(current_list),[1,1,1]).reshape(I+1,J))
    else:
        generate_binary_lists(n - 1,I,J, current_list + [0], all_lists)
        generate_binary_lists(n - 1,I,J, current_list + [1], all_lists)
    return all_lists



all_lists = generate_binary_lists(I*J,I,J)

worst_case = []

for num,y in enumerate(all_lists):
  print(f'Starting iter: {num}\n Setup: {y}')

  for s in range(S):
      print(s)

      #Solve an iteration of the recourse decision with available links y
      model = gp.Model("recourse", env=env)

      # Create variables
      # X_ijs - Flow on a link under scenario s
      x = model.addMVar((S, L, I + 1, J), vtype=GRB.CONTINUOUS, name="x")

      # Z_is - Is supply node activated under scenario s
      z = model.addMVar((S, I + 1, L), vtype=GRB.BINARY, name="z")

      # Define the objective function
      model.setObjective(gp.quicksum(setup_cost[j, i]*y[i][j] for i in range(I+1) for j in range(J))+gp.quicksum(shipment_cost[l, j, i]*x[s, l, i, j] for l in range(L) for i in range(I+1) for j in range(J))+gp.quicksum(handling_cost[l,i]*z[s, i, l] for l in range(L) for i in range(I+1)), GRB.MINIMIZE)

      # Only activated links are used

      for j in range(J):
        for i in range(I + 1):
          for l in range(L):
            model.addConstr((x[s, l, i, j] <= demand[s, j, l] * y[i][j]), name=f'link_activation {s, j, i, l}')

      # Ensures that the demand is satisfied for all s in S

      for j in range(J):
        for l in range(L):
          model.addConstr((gp.quicksum(x[s, l, i, j] for i in range(I + 1)) >= demand[s, j, l]),
                        name=f'demand_satisfaction {s, j, l}')

      # Ensures that the supply cap in each origin is not exceeded for all s in S

      for i in range(I):
        model.addConstr((gp.quicksum(x[s, l, i, j] for j in range(J)) <= supply_cap[s, l, i] * z[s, i, l]),
                      name=f'supply_cap {s, i}')

      # Ensures positive flow for all s in S

      for j in range(J):
        for i in range(I + 1):
          for l in range(L):
            model.addConstr((x[s, l, i, j] >= 0), name=f'positive_flow {s, j, i, l}')

      # Ensures that overall capacity in origins are must be enough to supply the demand for all s in S
      for l in range(L):
        model.addConstr(gp.quicksum(supply_cap[s, l, i] * z[s, i, l] for i in range(I + 1)) >= gp.quicksum(
          demand[s, j, l] for j in range(J)))

      # solving problem
      model.Params.outputFlag = 0
      model.Params.LogToConsole = 0
      model.optimize()
      if model.status == GRB.OPTIMAL:
          print(f"Optimal objective value: {model.objVal}")
          if s > 0:
              worst_case[num] = max(model.objVal, worst_case[num])
          else:
              worst_case.append(model.objVal)

      else:
          print("Optimization failed. Status code: ", model.status)
          break


min_element = min(worst_case)
min_index = worst_case.index(min_element)

print(f'Best worst case cost:\n {min_element}')
print(f'Best worst case setup:\n {all_lists[min_index]}')

alpha = 0.95
solution = optimal_cvar(env,shipment_cost, handling_cost, setup_cost, demand, supply_cap, alpha, S, I, J, L)
print(f'Model solver objective:\n {solution[0].ObjVal}')
print(f'Model solver setup:\n {solution[1]}')

