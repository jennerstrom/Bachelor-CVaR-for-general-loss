import gurobipy as gp
import scipy as sp
import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt

params = {
"WLSACCESSID": '730eb8a4-9f5d-46f4-a00a-5a5a124b5cfc',
"WLSSECRET": 'fb659f16-0ba2-4a5d-8740-c85a5ba53c04',
"LICENSEID": 2482509,
}
env = gp.Env(params=params)


###
### DATA GENERATION
###

# We define functions that generate the data for the model, the logic of these are taken from the Y. Hinosha article

def unit_shipment_cost(I,J,L,p):
    d = []
    for l in range(L):
        for j in range(J):
            d = np.append(d,np.random.uniform(10, 100, I))
            d = np.append(d, p[j,l])
    # save this data in a LxIxJ array
    c = d.reshape(L,J,I+1)
    return c

def fixed_handling_cost(I,L):
    h = []
    for l in range(L):
        h = np.append(h,np.random.uniform(500,3500, I))
        h = np.append(h,0)
    # save this data in a LxI array
    f = h.reshape(L,I+1)
    return f

def set_up_cost (I,J,L):
    d = []
    dt = J*L*680/3
    center = 55*dt/(I*J)
    for j in range(J):
        d = np.append(d, np.random.uniform(center*3/4,center*5/4, I))
        d = np.append(d, 0)

    c = d.reshape(J, I+1)
    return c

def S1_creator(J):
    return np.random.uniform(10,50,J)

def S2_creator(J):
    return np.random.uniform(100,200,J)

def S3_creator(J):
    return np.random.uniform(400,600,J)

def S4_creator(intervals, J):
    h = []
    for j in range(0,J):
        # Choose a random interval
        interval = random.choice(intervals)
        # Generate a random number within the chosen interval
        h.append(random.uniform(interval[0], interval[1]))
    return h

def split_int(n):
    a = math.ceil(n/2)
    b = math.ceil((n-a)/2)
    c = math.ceil((n-a-b)/2)
    d = math.floor((n-a-b)/2)
    return [d,c,b,a]

def demander(J,S,L):
    scenarios = split_int(S)
    x = np.array([])
    intervals = [(0,50),(100,200),(400,600)]
    for l in range(L):
        for s in range(0,scenarios[0]):
            x = np.append(x, S1_creator(J))
        for s in range(0,scenarios[1]):
            x = np.append(x, S2_creator(J))
        for s in range(0,scenarios[2]):
            x = np.append(x, S3_creator(J))
        for s in range(0,scenarios[3]):
            x = np.append(x, S4_creator(intervals,J))
    d = x.reshape(S,J,L)
    return d

def max_supply (I,J,L,S,d):
    n = I*L
    dt = J*L*680/3
    center = dt/(I*L)

    x = np.random.uniform(center*4/5,center*6/5,n)

    c = x.reshape(L,I)

    # Calculating the surplus demand for each commodity in each scenario
    diff = np.array([])
    for l in range(L):
        for s in range(S):
            diff = np.append(diff, sum(d[s, :, l]) - sum(c[l, :]))
    diff = diff.reshape(L,S)

    # making sure we only have positive values
    x = []
    for l in range(L):
        for s in range(S):
            x = np.append(x, max(diff[l,s], 0))
    x = x.reshape(L,S)

    # for each scenario we make the LxI+1 matrix with the appropriate supply cap
    k = []
    for s in range(S):
        for l in range(L):
            k = np.append(k, c[l,:])
            k = np.append(k, x[l,s])
    k = np.array(k)
    k = k.reshape(S, L, I+1)

    return k

def unitary_penalty (I,J,L):
    n = J*L
    dt = J*L*680/3
    alpha = 55*dt/(I*J)
    beta = 100 + 3*(3500+1.25*alpha)/680

    d = np.random.uniform(beta*0.9,beta*1.1,n)

    c = d.reshape(J,L)
    return c

###
### IMPLEMENTING THE MODEL
###


from gurobipy import GRB

# Create a new model
def optimal_cvar (shipment_cost,handling_cost,setup_cost,demand,supply_cap,alpha,combi,S,I,J,L):
    C = len(combi)

    model = gp.Model("cvar_tp", env = env)

    # Create variables
    #VaR
    eta = model.addMVar(shape =(C,), name = "VaR", vtype = GRB.CONTINUOUS)

    #PSI
    psi = model.addMVar(shape = (S,C), name = "psi", vtype = GRB.CONTINUOUS)

    # Y_ij - Is link (i,j) used in the realization
    y = model.addMVar((I+1,J), vtype = GRB.BINARY, name = "y")

    # X_ijs - Flow on a link under scenario s
    x = model.addMVar((S,L,I+1,J), vtype = GRB.CONTINUOUS, name = "x")

    # Z_is - Is supply node activated under scenario s
    z = model.addMVar((S,I+1,L), vtype = GRB.BINARY, name = "z")


    # Define the objective function
    model.setObjective(gp.quicksum(combi[c]*(eta[c] + (1/(1-alpha[c]))*gp.quicksum(psi[s,c]/S for s in range(S))) for c in range(C)), GRB.MINIMIZE)

    #Define constraints
    for c in range(C):
        for s in range(S):
          model.addConstr(psi[s,c] >= gp.quicksum(setup_cost[j, i]*y[i, j] for i in range(I+1) for j in range(J))+
                         gp.quicksum(shipment_cost[l, j, i]*x[s, l, i, j] for l in range(L) for i in range(I+1) for j in range(J))+
                         gp.quicksum(handling_cost[l,i]*z[s, i, l] for l in range(L) for i in range(I+1))-eta[c], name = f'psi {s}')



    # Define a constraint that ensures psi greater than 0 for all s in S
    for s in range(S):
        for c in range(C):
            model.addConstr((psi[s] >= 0), name = f'psi_positive {s}')

    # Only activated links are used
    for s in range(S):
      for j in range(J):
        for i in range(I+1):
            for l in range(L):
                model.addConstr((x[s, l, i, j] <= demand[s, j, l]*y[i, j]), name = f'link_activation {s,j,i,l}')

    # Ensures that the demand is satisfied for all s in S
    for s in range(S):
      for j in range(J):
          for l in range(L):
            model.addConstr((gp.quicksum(x[s, l, i, j] for i in range(I+1)) >= demand[s,j,l]), name=f'demand_satisfaction {s,j,l}')

    # Ensures that the supply cap in each origin is not exceeded for all s in S
    for s in range(S):
      for i in range(I):
        model.addConstr((gp.quicksum(x[s, l, i, j] for j in range(J)) <= supply_cap[s, l, i]*z[s, i, l]), name = f'supply_cap {s,i}')

    # Ensures positive flow for all s in S
    for s in range(S):
      for j in range(J):
        for i in range(I+1):
            for l in range(L):
                model.addConstr((x[s, l, i, j] >= 0), name = f'positive_flow {s,j,i,l}')

    # Ensures that overall capacity in origins are must be enough to supply the demand for all s in S
    for s in range(S):
        for l in range(L):
            model.addConstr(gp.quicksum(supply_cap[s,l,i]*z[s, i, l] for i in range(I+1)) >= gp.quicksum(demand[s,j,l] for j in range(J)))

    # solving problem
    model.Params.outputFlag = 0
    model.Params.LogToConsole = 0
    model.optimize()
    return [model, y.X, x.X, z.X]

# Set parameters
I = 5
J = 7
S = 100
L = 3
alpha = [0.5,0.75, 0.95]
combi = [0.2, 0.4, 0.4]

# Here we can solve the model for a single set of parameters to verify that the model works i.e. the solution seems
# reasonable and the model is feasible

## Generate data
unit_penalty = unitary_penalty(I, J, L)

shipment_cost = unit_shipment_cost(I, J, L, unit_penalty)

handling_cost = fixed_handling_cost(I, L)

setup_cost = set_up_cost(I, J, L)

demand = demander(J, S, L)

supply_cap = max_supply(I, J, L, S, demand)

## Solve model
solution = optimal_cvar(shipment_cost,handling_cost,setup_cost,demand,supply_cap,alpha,combi,S,I,J,L)
print("first model solved")


def cost_calculator (s, setup_cost, shipment_cost, handling_cost, y, x, z):
    cost = (gp.quicksum(setup_cost[j, i] * y[i, j] for i in range(I + 1) for j in range(J)) + gp.quicksum(shipment_cost[l, j, i] * x[s, l, i, j] for l in range(L) for i in range(I + 1) for j in range(J)) +
     gp.quicksum(handling_cost[l, i] * z[s, i, l] for l in range(L) for i in range(I + 1)))
    return cost.getValue()


def cost_distribution(S, setup_cost, shipment_cost, handling_cost, y, x, z):
    c = np.array([])
    for s in range(S):
        c = np.append(c, cost_calculator(s, setup_cost, shipment_cost, handling_cost, y, x, z))
    return c

# Plot the cost distribution as a histogram
def CVaR_calc(cost_dis, alpha):
    k_alpha = 0
    summ = 0
    while summ < alpha:
        summ = summ + 1/S
        k_alpha = k_alpha+1

    count = np.sort(cost_dis)

    cvar = 1/(1-alpha)*(np.sum(count[k_alpha:] * 1/S)+(k_alpha/S-alpha)*count[k_alpha])

    return cvar


def histogram_plot (S, setup_cost, shipment_cost, handling_cost, solution,alpha):
    cost = cost_distribution(S, setup_cost, shipment_cost, handling_cost, solution[1], solution[2], solution[3])
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    worst_case = np.quantile(cost, 1 - 1 / S)
    cvar05 = CVaR_calc(cost, 0.5)
    cvar075 = CVaR_calc(cost, 0.75)
    cvar095 = CVaR_calc(cost, 0.95)

    my_text = '- Estimated quantities -\n'
    my_text += fr'$Worst case=${worst_case:.3f}' + '\n' + fr'$CVaR(0.5))=${cvar05:.3f}' + '\n' + fr'$CVaR(0.75)=${cvar075:.3f}' + '\n' + fr'$CVaR(0.95)=${cvar095:.3f}'

    ax.hist(cost, edgecolor='black', bins=20, alpha=0.7, color='blue')
    props = dict(boxstyle='round', facecolor='grey', alpha=0.15)  # bbox features
    ax.text(1.03, 0.98, my_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    plt.tight_layout()
    if type(alpha) == list:
        plt.title(
            f'Cost distribution for alpha = {alpha[0]}, {alpha[1]}, {alpha[2]}, weights = {combi[0]}, {combi[1]}, {combi[2]}, objective value = {solution[0].objVal}')
    else:
        plt.title(f'Cost distribution for alpha = {alpha}')

    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.subplots_adjust(top=0.9,left=0.1,bottom=0.1)
    plt.show()


solution_alpha0 = optimal_cvar(shipment_cost,handling_cost,setup_cost,demand,supply_cap,[0.5],[1],S,I,J,L)
print("second model solved")

solution_alpha1 = optimal_cvar(shipment_cost,handling_cost,setup_cost,demand,supply_cap,[0.75],[1],S,I,J,L)
print("third model solved")

solution_alpha2 = optimal_cvar(shipment_cost,handling_cost,setup_cost,demand,supply_cap,[0.95],[1],S,I,J,L)
print("fourth model solved")

histogram_plot(S, setup_cost, shipment_cost, handling_cost, solution, alpha)

histogram_plot(S, setup_cost, shipment_cost, handling_cost, solution_alpha0, 0.5)

histogram_plot(S, setup_cost, shipment_cost, handling_cost, solution_alpha1, 0.75)

histogram_plot(S, setup_cost, shipment_cost, handling_cost, solution_alpha2, 0.95)
