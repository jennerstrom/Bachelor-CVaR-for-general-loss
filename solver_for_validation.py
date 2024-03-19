import gurobipy as gp
import scipy as sp
import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt

#params = {
#"WLSACCESSID": '730eb8a4-9f5d-46f4-a00a-5a5a124b5cfc',
#"WLSSECRET": 'fb659f16-0ba2-4a5d-8740-c85a5ba53c04',
#"LICENSEID": 2482509,
#}
#
#env = gp.Env(params=params)

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
def optimal_cvar (env,shipment_cost,handling_cost,setup_cost,demand,supply_cap,alpha,S,I,J,L):

    model = gp.Model("cvar_tp", env = env)

    # Create variables
    #VaR
    eta = model.addVar(name = "VaR", vtype = GRB.CONTINUOUS)

    #PSI
    psi = model.addMVar(shape = (S,), name = "psi", vtype = GRB.CONTINUOUS)

    # Y_ij - Is link (i,j) used in the realization
    y = model.addMVar((I+1,J), vtype = GRB.BINARY, name = "y")

    # X_ijs - Flow on a link under scenario s
    x = model.addMVar((S,L,I+1,J), vtype = GRB.CONTINUOUS, name = "x")

    # Z_is - Is supply node activated under scenario s
    z = model.addMVar((S,I+1,L), vtype = GRB.BINARY, name = "z")


    # Define the objective function
    model.setObjective(eta + (1/(1-alpha))*gp.quicksum(psi[s]/S for s in range(S)), GRB.MINIMIZE)

    #Define constraints
    for s in range(S):
      model.addConstr(psi[s] >= gp.quicksum(setup_cost[j, i]*y[i, j] for i in range(I+1) for j in range(J))+
                     gp.quicksum(shipment_cost[l, j, i]*x[s, l, i, j] for l in range(L) for i in range(I+1) for j in range(J))+
                     gp.quicksum(handling_cost[l,i]*z[s, i, l] for l in range(L) for i in range(I+1))-eta, name = f'psi {s}')

    # Define a constraint that ensures psi greater than 0 for all s in S
    for s in range(S):
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




