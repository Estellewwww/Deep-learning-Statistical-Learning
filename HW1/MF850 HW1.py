#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:50:27 2023
MF850 Homework1

@author: estelle
"""
import numpy as np
import math
import random
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

#A
r = 0.01
sig = 0.1
T = 1
K = 95
N = 100000
def put(K,S):
    return max(K-S,0)
#xd are constant and only x0 is needed
u = np.exp(sig*np.sqrt(1/1))
d = 1/u
q = (math.exp(r/1)-d)/(u-d)
X_delta_t = 100
Su = u*X_delta_t
Sd = d*X_delta_t
vt = []
xt = np.zeros(N)
for i in range(N):
    St = random.choices([Su,Sd], weights=[q,1-q], k=1)[0]
    vt.append(put(K,St))
model = sm.OLS(vt,xt).fit()
tvr = model.params[0]*(math.exp(-r))
ls_list = []
for i in range(100000):
    if 0 > tvr:
        ls_list.append(0)
    else:
        ls_list.append(vt[i]*(math.exp(-r)))
ls = np.mean(ls_list)
print("The result for TVR is ", tvr)
print("The result for  LS is ", ls)

#B
def TVR(n,p):
    u = math.exp(sig*math.sqrt(1/n))
    d = 1/u
    q = (math.exp(r/n)-d)/(u-d)
    outcomes = [u, d]
    weights = [q, 1-q]
    sim_path = []
    for i in range(100000):
        path = [100]
        for j in range(n):
            random_result = random.choices(outcomes, weights=weights, k=1)[0]
            path.append(path[-1]*random_result)
        sim_path.append(path)
    v_last = []
    x_last = []
    for path in sim_path:
        v_last.append(max(95-path[-1], 0))
        p_var = []
        for k in range(p+1):
            p_var.append(path[-2]**k)
        x_last.append(p_var)
    model = sm.OLS(v_last, x_last).fit()
    v_hat = list(model.fittedvalues)
    v_now = []
    for m in range(100000):
        v_now.append(max(95-sim_path[m][-2], math.exp(-r/n)*v_hat[m]))
    # Do fitting backward
    for a in range(n-2, 0, -1):
        x_now = []
        for m in range(100000):
            p_var = []
            for k in range(p+1):
                p_var.append(sim_path[m][a]**k)
            x_now.append(p_var)
        model_now = sm.OLS(v_now, x_now).fit()
        v_hat = list(model_now.fittedvalues)
        v_now = []
        for m in range(100000):
            v_now.append(max(95-sim_path[m][a], math.exp(-r/n)*v_hat[m]))
    x_now = [1]*100000
    model_now = sm.OLS(v_now, x_now).fit()
    return math.exp(-r/n)*model_now.params[0]

def LS(n,p):
    u = math.e**(sig*math.sqrt(1/n))
    d = 1/u
    q = (math.e**(r/n)-d)/(u-d)
    outcomes = [u, d]
    weights = [q, 1-q]
    sim_path = []
    # simulate price path for 100000 times
    for i in range(100000):
        path = [100]
        for j in range(n):
            random_result = random.choices(outcomes, weights=weights, k=1)[0]
            path.append(path[-1]*random_result)
        sim_path.append(path)
    # fit the last values
    v_last = []
    x_last = []
    for path in sim_path:
        v_last.append(max(95-path[-1], 0))
        p_var = []
        for k in range(p+1):
            p_var.append(path[-2]**k)
        x_last.append(p_var)
    model = sm.OLS(v_last, x_last).fit()
    v_hat = list(model.fittedvalues)
    v_now = []
    for m in range(100000):
        if 95-sim_path[m][-2] > math.exp(-r/n)*v_hat[m]:
            v_now.append(95-sim_path[m][-2])
        else:
            v_now.append(math.exp(-r/n)*v_last[m])
    # Do fitting backward
    for a in range(n-2, 0, -1):
        v_last = v_now.copy()
        x_now = []
        for path in sim_path:
            p_var = []
            for k in range(p+1):
                p_var.append(path[a]**k)
            x_now.append(p_var)
        model_now = sm.OLS(v_now, x_now).fit()
        v_hat = list(model_now.fittedvalues)
        v_now = []
        for m in range(100000):
            if 95-sim_path[m][a] > math.exp(-r/n)*v_hat[m]:
                v_now.append(95-sim_path[m][a])
            else:
                v_now.append(math.exp(-r/n)*v_last[m])
    v_last = v_now.copy()
    x_now = [1]*100000
    model_now = sm.OLS(v_now, x_now).fit()
    v_first = []
    for m in range(100000):
        if 95-100 > math.exp(-r/n)*model_now.params[0]:
            v_first.append(95-100)
        else:
            v_first.append(math.exp(-r/n)*v_last[m])
    return np.mean(v_first)

"""fix p=2
"""
print("The result for TVR when n=100 is ", TVR(100, 2))
print("The result for  LS when n=100 is ", LS(100, 2))


# C

def LS_modified(n, p, sigma, r):
    u = math.exp(sigma * math.sqrt(1/n))
    d = 1/u
    q = (math.exp(r/n) - d) / (u - d)
    thres = int(math.log(0.95) / math.log(d))
    outcomes = [u, d]
    weights = [q, 1-q]
    sim_path = []

    for i in range(100000):
        path = [100]
        for j in range(n):
            random_result = random.choices(outcomes, weights=weights, k=1)[0]
            path.append(path[-1] * random_result)
        sim_path.append(path)

    v_last = []
    v_modified = []
    x_modified = []
    x_last = []

    for path in sim_path:
        p_var = [path[-2] ** k for k in range(p+1)]
        v_modified.append(max(95 - path[-1], 0))
        x_modified.append(p_var)
        v_last.append(max(95 - path[-1], 0))
        x_last.append(p_var)

    model = sm.OLS(v_modified, x_modified).fit()
    v_hat = [max(0, val) for val in model.predict(x_last)]

    v_now = [95 - sim_path[m][-2] if 95 - sim_path[m][-2] > math.exp(-r/n) * v_hat[m] else math.exp(-r/n) * v_last[m] for m in range(100000)]

    for a in range(n-2, thres, -1):
        v_last = v_now.copy()
        x_now = []
        v_modified = []
        x_modified = []

        for m in range(100000):
            p_var = [sim_path[m][a] ** k for k in range(p+1)]
            v_modified.append(max(95 - sim_path[m][a], v_last[m]))
            x_modified.append(p_var)
            x_now.append(p_var)

        model_now = sm.OLS(v_modified, x_modified).fit()
        v_hat = [max(0, val) for val in model_now.predict(x_now)]

        v_now = [95 - sim_path[m][a] if 95 - sim_path[m][a] > math.exp(-r/n) * v_hat[m] else math.exp(-r/n) * v_last[m] for m in range(100000)]

    return np.mean(v_now) * math.exp(-r * (thres + 1) / n)

print("The result for  LS_modified when n=100 is ", LS_modified(100, 2, 0.1, 0.01))
    
    
    
        
    