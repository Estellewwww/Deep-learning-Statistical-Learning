#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MF850 HW2
Sike Yang

Created on Tue Oct  3 16:40:07 2023

@author: estelle
"""
import pandas as pd
import numpy as np
rt = pd.read_csv("/Users/estelle/Desktop/MF850/returns.csv")
#a
rt["log_rt"] = np.log(1+rt["return"])
alpha = 0.05
ret_quantile = rt.log_rt.quantile(alpha, interpolation="higher")
var_emp = -(np.exp(ret_quantile)-1)
print(var_emp)
#0.11749633800000003

#b
from scipy.stats import norm
n = len(rt["return"])
ret_mean = rt.log_rt.mean()
ret_std = np.sqrt(rt.log_rt.var()*n/(n-1))
var_thry = 1-np.exp(ret_mean+ret_std*norm.ppf(alpha))
print(var_thry)
#0.10642762058430388

#c
w=100
var_w = -(np.exp(ret_quantile)-1)*w
print(var_w)
#11.749633800000003

#d
from scipy.stats import bootstrap
series = ((np.exp(rt["log_rt"])-1)*w,)
bootstrap_var = bootstrap(series, np.std, confidence_level=0.95,
                         random_state=1, method='percentile')
print(bootstrap_var.confidence_interval)
#ConfidenceInterval(low=8.885532905940536, high=10.364017174039901)
















