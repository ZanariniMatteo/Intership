import numpy as np
import pandas as pd
import inspect

def ismale(x):
    if 'Male' in x.unique(): 
        return 1
    else:
        return 0

def is_weekend (x):
    if (x.weekday() >= 5):
        return 1
    else:
        return 0

def meanFreqPurchase(x):
    tempX = sorted(list(x))
    tempVal = []
    if len(tempX) == 1:
        output = 0
    else:
        for i in range(1, len(tempX)):
            tempVal.append((tempX[i]-tempX[i-1]).days)
        output = sum(tempVal)/len(tempVal)
        tempVal = []
    
    return(output)

def get_var_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]