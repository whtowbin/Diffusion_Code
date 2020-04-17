# %%

import numpy as np
import scipy.linalg as la
import sympy as sympy
from numba import jit
from matplotlib import pyplot as plt

# %%

# %%
'''

{Ti_m + 2H_si} + {2Fe3_m + V_m} = Ti_si + 2Fe2_m + 2H_m

K = (H_m^2 * Fe2_m^2 * Ti_si)/(Ti_Cli * {2Fe3_m + V_m})

sum_Ti = Ti_Cli + Ti_si
sum_H = H_m + Ti_Cli + {2Fe3_m + V_m} This is wrong it should be sum_H = H_m + Ti_Cli
sum_Fe = Fe2_m + {2Fe3_m + V_m}


#_________________________________________ Mg-Site reaction
{Ti_m + 2H_si} = [Ti_si] + [2H_m]

K = ([Ti_si][2H_m])/[Ti_m+2H_si]

sum_Ti = [Ti_si] + [Ti_m+2H_si]
sum_H = [2H_m] + [Ti_m+2H_si]


eq = solve(eq2,Ti_Cli)
eq1 = subs(eq1,Ti_Cli, eq)
'''
K = sympy.symbols("K", positive = True, real=True)
Ti_Cli, Ti_si, H_m, sum_Ti, sum_H, Fe3V, sumFe, Fe2_m = sympy.symbols(
    "Ti_Cli Ti_si H_m sum_Ti sum_H Fe3V sumFe Fe2_m", positive=True, real = True)


# %%

# %%

# %%

# %%

# %%

# %%

#%%
eq1 = (sum_Ti - Ti_si - Ti_Cli)
eq2 = (sum_H - H_m - Ti_Cli ) #- Fe3V)
eq3 = (sumFe - Fe2_m - Fe3V)
# I am wondering if there are other reactions we could write to help combine some of these other equations
#K = (H_m ^ 2 * Fe2_m ^ 2 * Ti_si)/(Ti_Cli * {2Fe3_m + V_m})
eqk = sympy.Eq((H_m ** 2 * Fe2_m ** 2 * Ti_si)/(Ti_Cli * Fe3V),K)
# %%

# %%
# %%
sympy.solve([eq1, eq2, eq3, eqk], [Ti_si, Ti_Cli, Fe3V, Fe2_m], exclude = [sum_H, sum_Ti, sumFe, K], dict = True) 

# %% 
sympy.nonlinsolve([eq1, eq2, eq3, eqk], [Ti_si, Ti_Cli, Fe3V, Fe2_m])
# %%

# %%

# %%

# %%

# %%
eq = sympy.solve(eq1_new, Ti_si)
eqk_new = eqk_new.subs(Ti_si, eq[0])
sympy.solve(eqk_new, H_m)

# this returns an empty list. I am not sure why?
sympy.nonlinsolve([eq1, eqk, eq2], [Ti_si, Ti_Cli])

