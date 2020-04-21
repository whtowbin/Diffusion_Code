#%%
import numpy as np
import scipy.linalg as la
import sympy as sympy
from numba import jit
from matplotlib import pyplot as plt

# %%
'''
Anna testing 123, Great! 
{Ti_m + 2H_si} = [Ti_si] + [2H_m]

K = ([Ti_si][2H_m])/[Ti_m+2H_si]

sum_Ti = [Ti_si] + [Ti_m+2H_si]
sum_H = [2H_m] + [Ti_m+2H_si]


eq = solve(eq2,Ti_Cli)
eq1 = subs(eq1,Ti_Cli, eq)
'''
K = sympy.symbols("K", positive=True)
Ti_Cli, Ti_si, H_m, sum_Ti, sum_H = sympy.symbols("Ti_Cli Ti_si H_m sum_Ti sum_H", positive =True)


#%%
eq1 = (sum_Ti - Ti_si - Ti_Cli)
eq2 = (sum_H - H_m - Ti_Cli)
eqk = (K * Ti_Cli - (Ti_si * H_m))

# %%

# %%

# %%

# %%

# %%
eq = sympy.solve(eq2,Ti_Cli)
eq1_new = eq1.subs({Ti_Cli: eq[0]})
eqk_new = eqk.subs({Ti_Cli: eq[0]})
# %%
eq = sympy.solve(eq1_new, Ti_si)
eqk_new = eqk_new.subs(Ti_si, eq[0])
sympy.solve(eqk_new, H_m)
# %%
f1 = eq1-eq2-eqk
sympy.solve(f1, Ti_Cli)
# %%

x, y, z = sympy.symbols('x y z')
expr = 2*x + y
expr3 = y - 6*x 
expr4 = sympy.solveset(exp3, x )
#exp4 = y/6
expr2 = expr.subs(x, exp4)
expr2

# %%
sympy.solve([eq1, eqk, eq2], [Ti_si, Ti_Cli, sum_H] , dict=True)
# %%
Equations = sympy.nonlinsolve([eq1, eqk, eq2], [Ti_si, Ti_Cli]) 
str(Equations)
# %%


def DH2O_Ol(T_C):
    """
This function returns the diffusivity of H+ cations along the a [100] axis in
olivine at a given temperature in Celcius.
Ferriss et al. 2018 diffusivity in um2/s

Parameters:
T_C: Temperature in degrees C

Returns:
The diffusivity of H+ cations along the a [100] axis of olivine in um^2/S
    """
    T_K = T_C + 273
    DH2O = 1e12 * (10 ** (-5.4)) * np.exp(-130000 / (8.314 * T_K))
    return DH2O

# %%
def VectorMaker(init_Concentration, N_points):
    return init_Concentration * np.ones(N_points)

def diffusion_kernel(Diffusivity, dt, dx):
    delta = (Diffusivity * dt) / ((dx) ** 2)
    kernel = np.zeros(3)
    kernel[1] = 1 - 2 * delta
    kernel[0] = delta
    kernel[2] = delta
    return kernel

def boundary_cond(C= 0):
    # C: single concentration at boundary ( in the future we will support changing boundary conditions)
    pad = np.ones(3) * C
    return pad

def diffusion_step(vector_in, diffusion_kernel, pad):
    # pad = np.ones(3) * Bound_Concentration put back in
    vector = np.concatenate([pad, vector_in, pad])
    return np.convolve(vector, diffusion_kernel, mode="same")[3:-3]
# %%
def time_steper(sum_H, Diffusivity, timesteps, dt=0.5, dx =10 , N_points=100, sum_Ti=100, K=0.8, bound_concentration=0):
    """
Steps a finite element 1D diffusion model forward.

parameters
----------------
v_in: input concentration profile (N x 1) column vector
timesteps : number of timesteps to calculate.
boundaries : Boundary conditions (limited to single value for now)
dt: seconds
dx: microns
sum_H = ppm water in crystal
sum_Ti = ppm Ti in crystal divide by 5 to scale to molar compared to water
Return
--------------
 An updated concentration profile.
    """

  
    H_m2 = -K/2 + sum_H/2 - sum_Ti/2 + np.sqrt(K**2 + 2*K*sum_H + 2*K*sum_Ti + sum_H ** 2 - 2*sum_H*sum_Ti + sum_Ti**2)/2
    Ti_Cli = sum_H - H_m2
    Ti_Si = sum_Ti - Ti_Cli

    H_m_loop = np.ones(N_points) * H_m2
    sum_H_loop = np.ones(N_points) * sum_H
    Ti_Cli_loop = np.ones(N_points) * Ti_Cli
    Ti_Si_loop = np.ones(N_points) *Ti_Si

    # Building diffusion code
    kernel =diffusion_kernel(Diffusivity, dt, dx) 
    bound = boundary_cond(C=bound_concentration)
    

    # Loop to iterate diffusion and reaction
    #for idx, x in enumerate(range(round(timesteps))):
    for x in range(round(timesteps)):
        H_m_loop = diffusion_step(H_m_loop, kernel, bound)
        # Question for Mike: Do we need to calculate all concentrations at each step of the loop or only once to update how H_m changesand then calculate the rest at the end? 
     
        sum_H_loop = Ti_Cli_loop + H_m_loop
        H_m_loop = -K/2 + sum_H_loop/2 - sum_Ti/2 + np.sqrt(K**2 + 2*K*sum_H_loop + 2*K*sum_Ti + sum_H_loop ** 2 - 2*sum_H_loop*sum_Ti + sum_Ti**2)/2
        Ti_Cli_loop = sum_H_loop - H_m_loop
        Ti_Si_loop = sum_Ti - Ti_Cli_loop

        #then update all defects with reaction equations.
        
        # if boundaries is not None:
        # this currently wont work. I need to make it so the boundaries and timesteps are the same.
        # v_loop[0], v_loop[-1] = boundaries[idx], boundaries[idx]
    return {'H_m_loop': H_m_loop, 'Ti_Si_loop': Ti_Si_loop, 'Ti_Cli_loop': Ti_Cli_loop, 'sum_H_loop': sum_H_loop}

# %%

#%%timeit #-r 100

dt = 0.5  # 1 #0.0973 # time step seconds
N_points = 100
profile_length = 1500  # Microns
dx = profile_length / (N_points - 1)  # Microns

#dicts= time_steper(sum_H =100, sum_Ti = 60, K=0.8, Diff_Matrix =B, timesteps = 60*60, N_points= 100, boundaries=None)

dicts = time_steper(sum_H=30, Diffusivity=DH2O_Ol(1200), timesteps=60*60,
                    dt=0.5, dx=10, N_points=100, sum_Ti=25/5, K=1, bound_concentration=0)
 
fig, ax = plt.subplots(figsize=(12,6))
plt.plot(dicts['H_m_loop'], Label = 'H_m')
plt.plot(dicts['Ti_Cli_loop'],Label='Ti_Cli')
plt.plot(dicts['sum_H_loop'], Label='Total_H')
#plt.plot(dicts['Ti_Si_loop'], Label='Ti_Si')


plt.legend(prop={'size': 20})



# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

