import numpy as np
import scipy.linalg as la
import pandas as pd
from matplotlib import pyplot as plt

# %%
# I am writing a code that will  support diffusion modeling

T_C = 1200;
T_K = T_C+273;
DH2O = 1e12*(10**(-5.4))*np.exp(-130000/(8.314*T_K))  # Ferriss diffusivity in um2/s
# These are critical for stability of model. Must think about time steps and model points.
N_points = 10
profile_length = 1500 # Microns
dX = profile_length / (N_points -1) # Microns



max_time_step = DH2O**2 / (4*dX)
max_time_step
dt = 1 #0.0973 # time step seconds

boundary = 0 #ppm
initial_C = 10 # ppm

v = np.mat(np.ones(N_points)* initial_C).T
v[0], v[-1] = boundary, boundary
v_initial = v

# %%
# this is a term that take all the multiplication terms at each time step of the finite difference model into one term
# In the future I can update this every timestep if temperature changes with time.
# also update boundary conditions with time
delta = (DH2O * dt)/ (dX **2)
delta
mat_base = np.zeros(N_points)
mat_base[0] = 1 - 2*delta
mat_base[1] = delta

B = np.mat(la.toeplitz((mat_base)))
B[0,0] , B[-1,-1]  = 1, 1
B[0,1] , B[-1,-2] = 0 , 0

# %%
def time_steper(v_in, timesteps, boundaries):
"""
Steps a finite element 1D diffusion model forward.
parameters:
v_in: input concentration profile (N x 1) column vector
timesteps : number of timesteps to calculate.
boundaries : Boundary conditions

"""
    v_loop = v_in
    for x in range(timesteps):
        v_loop = B * v_loop
    return v_loop

 # Estimate how long olivine took to reach present concnetration. Use boundary condition at depthself.
 # put distance on points. Make center zero and then measure distance too rims

 # And calcuate boundary condition for different partitioning values.
 # Cacluate length of Stalling ... 0.5 - 2 hours...
 # Caculate time for ascent ... ascent rate and change in pressure. from storage depth to surface. + Surface cooling ~ 10 minutes.

 # give min time range for olivine water loss since ascent started.

 # Cacluate time for CPX to lose water
 # Measure Profile for Opx too.

# Conclude that if CPX and Opx can lose water in this time scale they very likely equilibrated with the magma during ascent.

plt.plot(time_steper(v,0))
plt.plot(time_steper(v,10))
plt.plot(time_steper(v,100))
plt.plot(time_steper(v,1000))
plt.plot(time_steper(v,5000))
