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
N_points = 100
profile_length = 1500 # Microns
dX = profile_length / (N_points -1) # Microns

Distances = [0+dX*Y - profile_length/2 for Y in range(N_points)]


max_time_step = DH2O**2 / (4*dX)
max_time_step
dt = 0.5 #1 #0.0973 # time step seconds

boundary = 0 #ppm
initial_C = 10 # ppm

v = np.mat(np.ones(N_points)* initial_C).T
v[0], v[-1] = boundary, boundary
v_initial = v

# %%
# this is a term that take all the multiplication terms at each time step of the finite difference model into one term
# In the future I can update this every timestep if temperature changes with time.
# also update boundary conditions with time

def diffusion_matrix(DH2O, dt, dx):
    delta = (DH2O * dt)/ (dx**2)

    mat_base = np.zeros(N_points)
    mat_base[0] = 1 - 2*delta
    mat_base[1] = delta

    B = np.mat(la.toeplitz((mat_base)))
    B[0,0] , B[-1,-1]  = 1, 1
    B[0,1] , B[-1,-2] = 0 , 0
    return B

# %%

def time_steper(v_in, Diff_Matrix, timesteps, boundaries = None):
    """
Steps a finite element 1D diffusion model forward.

parameters
----------------
v_in: input concentration profile (N x 1) column vector
timesteps : number of timesteps to calculate.
boundaries : Boundary conditions

Return
--------------
 An updated concentration profile.
 """

    v_loop = v_in
    for idx, x in enumerate(range(timesteps)):
        v_loop = Diff_Matrix * v_loop
        if boundaries is not None:
            # this currently wont work. I need to make it so the boundaries and timesteps are the same.
            v_loop[0], v_loop[-1] = boundaries[idx]
    return v_loop


B = diffusion_matrix(DH2O,dt,dX)

fig, ax = plt.subplots(figsize=(12,8))
plt.plot(Distances, time_steper(v, B, 0) )
plt.plot(Distances, time_steper(v, B, 10))
plt.plot(Distances, time_steper(v, B, 100))
plt.plot(Distances, time_steper(v, B, 1000))
plt.plot(Distances, time_steper(v, B, 10000))

ax.set_xlabel("Distance to center of crystal ")
ax.set_ylabel("ppm water")

#plt.savefig(" ")

D_OPX = 1e12*(10**(-3))*np.exp(-181000/(8.314*T_K))  * 10
D_CPX = 1e12*(10**(-3))*np.exp(-181000/(8.314*T_K))  * 100





# %%
# Max water in ol at depth
Intial_low = 2.7 * 0.0007 * 10000
Initial_high = 2.7 * 0.0015 * 10000
Final = 10
