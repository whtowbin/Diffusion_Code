# %%
import numpy as np
import scipy.linalg as la
import sympy as sympy
from matplotlib import pyplot as plt


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

def DH2O_GB(T_C):
    '''
    This function returns the diffusivity of H + cations along grain boundaries
    This should be longer in terms of 
    ''' 
    T_K = T_C + 273
    #### Get from Demouchy DH2O = 1e12 * (10 ** (-5.4)) * np.exp(-130000 / (8.314 * T_K))
    #return DH2O
    pass

# %%

def VectorMaker(init_Concentration, N_points):
    return init_Concentration * np.ones(N_points)


def Multi_vector_Maker(init_Concentration, N_points, kd = 1, N_points_grain=50):
    return init_Concentration * kd * np.ones(N_points)

def diffusion_kernel(Diffusivity, dt, dx):
    delta = (Diffusivity * dt) / ((dx) ** 2)
    kernel = np.zeros(3)
    kernel[1] = 1 - 2 * delta
    kernel[0] = delta
    kernel[2] = delta
    return kernel


def boundary_cond(C=0):
    # C: single concentration at boundary ( in the future we will support changing boundary conditions)
    pad = np.ones(3) * C
    return pad


def diffusion_step(vector_in, diffusion_kernel, pad):
    # pad = np.ones(3) * Bound_Concentration put back in
    vector = np.concatenate([pad, vector_in, pad])
    return np.convolve(vector, diffusion_kernel, mode="same")[3:-3]

# %%



"""
Partition water between Grain Boundary and Olivine
kd= H2O_ol/H2O_GB

kd= H2O_GB/H2O_Magma  # Maybe not needed yet but would be good for degassing 

Sum_H = H2O_ol + H2O_GB
"""
y= VectorMaker(init_Concentration=1, N_points=(100))
f= VectorMaker(init_Concentration=0, N_points=(100, 10))

f[:, 0] = f[:, 0] + y
f[:, 1] = f[:, 1] + f[:, 0]
f[:, 2] = f[:, 2] + f[:, 1]
f
