import numpy as np
import scipy.linalg as la


def diffusion_matrix(DH2O, dt, dx, N_points):
"""
    Makes a matrix that can be used for a finite diference diffusion numerical
    model.

    parameters
    ----------------
    DH2O: Diffusivity
    dt: time step
    dx: distance between x N_points
    N_points: number of points in your profile

    Return
    --------------
     A matrix that can be used to multiply by a concentration profile to
     make diffusion code 1 timestep. 
"""
    delta = (DH2O * dt) / (dx ** 2)

    mat_base = np.zeros(N_points)
    mat_base[0] = 1 - 2 * delta
    mat_base[1] = delta

    B = np.mat(la.toeplitz((mat_base)))
    B[0, 0], B[-1, -1] = 1, 1 # For fixed boundary condition
    B[0, 1], B[-1, -2] = 0, 0  # For fixed boundary condition
    return B

def time_steper(v_in, Diff_Matrix, timesteps, boundaries=None):
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
    for idx, x in enumerate(range(round(timesteps))):
        v_loop = Diff_Matrix * v_loop
        # if boundaries is not None:
        # this currently wont work. I need to make it so the boundaries and timesteps are the same.
        # v_loop[0], v_loop[-1] = boundaries[idx], boundaries[idx]
    return v_loop


B = diffusion_matrix(DH2O, dt, dX)
