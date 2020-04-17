#%%
import numpy as np
import scipy.linalg as la
import sympy as sympy
# %%
'''
{Ti_m + 2H_si} = [Ti_si] + [2H_m]

K = ([Ti_si][2H_m])/[Ti_m+2H_si]

sum_Ti = [Ti_si] + [Ti_m+2H_si]
sum_H = [2H_m] + [Ti_m+2H_si]


eq = solve(eq2,Ti_Cli)
eq1 = subs(eq1,Ti_Cli, eq)
'''
K, Ti_Cli, Ti_si, H_m, sum_Ti, sum_H = sympy.symbols("K Ti_Cli Ti_si H_m sum_Ti sum_H")
#%%
eq1 = (sum_Ti - Ti_si - Ti_Cli)
eq2 = (sum_H - H_m - Ti_Cli)
eqk = (K * Ti_Cli - (Ti_si * H_m))

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
expr2
# %%
type(expr4)
type(expr2)
# %%
sympy.solve([eq1, eqk, eq2], [Ti_si, Ti_Cli, sum_H] , dict=True)
# %%
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

def time_steper(Ti_si, H_m, Ti_Cli, Diff_Matrix, timesteps, boundaries=None):
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
    K = 0.8 # First approx from some abs ratios 
    sum_Ti = 400 # We should estimate this based on ppm divided by molar mass compared to H/molar-mass 
    # H_m K and Sum_Ti should be the inputs 

    #Ti_si_loop = Ti_si
    #H_m_loop = H_m
    #Ti_Cli_loop = Ti_Cli
    Ti_si_loop = K*sum_Ti/(H_m_loop + K)
    Ti_Cli_loop = H_m_loop * sum_Ti/(H_m_loop + K)
    sum_H_loop = H_m_loop*(H_m_loop + K + sum_Ti)/(H_m_loop + K)

    for idx, x in enumerate(range(round(timesteps))):
        H_m_loop = Diff_Matrix * H_m_loop

        Ti_si_loop = K*sum_Ti/(H_m_loop + K)
        Ti_Cli_loop = H_m_loop * sum_Ti/(H_m_loop + K)
        sum_H_loop = H_m_loop*(H_m_loop + K + sum_Ti)/(H_m_loop + K)

        H_m_loop = (sum_H_loop - Ti_Cli)


        #then update all defects with reaction equations.
        
        # if boundaries is not None:
        # this currently wont work. I need to make it so the boundaries and timesteps are the same.
        # v_loop[0], v_loop[-1] = boundaries[idx], boundaries[idx]
    return v_loop


B = diffusion_matrix(.000000001, 1, .0001)



