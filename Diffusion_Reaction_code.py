#%%
import numpy as np
import scipy.linalg as la
import sympy as sympy
from numba import jit
from matplotlib import pyplot as plt

# %%
'''
Anna testing 123
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

# %%
@jit(nopython=True)
def time_steper(H_mvector, Diff_Matrix, timesteps, sum_Ti=400, K=0.8, boundaries=None):
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

    H_m_loop= H_mvector
    
    #Ti_si_loop=- (H_m_loop*K - K * sum_H_loop)/H_m
    #Ti_Cli_loop= -H_m_loop + sum_H_loop
    # I need to determine how Sum_H Progresses...

    Ti_si_loop = K*sum_Ti/(H_m_loop + K)
    Ti_Cli_loop = H_m_loop * sum_Ti/(H_m_loop + K)
    sum_H_loop = np.multiply(H_m_loop, (H_m_loop + K + sum_Ti))/(H_m_loop + K)

    
    #for idx, x in enumerate(range(round(timesteps))):
    for x in range(round(timesteps)):
        H_m_loop = Diff_Matrix @ H_m_loop
        # Question for Mike: Do we need to calculate all concentrations at each step of the loop or only once to update how H_m changesand then calcualte the rest at the end? 
        # Question 2: How should we constrain total water. What if Sum_H or Sum Ti constrains Ti_Clin. A it stands The total water isnt being constrained it is just calcualted based on the K relationship

        Ti_si_loop = K*sum_Ti/(H_m_loop + K)
        Ti_Cli_loop = H_m_loop * sum_Ti/(H_m_loop + K)
        sum_H_loop = np.multiply(H_m_loop,(H_m_loop + K + sum_Ti))/(H_m_loop + K)

        H_m_loop = (sum_H_loop - Ti_Cli_loop)

        #then update all defects with reaction equations.
        
        # if boundaries is not None:
        # this currently wont work. I need to make it so the boundaries and timesteps are the same.
        # v_loop[0], v_loop[-1] = boundaries[idx], boundaries[idx]
    return {'H_m_loop': H_m_loop, 'Ti_si_loop': Ti_si_loop, 'Ti_Cli_loop': Ti_Cli_loop, 'sum_H_loop': sum_H_loop}

# %%

# %%

# %%
#%%timeit -r 100

dt = 0.5  # 1 #0.0973 # time step seconds
N_points = 100
profile_length = 1500  # Microns
dX = profile_length / (N_points - 1)  # Microns
boundary = 0  # ppm
initial_C = 20  # ppm
v = np.mat(np.ones(N_points) * initial_C).T
v[0], v[-1] = boundary, boundary
v_initial = v

B = diffusion_matrix(DH2O = DH2O_Ol(1200), dt= dt, dx = dX, N_points= 100)

dicts= time_steper(v_initial, sum_Ti = 60, K=0.9, Diff_Matrix =B, timesteps = 60*60, boundaries=None)

plt.plot(dicts['H_m_loop'], Label = 'H_m')
plt.plot(dicts['Ti_Cli_loop'],Label='Ti_Cli')
plt.plot(dicts['sum_H_loop'], Label='Total_H')

plt.legend()



# %%
