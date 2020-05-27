# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.optimize import fsolve, root
 # %%

def F(u,sum_Ti,sum_H,sum_m,K1,K2):
    Ti_si = u[0]
    Ti_Cli = u[1]
    H_m = u[2]
    H = u[3]
    Fe3_m = u[4]
    
    eq_Ti = (sum_Ti - Ti_si - Ti_Cli)
    eq_H = (sum_H - 2*H_m - 2*Ti_Cli - H)
    eq_m = sum_m - H_m - Ti_Cli - 3*Fe3_m
    eqk1 = (K1 * (H_m)) - (H**2 * Fe3_m)
    eqk2 = (K2 * (Ti_Cli)) - (Ti_si * H_m)
    
    F = np.array([eq_Ti,eq_H,eq_m,eqk1,eqk2])
    return F


[sum_Ti,sum_H,sum_m,K1,K2] = [100.,200.,1000.,1.,1.]
# u = []
"""
sum_Ti=np.array([100]) 
sum_H=np.array([200,400,600])
sum_m=np.array([1000]) 
K1=np.array([1]) 
K2=np.array([1])
"""
Ti_si = np.array([5,10,15])
Ti_Cli = np.array([5, 10, 15])
H_m = np.array([10, 20, 30])
H = np.array([10,20,30])
Fe3_m = np.array([100,200,300])

"""
u0 = np.array([Ti_si,
      Ti_Cli,
      H_m,
      H,
      Fe3_m])
"""
Ti_Cli_guess= np.array([10,20,30])
H_m_guess = np.array([30,20,10])

u0 =np.array([sum_Ti-Ti_Cli_guess, Ti_Cli_guess,H_m_guess, sum_H - 2*H_m_guess - 2*Ti_Cli_guess,(sum_m - H_m_guess - Ti_Cli_guess)/3 ])

u0 = np.array([5., 5., 10., 10., 100.])
#root(residual, guess, method='krylov', options={'disp': True})

u = fsolve(F,u0,(sum_Ti,sum_H,sum_m,K1,K2))
#u = root(F, u0, (sum_Ti, sum_H, sum_m, K1, K2),
         #method='krylov', options={'disp': True})
error = F(u,sum_Ti,sum_H,sum_m,K1,K2)
print('u={} \nerr={}'.format(u,error))
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


def boundary_cond(C=0):
    # C: single concentration at boundary ( in the future we will support changing boundary conditions)
    pad = np.ones(3) * C
    return pad


def diffusion_step(vector_in, diffusion_kernel, pad):
    # pad = np.ones(3) * Bound_Concentration put back in
    vector = np.concatenate([pad, vector_in, pad])
    return np.convolve(vector, diffusion_kernel, mode="same")[3:-3]
# %%


def initial_root_find(sum_Ti, sum_H, sum_m, K1, K2):
    root_values = [-1, -2, -3, -4, -5]
    while min(root_values) < 0:

            Ti_Cli_guess = np.random.randint(0, sum_Ti + 1)
            H_m_guess = np.random.randint(0, sum_H+1)

            try:
                H_m_guess = np.array([30, 20, 10])

                u0 = np.array([sum_Ti-Ti_Cli_guess, Ti_Cli_guess, H_m_guess, sum_H -
                    2*H_m_guess - 2*Ti_Cli_guess, (sum_m - H_m_guess - Ti_Cli_guess)/3])

                u0 = np.array([5., 5., 10., 10., 100.])

                root_values = fsolve(F, u0, (sum_Ti, sum_H, sum_m, K1, K2))

            except:
                print('bad values')
                Ti_Cli_guess = np.random.randint(0, sum_Ti + 1)
                H_m_guess = np.random.randint(0, sum_H+1)
    return root_values


# %%
sum_Ti, sum_H, sum_m, K1, K2 = 10, 40, 100, 1, 1
initial_root_find(sum_Ti, sum_H, sum_m, K1, K2)

# %%
def time_steper(Diffusivity, timesteps, sum_Ti, sum_H, sum_m, K1, K2, dt=0.5, dx=10, N_points=100, bound_concentration=0):
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

    """[summary]
    Ti_si = u[0]
    Ti_Cli = u[1]
    H_m = u[2]
    H = u[3]
    Fe3_m = u[4]
    """

# We should update this initial to calculate sum_m. That is because we have an initial understanding of Fe3+ I forget the paper but I think it is generally ~3%
#  
    u0 = initial_root_find(sum_Ti, sum_H, sum_m, K1, K2)

    u = ( 
    np.ones(N_points) * u0[0],
    np.ones(N_points) * u0[1],
    np.ones(N_points) * u0[2],
    np.ones(N_points) * u0[3],
    np.ones(N_points) * u0[4])


    sum_H_loop = np.ones(N_points) * sum_H

    # Building diffusion code
    kernel = diffusion_kernel(Diffusivity, dt, dx)
    bound = boundary_cond(C=bound_concentration)

    # Loop to iterate diffusion and reaction
    #for idx, x in enumerate(range(round(timesteps))):
    for x in range(round(timesteps)):
        # Diffusing Hydrogen u[3]
        print(u)
        H_loop = diffusion_step(u[3], kernel, bound)
        sum_H_loop = 2*u[2] + 2*u[1] + H_loop

        u = root(F, u, (sum_Ti, sum_H_loop, sum_m, K1, K2), method='krylov').x
        
    return {'H_m_loop': u[2], 'Ti_Si_loop': u[0], 'Ti_Cli_loop': u[1], 'Fe3_m': u[4], 'sum_H_loop': sum_H_loop, 'H_loop': u[3]}


# %%

#%%timeit #-r 100


dt = 1  # 1 #0.0973 # time step seconds
N_points = 10
profile_length = 1500  # Microns
dx = profile_length / (N_points - 1)  # Microns

#dicts= time_steper(sum_H =100, sum_Ti = 60, K=0.8, Diff_Matrix =B, timesteps = 60*60, N_points= 100, boundaries=None)

dicts = time_steper(sum_Ti=100, sum_H=200, sum_m=250, K1=1, K2=2, Diffusivity=DH2O_Ol(
    1200), timesteps=60*5, dt=dt, dx=10, N_points=N_points, bound_concentration=0)
# %%
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(2*dicts['H_m_loop'], Label='H_m')
plt.plot(2*dicts['Ti_Cli_loop'], Label='Ti_Cli')
plt.plot(dicts['sum_H_loop'], Label='Total_H')
plt.plot(2*dicts['Fe3_m'], Label='Fe3+')
plt.plot(dicts['H_loop'], Label='H')
#plt.plot(dicts['Ti_Si_loop'], Label='Ti_Si')

#eq_H = (sum_H - 2*H_m - 2*Ti_Cli - H)
plt.legend(prop={'size': 20})


#%%

Diffusivity = DH2O_Ol(1200)
