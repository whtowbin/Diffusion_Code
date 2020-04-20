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
Ti_Cli, Ti_si, H_m, sum_Ti, sum_H = sympy.symbols(
    "Ti_Cli Ti_si H_m sum_Ti sum_H", positive=True)
K = sympy.symbols("K", positive=True)
#Ti_Cli, Ti_si, H_m, sum_Ti, sum_H = sympy.symbols("Ti_Cli Ti_si H_m sum_Ti sum_H", positive=True)


#%%
eq1 = (sum_Ti - Ti_si - Ti_Cli)
eq2 = (sum_H - H_m - Ti_Cli)
eqk = (K * Ti_Cli - (Ti_si * H_m))

# %%


# %%
sympy.nonlinsolve([eq1, eqk, eq2], [Ti_si, Ti_Cli, sum_H])
# %%
sympy.nonlinsolve([eq1, eqk, eq2], [Ti_si, Ti_Cli]) 
#str(Equations)
# %%

Equations = sympy.nonlinsolve([eq1, eqk, eq2], [Ti_si, Ti_Cli, H_m])
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

# %%
@jit(nopython=True)
def alt_time_steper(H_mvector, Diff_Matrix, timesteps, sum_Ti=400, K=0.8, boundaries=None):
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

    H_m_loop = H_mvector

    #Ti_si_loop=- (H_m_loop*K - K * sum_H_loop)/H_m
    #Ti_Cli_loop= -H_m_loop + sum_H_loop
    # I need to determine how Sum_H Progresses...

    # Goal: Input total_H and K and Sum_Ti
    # Diffuse H_m
    # React Ti_Cli and H_m to balance according to K
    # Check that everything is still less that Sum_H and sum_Ti? 
    # -how to do this last step? 


    Ti_si_loop = K*sum_Ti/(H_m_loop + K)
    Ti_Cli_loop = H_m_loop * sum_Ti/(H_m_loop + K)
    sum_H_loop = np.multiply(H_m_loop, (H_m_loop + K + sum_Ti))/(H_m_loop + K)

    #for idx, x in enumerate(range(round(timesteps))):
    for x in range(round(timesteps)):
        H_m_loop = Diff_Matrix @ H_m_loop


        Ti_si_loop = K*sum_Ti/(H_m_loop + K)
        Ti_Cli_loop = H_m_loop * sum_Ti/(H_m_loop + K)
        sum_H_loop = np.multiply(
        H_m_loop, (H_m_loop + K + sum_Ti))/(H_m_loop + K)

        H_m_loop = (sum_H_loop - Ti_Cli_loop)

        #then update all defects with reaction equations.

        # if boundaries is not None:
        # this currently wont work. I need to make it so the boundaries and timesteps are the same.
        # v_loop[0], v_loop[-1] = boundaries[idx], boundaries[idx]
    return {'H_m_loop': H_m_loop, 'Ti_si_loop': Ti_si_loop, 'Ti_Cli_loop': Ti_Cli_loop, 'sum_H_loop': sum_H_loop}


# %%
dt = 0.5  # 1 #0.0973 # time step seconds
N_points = 100
profile_length = 1500  # Microns
dX = profile_length / (N_points - 1)  # Microns
boundary = 0  # ppm
initial_C = 20  # ppm
v = np.mat(np.ones(N_points) * initial_C).T
v[0], v[-1] = boundary, boundary
v_initial = v

B = diffusion_matrix(DH2O=DH2O_Ol(1200), dt=dt, dx=dX, N_points=100)

dicts = alt_time_steper(v_initial, sum_Ti=60, K=0.9,
                    Diff_Matrix=B, timesteps=60*60, boundaries=None)

plt.plot(dicts['H_m_loop'], Label='H_m')
plt.plot(dicts['Ti_Cli_loop'], Label='Ti_Cli')
plt.plot(dicts['sum_H_loop'], Label='Total_H')

plt.legend()

# %%
# %% 
%%timeit
B @ v_initial

# %%

# %%

# %%

# %%

# %%

delta = (DH2O_Ol(1200) * dt) / ((dX) ** 2)
#delta = (DH2O * dt) / (dx ** 2)
mat_base = np.zeros(3)
mat_base[1] = 1 - 2 * delta
mat_base[0] = delta
mat_base[2] = delta
mat_base
test = 20*np.ones(100)
#test = np.reshape(test, test.size)
#%%

def conv_diff(vector_in, diffusion_kernel):
    return np.convolve(vector_in, diffusion_kernel, mode = "same")
    


# %%



# %%
%%timeit
conv_diff(vector_in=test, diffusion_kernel=mat_base)
# %%
f=conv_diff(vector_in=test, diffusion_kernel=mat_base)
# %%
f = conv_diff(vector_in=f, diffusion_kernel=mat_base)
f
# %%
test = test
pad = np.ones(5)*100
new = np.concatenate([pad, test, pad])
# %%
test = test 
pad = np.ones(3)*100
new = np.concatenate([pad,test,pad])
# %%
%%timeit
new = np.concatenate([pad, test[3:-3], pad])
# %%
%%timeit
new[0:3] = 70*np.ones(3)
#new[-1, -3] = new[0:3]
test = np.ones(100) * 20 
# %%
%%timeit
t1= np.pad(test, (3), 'constant', constant_values=(150))

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


pad = np.ones(3) * 100


def diffusion_step(vector_in, diffusion_kernel, pad, Bound_Concentration=0, ):
    # pad = np.ones(3) * Bound_Concentration put back in
    vector = np.concatenate([pad, vector_in, pad])
    return np.convolve(vector, diffusion_kernel, mode="same")[3:-3]
    # This is a slow iteration. Is there a better way to incorporate the padding for the convolution? It would be ideal if this only needed to be called every few iterations.   

def reaction_step():
    # define all reactions to work in one function 
    test = 5

def diffuse_react_iteration(init_vectors, diff_step_func, react_step_func,iterations, dt = 1):
    """
    
    Params
    ====================
    init_vectors: dict of all the initial vectors used
    diff_step_func: name of function used for 
    react_step_func: name of function used for reaction
    iterations: number of iterations to step
    dt=1: defaults to 1 second

    Return
    =====================

    """
# Output all of the function as dictionaries. These dictionaries will then be fed into the subsuquent loop. 

# %%

#loop = t1

def looper(init, N_it, kernel=kernel, pad=pad):
    loop = init
    for  c in range(N_it):
        loop = diffusion_step(loop, diffusion_kernel=kernel, pad=pad)
    return loop


#%%
%%timeit
loop1 = looper(t1, N_it=1000, kernel=kernel, pad = pad)
#%%
def conv_diff_bound(vector_in=new, diffusion_kernel=mat_base, bound =pad):
    new = conv_diff(vector_in, diffusion_kernel)
    new = np.concatenate([pad, new[3:-3], pad])
    return new


def conv_diff(vector_in, diffusion_kernel):
    return np.convolve(vector_in, diffusion_kernel, mode="same")
