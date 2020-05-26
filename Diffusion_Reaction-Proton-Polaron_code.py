#%%
import numpy as np
import scipy.linalg as la
import sympy as sympy

from matplotlib import pyplot as plt


# %%

# %%
''' 
TODO: Think about whether Trivalent vacancies are mostly just a halfway point in diffusion and those defects and continue to diffuse. Have a population of interstitial hydrogen that can diffuse. All other sites must react with this i in order to move.
Confused about whether to think of proton polaron as a reaction (like below) or as a 'magic' diffusivity that doesn't get specifically modelled as a reaction 
Also confused about how to incorporate other defects -- somehow have added an equation but still only really modelling the Ti defect
Also confused about how to solve equations for multiple defects
Maybe just need to substitute pp into TiCli equation to get one equation

In spectrum [H_i]+ is the sum of 2*Mg peak and the tri peak..?
2 Hydrogen on one site and 2 on separate sites. 

We need to account for how the 2H+ in Mg-vacancies become a Trivalent defect once one of the H+ leaves via proton-polaron. Do we need another K to balance this? Do we instead keep track of loss from each site? probably easier to use a K to balance? 

Ti_Clinohumite to Proton Polaron (not explicitly modeling)
{Ti_m + 2H_si} = [Ti_m + V_si]'' + [H_i]+

{Ti_m + 2H_si} = [Ti_si]+ {2H+ + V_m}
# This assumes H in Ti-Cli can only diffuse from M-site

2[H_i]+  = 2H_m
1) Mg-site reaction (Proton Polaron)
[H_i]+ + Fe_m = Fe_m+ + H 
Can also be written as 

{2H+ + V_m} + 2{Fe_m} =  {2Fe3_m+ + V_m} + 2H

or 
{H_m} + Fe_m = Fe_m+ + H 

Trivalent reaction is
{2H+ + V_m} + 1{Fe_m} =  {Fe3_m+ + V_m + H_m+} + H


2) Ti-Clinohumite reaction (Proton Polaron)
{Ti_m + 2H_si} +2Fe_m = [Ti_m + V_si]'' + 2Fe_m+ + 2H
# This assumes H can diffuse directly from Ti_Cli



[2fe3+ + V_m] + [v_m +2H_i+] = [2fe3+ 2v_m + 2H_i] 
[2fe3+ + V_m] + [v_m +2H_i+] + 2[Fe_m] =[2fe2+ + V_m + 2H_i+] + [2fe3+ + V_m]

maybe solve equations in log space 


K1 = (Fe_m+ * H) / ([H_m]+ * Fe_m)  #Unclear which way is most favorable should driven by concnetration gradient more than other thing 
K1 = (Fe_m+ * H^2) / ([H_m]+ )


K2 =  ([Ti_si]*[{2H+ + V_m}]) / {Ti_m + 2H_si} should be between 0 and 1. 



sum_Ti = [Ti_m + V_si]'' + [Ti_m+2H_si]
sum_H = 2{2H+ + V_m} + 2[Ti_m+2H_si] + H
sum_Fe = Fe_m + Fe_m+

think about lag in ferrous iron.  Coordination around M sites. 1 in 10 Msites has an Fe. 
sum_m = {2H+ + V_m}+ 2{Fe_m}+ 3{2Fe_m+ + V_m}+ {Ti_m + 2H_si}
Defining the total M-sites might be useful if I include the coordination of Fe near M vacancies. 
 1/10 M-sites are  Fe. How can I model this stochastically? 
# sum_m  seems redundant 
Total M sites vs total Fe. 
Solved for 

eq = solve(eq2,Ti_Cli)
eq1 = subs(eq1,Ti_Cli, eq)
'''
"""
K1, K2 = sympy.symbols("K1 K2", positive=True, real=True)
Ti_Cli, Ti_si, H_m, H, Fe_m, Fe3_m, sum_Ti, sum_H, sum_Fe, sum_m = sympy.symbols(
    "Ti_Cli Ti_si H_m H Fe_m Fe3_m sum_Ti sum_H sum_Fe sum_m", positive=True, real=True)
"""

#K2 = sympy.symbols("K2", positive=True, real=True)
Ti_Cli, Ti_si, H_m, H, Fe_m, Fe3_m = sympy.symbols( "Ti_Cli Ti_si H_m H Fe_m Fe3_m", positive=True, real=True)

#sum_Ti, sum_H, sum_Fe, sum_m, K1, K2 = 10, 20, 10, 30, 1, 1

#%%
# Ferrous Iron Free formulation
eq_Ti = (sum_Ti - Ti_si - Ti_Cli)
eq_H = (sum_H - 2*H_m - 2*Ti_Cli - H)
eq_m = sum_m - H_m - Ti_Cli - 3*Fe3_m

eqk1 = (K1 * (H_m)) - (H**2 * Fe3_m)
eqk2 = (K2 * (Ti_Cli)) - (Ti_si * H_m)

#eq_Fe = sum_Fe - 2*Fe_m - 2*Fe3_m
#eq_mfe = sum_m - H_m - Ti_Cli - 3*Fe3_m -3*Fe_m
#eqk1fe = (K1 * (H_m * Fe_m**2)) - (H**2 * Fe3_m) 
#eqk1 = (K1 * (H_m * Fe_m)) - (H * Fe3_m) # This is a half filled site. Mike has it multiplied differently...  This seems to be the Trivalent equation but we must use the Tri site to account for hydrogen in that site. 


#{Ti_m + 2H_si} = [Ti_si] + {2H + + V_m}
#K1 = (Fe3_m + * H**2) / ([H_i] + * Fe_m**2)

#K2 = ([Ti_m + V_si]'' * [H] ^ 2 * Fe_m+^2) / ([Ti_m+2H_si]*Fe_m ^ 2)


# %%
# Solve for M-sites
# This equation assumes Fe is an infinite res compared to Fe3+
# We should think about how to model % of M-sites coordinated by Fe 
Equations = sympy.nonlinsolve(
    [eq_H, eq_Ti, eq_m, eqk1, eqk2], [Ti_si, Ti_Cli, H_m, Fe3_m, H])

Equations

# Ti_si, Ti_Cli, H_m, Fe3_m

#%%
sum_Ti, sum_H, sum_Fe, sum_m, K1, K2 = 10, 600000, 100000, 300, 1000, 10000

Values = sympy.nsolve(
    [eq_Ti, eq_m, eqk1, eqk2,eq_H],
    [Ti_si, Ti_Cli, H_m, H, Fe3_m],
    ( 5,5,5,5,59))

Values
# %%
"""
A few thoughts on parameter spaces to search. 
as we have it set up the only parameter that changes in our model between timesteps is the Hydrogen concentration. Maybe we can generate a new lookup table everytime we run the diffusion model and only search the space limited by the max hydrogen(sum_h) concentrations and the other set parameters.

I am a bit worried that our reactions don't allow for hysteresis i.e. what happened in the previous timesteps doesnt effect hydrogen incorporation in the subsequent. Hydration and dehydration are perfectly reversible. 

As we diffuse water out we should expect Fe3+ to grow and limit diffusion. Maybe this set up is actually fine since we are contraining the total number of M-sites but this is worth considering.

We should set out initial total Msites based on the assumptions we have about the amount of Fe#+ preexisting in the crystal, The amount of Ti in the crstal, and the amount of water. It would be great to try to fit elizabeth's data besides Anna's 
"""
n= 5
Values = [-1, -2, -3, -4, -5]
Values_Matrix=np.zeros((n,n,n,n,n,n,5))

sum_Ti_list = np.linspace(1,500,n)
sum_H_list = np.linspace(1, 500, n)
sum_Fe_list = np.linspace(1, 500, n)
sum_m_list = np.linspace(1, 500, n)
K1_list = np.logspace(-2,2,n)
K2_list = np.logspace(-2, 2, n)
        

# I think the problem might be that the equations are refering to sympy equations that are set up with the variable before they are assigned in the loop              

for idx0, sum_Ti in enumerate(sum_Ti_list):
    for idx1, sum_H in enumerate(sum_H_list):
         for idx2,sum_Fe in enumerate(sum_Fe_list): # This needs to be cut...
             for idx3,sum_m in enumerate(sum_m_list): 
                 for idx4,K1 in enumerate(K1_list):
                     for idx5,K2 in enumerate(K2_list):
                        root_values = [-1, -2, -3, -4, -5]
                        print('next one')
                        print(sum_Ti,
                              sum_H,
                              sum_Fe,
                              sum_m,
                              K1,
                              K2)
                        while min(root_values) < 0:
                        
                            #Ti_si_guess, Ti_Cli_guess, H_m_guess, H_guess,Fe3_m_guess = np.random.randint(0,100,5)
                            #Ti_si_guess= np.random.randint(0,sum_Ti+ 1) 
                            Ti_Cli_guess = np.random.randint(0,sum_Ti+ 1)
                            H_m_guess= np.random.randint(0,sum_H+1)
                            #H_guess = np.random.randint(0, sum_H)
                            #Fe3_m_guess= np.random.randint(0,sum_m) 
                            
                            #We should start our guess with the last sucessful guess in the column we are iterating. 
                            # we can start our guesses randomly for the column we are iterating but also we can try starting them with the first sucessful solution of the previous column. 
                            try:
                                Values = sympy.nsolve(
                                    [eq_H, eq_Ti, eq_m, eqk1, eqk2], 
                                    [Ti_si, Ti_Cli, H_m, H, Fe3_m], 

                                    (sum_Ti-Ti_Cli_guess, Ti_Cli_guess,
                                        H_m_guess, sum_H - 2*H_m_guess- 2*Ti_Cli_guess, 
                                        (sum_m - H_m_guess- Ti_Cli_guess)/3
                                        ))
                                
                            except:
                                #Values = [-1, -1, -1, -1, -1]
                                print('bad values')
                                Ti_Cli_guess = np.random.randint(0, sum_Ti + 1)
                                H_m_guess = np.random.randint(0, sum_H+1)
                            #Values_Matrix[idx0, idx1, idx2, idx3, idx4,idx5] = np.array(list(Values), float)

                            if min(list(Values)) > 0:
                                print(Values)
                                root_values=list(Values)

# Use the previous values as the starting guess for the next guess. Only use rand if fails. 

"""
eq_Ti = (sum_Ti - Ti_si - Ti_Cli)
eq_H = (sum_H - 2*H_m - 2*Ti_Cli - H)
eq_m = sum_m - H_m - Ti_Cli - 3*Fe3_m
"""

"""
[Ti_si, Ti_Cli, H_m, H, Fe3_m]

sum_Ti-Ti_Cli_guess

Ti_Cli_guess

H_m_guess

sum_H - 2*H_m_guess - 2*Ti_Cli_guess

(sum_m - H_m_guess - Ti_Cli_guess)/3
"""
# %%
subeq = Equations1.subs({K1:1, K2:1, H:100, sum_m:500 })
#  H, sum_Ti, sum_H, sum_m, K1, K2
sympy.plotting.plot3d(subeq.args[1], (sum_H,0, 500), (sum_Ti,0, 400), )

#subeq = Equations2.subs({sum_Ti: 20, H: 100, sum_H: 200, sum_m: 500})
#sympy.plotting.plot3d(subeq.args[2], (K1, -100, 100), (K2, -105, 105), )
#ylabel='K1', xlabel ='k2')

# %%

# Solve for K Ranges
# This equation assumes Fe is an infinite res compared to Fe3+
# We should think about how to model % of M-sites coordinated by Fe
K_Equations = sympy.nonlinsolve(
    [eq_H, eq_Ti, eq_m, eqk1, eqk2], [K1, K2])

list(K_Equations)
# %%
# Solve for all species including Fe2+
# Maybe we just set the amount of Fe2+ low and assume that means we can't diffuse more near the vacancies. 
#sympy.nonlinsolve(
Equations = sympy.nonlinsolve(
    [eq_H, eq_Ti, eq_Fe, eqk1fe, eqk2, eq_mfe], [Ti_si, Ti_Cli, H_m, Fe3_m, Fe_m],)
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


#%%

Diffusivity = DH2O_Ol(1200)
