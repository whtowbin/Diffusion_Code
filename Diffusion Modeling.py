#%%
import numpy as np
import scipy.linalg as la
import pandas as pd
import janitor
from matplotlib import pyplot as plt

"""
 At each step evaluate the liklihood and store it in a list. Need an error on the measrument and position
 This way I can run a montecarlo on the input parameters and sample the MAP point

 All I need to do to make this fit Megan's code is to interpolate the water in the melt to the decompression rate



1) Make a function that constructs concenetion profiles and diffusion matrix given a spacing.
2) Make diffusion model code as simply functionilized as possible.
3) Write funtion that minimizes sum of the residuals. Once it starts to grow again than it
4) The alternative would be to sample time with other factors but this seems wasteful since if time is a bad fit the other settings wont be fit properly.
We will see what is the most useful way to do this fit.
5) We can use a rapid optimizer to sample the other parameters.
"""
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


# These are critical for stability of model. Must think about time steps and model points.
N_points = 100
profile_length = 1500  # Microns
dX = profile_length / (N_points - 1)  # Microns

Distances = [0 + dX * Y - profile_length / 2 for Y in range(N_points)]


max_time_step = DH2O_Ol(1200) ** 2 / (4 * dX)
max_time_step
dt = 0.5  # 1 #0.0973 # time step seconds

boundary = 10  # ppm
initial_C = 18  # ppm


v = np.mat(np.ones(N_points) * initial_C).T
v[0], v[-1] = boundary, boundary
v_initial = v


# %%
# this is a term that take all the multiplication terms at each time step of the finite difference model into one term
# In the future I can update this every timestep if temperature changes with time.
# also update boundary conditions with time

DH2O = DH2O_Ol(1200)


def diffusion_matrix(DH2O, dt, dx, N_points=N_points):
    delta = (DH2O * dt) / (dx ** 2)

    mat_base = np.zeros(N_points)
    mat_base[0] = 1 - 2 * delta
    mat_base[1] = delta

    B = np.mat(la.toeplitz((mat_base)))
    B[0, 0], B[-1, -1] = 1, 1
    B[0, 1], B[-1, -2] = 0, 0
    return B


"""
for code that has a changing boundary condition look to equation 3.17.
dDl(t) means that we need to add the change at the boundary to the nearest
edge points at each time step.

https://warwick.ac.uk/fac/cross_fac/complexity/study/msc_and_phd/co906/co906online/lecturenotes_2009/chap3.pdf
"""
diffusion_matrix(12, 2, 1)

# %%

# %%

# %%

# @jit # Use numba to speed this function up with jit. It might be able to do some of the matrix multplication in parallel
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
        # I need to interpolate the decompression path. Instead of defining dt. I need to define
        # dp/dt.
        # v_loop[0], v_loop[-1] = boundaries[idx], boundaries[idx]
    return v_loop


#def decomp_path(Solex, dP_dt, dt):
    # Interpolate Solex H2O/Pressure path to timestep. dP/dt * dt should give the difference in pressure we need.
    # then we can get the number of points from (P[-1]-P[0])/dp
    # Maybe start at the pressure where dH/dP starts to increase past a treshold.

    #break



B = diffusion_matrix(DH2O, dt, dX)


def plot_profiles(v, B, times_seconds, Distances, dt):
    fig, ax = plt.subplots(figsize=(12, 8))
    for time in times_seconds:
        plt.plot(
            Distances,
            time_steper(v, B, round(time / dt)),
            label=str(time / 60) + " Minutes",
        )

    ax.legend(loc=1)
    ax.set_xlabel("Distance to center of crystal ")
    ax.set_ylabel("ppm water")


times = [0, 60 * 5, 15 * 60, 60 * 60, 60 * 60 * 1.5, 60 * 60 * 2]

plot_profiles(v, B, times, Distances, dt)
plt.savefig(" Olivine Diffusion Stalled 18ppm initial")
# time given in seconds then divided by dt to give number of steps

"""
plt.plot(Distances, time_steper(v, B, 0/dt) )
plt.plot(Distances, time_steper(v, B, 10/dt))
plt.plot(Distances, time_steper(v, B, 100/dt))
plt.plot(Distances, time_steper(v, B, 1000/dt))
plt.plot(Distances, time_steper(v, B, 7000/dt))
plt.plot(Distances, time_steper(v, B, 10000/dt))

ax.set_xlabel("Distance to center of crystal ")
ax.set_ylabel("ppm water")
 7000/dt/60

"""
# The following parameters ae the wrong parameters
# D_OPX = 1e12 * (10 ** (-3)) * np.exp(-181000 / (8.314 * T_K)) / 10
# D_CPX = 1e12 * (10 ** (-3)) * np.exp(-181000 / (8.314 * T_K)) / 500


DH2O
# %%

# %%
import pynams.diffusion.models as pydiff


# CPX Diffusion
D_CPX = DH2O_Ol(1200) / 200
f, a, x, y = pydiff.diffusion1D(
    500, np.log10(D_CPX / 10 ** 12), 135 * 60, maximum_value=100, init=100, fin=0
)
# init = 100, fin=0);

fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(x, y)
max(y)
min(y)
CPX_Distances = np.array(
    [123, 217, 34, 80, 178, 176, 213, 40, 171, 37, 129, 51, 116, 94, 222, 33]
)
CPX_Concentration = np.array(
    [
        587.84,
        550.13,
        502.16,
        600.01,
        555.79,
        462.74,
        584.46,
        482.85,
        600.11,
        515.52,
        579.71,
        585.51,
        578.85,
        467.86,
        575.57,
        459.29,
    ]
)
plt.plot(CPX_Distances - 150, CPX_Concentration, marker="o", linestyle="None")
ax.set_xlabel("Distance (Microns)")
ax.set_ylabel("H2O ppm")
# ax.set_ylim(0,100)


# %%

# Iterate 1% , 50% and 90%equilibration to make final plot


N_points = 100
profile_length = 500  # Microns
dX = profile_length / (N_points - 1)  # Microns

Distances = [0 + dX * Y - profile_length / 2 for Y in range(N_points)]

boundary = 0  # 0 #ppm
initial_C = 100  # 100 # ppm
v = np.mat(np.ones(N_points) * initial_C).T
v[0], v[-1] = boundary, boundary
v_initial = v
df1 = pd.DataFrame(index=[10, 100, 200], columns=range(900, 1350, 50))
df50 = pd.DataFrame(index=[10, 100, 200], columns=range(900, 1350, 50))
df90 = pd.DataFrame(index=[10, 100, 200], columns=range(900, 1350, 50))

Panels = [df1, df50, df90]
for idx, percent_eq in enumerate([1, 50, 90]):
    percent_remain = 100 - percent_eq  # diffusion out
    df = Panels[idx]
    for T_C2 in range(900, 1350, 25):
        for F_cpx2 in [10, 100, 200]:
            D_cpx2 = DH2O_Ol(T_C2) / F_cpx2
            v_loop = v_initial
            time = 11 * 60
            time_tot = 0
            while v_loop[int(len(v_loop) / 2)] > percent_remain:
                # while np.max(v_loop) > percent_remain: # diffusion out
                # while v_loop[int(len(v_loop) / 2)] < percent_eq:  # diffusion in
                time = 1 * 60
                time_tot = 1 * 60 + time_tot
                dt = 1
                B = diffusion_matrix(D_cpx2, dt, dX)
                v_loop = time_steper(v_loop, B, time)
                # if np.max(v_loop) < percent_remain:
                #    df.loc[F_cpx2][T_C2] = time_tot/60
            df.loc[F_cpx2][T_C2] = time_tot / 60

df1
df50
df90

# %%


N_points = 100
profile_length = 500  # Microns
dX = profile_length / (N_points - 1)  # Microns

Distances = [0 + dX * Y - profile_length / 2 for Y in range(N_points)]

boundary = 0  # 0 #ppm
initial_C = 100  # 100 # ppm
v = np.mat(np.ones(N_points) * initial_C).T
v[0], v[-1] = boundary, boundary
v_initial = v

T_C3 = 1200
T_K3 = T_C3 + 273
D_cpx3 = DH2O_Ol(1200) / F_cpx2


Minutes = (250 ** 2 / D_cpx3) / 60

D_OL1 = 10 ** (-5.4) * np.exp(-130000 / ((8.314 * T_K3)))

(250 * 10 ** -6) ** 2 / (D_OL1 / 200) / 60 / 60
Minutes
Minutes / 60

60000 / 0.1 / 60
# %%

# %%
# font = {"family": "normal", "weight": "bold", "size": 20}

# plt.rc("font", **font)
fix, ax = plt.subplots(figsize=(12, 8))
fig, ax = plt.subplots(figsize=(12, 8))
T_K2 = 1200 + 273
time = 135 * 60
# time_tot = 15 * 60 + time_tot
D_cpx2 = DH2O_Ol(1200) / 200  # olivine diffusivity divided by 200
np.log10(D_cpx2 / (1e12))
dt = 1
B = diffusion_matrix(D_cpx2, dt, dX)
v_loop1 = time_steper(v_initial, B, 0)
v_loop2 = time_steper(v_initial, B, time)
plt.plot(Distances, v_loop1, linewidth=3, label="Initial")
plt.plot(Distances, v_loop2, linewidth=3, label="135 Minutes")
ax.legend()
ax.set_ylabel("% Initial Concentration")
ax.set_xlabel("Distance Microns")
plt.savefig("Core_Preserved")


# %%
# df1 = df
# df90 = df
# df50 = df

font = {"family": "normal", "weight": "bold", "size": 20}

plt.rc("font", **font)
fix, ax = plt.subplots(figsize=(12, 8))

df90.iloc[2].plot(label="90% Equilibrated with Magma", lw=4, c="#1f77b4")
# df50.iloc[2].plot(label = '50% Equilibrated with Magma', lw = 4, c = '#ff7f0e')
df1.iloc[2].plot(
    label="Initial Concentration Preserved only in Core", lw=4, c="#d62728"
)
# plt.plot(x= [800, 1400], y = [300, 300], color='k', linestyle='-', linewidth=2,)
ax.set_xlabel("Temperature ˚C")
ax.set_ylabel("Minutes to Equilibrate")
ax.set_title("Time to Equilibrate Water in 0.5mm Clinopyroxene")

ax.fill_between([800, 1350], [300, 300], [700, 700], color="green", alpha=0.3)
ax.legend(loc=2)
plt.savefig("Time to eq cpx")
ax.annotate("0.05 MPa/s ", xy=(1215, 710), color="green")
ax.set_xlim(900, 1300)
ax.set_ylim(0, 10000)
plt.savefig("Time to eq cpx_200slow_1%_zoom_")


3000 / 60
# %%
# Max water in ol at depth
Intial_low = 2.7 * 0.0007 * 10000
Initial_high = 2.7 * 0.0015 * 10000
Final = 10


# %%
""" I need to write a funtion that initiallizes a diffusion set up
It shoudl look like the code below and return an object  with each of the condition"""

N_points = 100
profile_length = 500  # Microns
dX = profile_length / (N_points - 1)  # Microns

Distances = [0 + dX * Y - profile_length / 2 for Y in range(N_points)]

boundary = 100  # 0 #ppm
initial_C = 0  # 100 # ppm
v = np.mat(np.ones(N_points) * initial_C).T
v[0], v[-1] = boundary, boundary
v_initial = v


font = {"family": "normal", "weight": "bold", "size": 20}

plt.rc("font", **font)
fig, ax = plt.subplots(figsize=(12, 8))
T_K2 = 1200 + 273
time = 135 * 60
# time_tot = 15 * 60 + time_tot
D_cpx2 = DH2O_Ol(1200) / 200  # olivine diffusivity divided by 200
np.log10(D_cpx2 / (1e12))
dt = 1
B = diffusion_matrix(D_cpx2, dt, dX)
v_loop1 = time_steper(v_initial, B, 0)
v_loop2 = time_steper(v_initial, B, 30 * 60)
v_loop3 = time_steper(v_initial, B, 135 * 60)
v_36hr = time_steper(v_initial, B, 36.6 * 60 * 60)

plt.plot(Distances, v_loop1, linewidth=3, label="Initial")
# plt.plot(Distances, v_loop2, linewidth=3, label="30 Minutes")
plt.plot(Distances, v_loop3, linewidth=3, label="2.2 Hours")
plt.plot(Distances, v_36hr, linewidth=3, label="36.6 Hours")

ax.legend()
ax.set_title("Equilibration of a 0.5mm clinopyroxene in un-degassed magma")
ax.set_ylabel("% Initial Concentration")
ax.set_xlabel("Distance Microns")
plt.savefig("500um CPX 37 hr Equilibration ")
