import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

OPX_R30 = {'Min_to_Rim_µm':[36,73,162,210,326],
'H2O_ppm': [167.86,249.11,357.40,385.77,395.45],
'StdDev_H2O': [5.47,7.94,6.03,6.75,9.87]}


dfopx30 = pd.DataFrame(OPX_R30, index = OPX_R30['Min_to_Rim_µm'])
#dfopx30.set_index('Min_to_Rim_µm',inplace=True)
dfopx30.sort_index(inplace = True)

CPX_All_Points = {

'Min_to_Rim_µm':[123,217,34,80,178,50,213,40,171,37,129,51,116,22,222,33],

'H2O_ppm':[587.84,550.13,502.16,600.01,555.79,462.74,584.46,482.85,600.11,515.52,579.71,585.51,578.85,467.86,575.57,459.29],

'StdDev_H2O':[8.71,11.61,17.21,9.50,12.13,6.28,7.58,
11.74,8.84,6.30,20.73,10.07,13.15,5.62,8.94,7.51]
}

dfcpx = pd.DataFrame(CPX_All_Points, index = CPX_All_Points['Min_to_Rim_µm'] )
#dfcpx.set_index('Min_to_Rim_µm',inplace=True)
dfcpx.sort_index(inplace = True)
dfcpx

Scoria_ol2 = {
'Min_to_Rim_µm':[154,117,56,23],
'H2O_ppm':[10.60,9.95,7.24,6.01],
'StdDev_H2O':[0.36,0.22,0.09,0.21]
}

df_sOl2 = pd.DataFrame(Scoria_ol2, index = Scoria_ol2['Min_to_Rim_µm'])
#df_sOl2.set_index('Min_to_Rim_µm',inplace=True)
df_sOl2

GCB_ol6 = {
'Min_to_Rim_µm':np.array([19,102,214,294,376,]),
'H2O_ppm':np.array([7.34,10.50,9.67,10.44,10.38,]),
'StdDev_H2O':np.array([0.20,0.21,0.45,0.23,0.23,])
}


dfGCB_ol6 = pd.DataFrame(GCB_ol6, index = GCB_ol6['Min_to_Rim_µm'])
#dfGCB_ol6.set_index('Min_to_Rim_µm',inplace=True)
dfGCB_ol6
# %%
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

plt.rc('font', **font)
fix, ax = plt.subplots(figsize = (12,8))

dfGCB_ol6.plot( x='Min_to_Rim_µm',y='H2O_ppm', linestyle='dashed', yerr='StdDev_H2O', xerr=15, color= '#1f77b4',
 marker= 'o', capsize=5, markeredgewidth = 1.0,  markersize= 10, ax=ax, label = "Xenolith Olivine")

df_sOl2.plot( x='Min_to_Rim_µm',y='H2O_ppm', linestyle='dashed', yerr='StdDev_H2O', xerr=15, color= '#d62728',
 marker= 'o', capsize=5, markeredgewidth = 1.0, ax=ax, markersize=10, label = "Scoria Phenocryst Olivine" )
plt.ylabel("Water (ppm)")
plt.xlabel("Distance from rim (microns)")

ax.set_ylim(0)
plt.savefig('Olivine Zoning',  dpi=300, bbox_inches='tight')


# %%

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

plt.rc('font', **font)
fix, ax = plt.subplots(figsize = (12,8))

dfopx30.plot( x='Min_to_Rim_µm',y='H2O_ppm', linestyle='dashed', yerr='StdDev_H2O', xerr=15, color= 'darkolivegreen',
 marker= 'o', capsize=5, markeredgewidth = 1.0,  markersize= 10, ax=ax, label = "Interior Orthopyroxene")


plt.ylabel("Water (ppm)")
plt.xlabel("Minium Distance from rim (microns)")

ax.set_ylim(0)
plt.savefig('OPx R3 Zoning',  dpi=300,)

# %%
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

plt.rc('font', **font)
fix, ax = plt.subplots(figsize = (12,8))

dfcpx.plot( x='Min_to_Rim_µm',y='H2O_ppm', linestyle='none', yerr='StdDev_H2O', xerr=15, color= 'forestgreen',
 marker= 'o', capsize=5, markeredgewidth = 1.0,  markersize= 10, ax=ax, label = "All Clinopyroxene Measurements")


plt.ylabel("Water (ppm)")
plt.xlabel("Minium Distance from rim (microns)")

ax.set_ylim(0)
plt.savefig('CPX Zoning',  dpi=300,)

# %%



def plot_profile(Name,df):
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 20}

    plt.rc('font', **font)
    df.plot( x='Min_to_Rim_µm',y='H2O_ppm', linestyle='dashed', yerr='StdDev_H2O', xerr=15, color= '#d62728',
    marker= 'o', capsize=5, markeredgewidth = 1.0, ax=ax, markersize=10, label = Name )


    plt.ylabel("Water (ppm)")
    plt.xlabel("Distance from rim (microns)")
    plt.savefig('Name',  dpi=300, bbox_inches='tight')

# %%
plot_profile("CPX_ALL", dfGCB_ol6)

plt.errorbar(x= Min_to_Rim_µm, y= H2O_ppm, yerr= StdDev_H2O, xerr = Xerr,
    marker='.', linestyle='dashed', capsize=10,
    markeredgewidth = 1.0,   markersize= 12, )

plt.ylabel("Water (ppm)")
plt.xlabel("Distance from rim (micron)")
plt.title("GCB OPX R3 interior Water Concentration")
plt.savefig('Name',  dpi=300, bbox_inches='tight')

CPX
    plt.errorbar(x= Min_to_Rim_µm, y= H2O_ppm, yerr= StdDev_H2O, xerr = Xerr,
        marker='.', linestyle='dashed', capsize=10,
        markeredgewidth = 1.0,   markersize= 12, )

    plt.ylabel("Water (ppm)")
    plt.xlabel("Distance from rim (micron)")
    plt.title("GCB OPX R3 interior Water Concentration")
    plt.savefig('Name',  dpi=300, bbox_inches='tight')
