from sigma_cdm_func import *
from classy import Class
import numpy as np
import scipy.interpolate as intp
import matplotlib.pyplot as plt
from alpha_func import *
from functions import *
import h5py

# a_here = 1
# mass = 1e10 #4/3*np.pi*rho_crit*8**3
# #print('m_8 = ',mass)
# print(sigma_cdm(mass))

# file_3 = './Code_own/Data/sigmacdm_1.00_+0.000_1.00.txt'
# data_sp= np.loadtxt(file_3,skiprows=1)
# # print(data_sp)
# def spl_intp(ms):
#     kplo = 200
#     kphi = 1
#     i_mod= 0
#     nmod = 2
#     m_arr= data_sp[:,0]
#     s_arr= data_sp[:,1]
#     s_2  = data_sp[:,2]
#     a_arr= data_sp[:,3]
#     a_2  = data_sp[:,4]
#     i_mod= 1+i_mod%nmod
#     klo = kplo-1
#     khi = kphi-1
#     if m_arr[khi] < ms or m_arr[klo] > ms:
#         klo = 0
#         khi = kplo-1
#         while khi-klo > 1:
#             k = int((khi+klo)/2)
#             if m_arr[k] > ms:
#                 khi=k
#             else:
#                 klo=k
#         h   = m_arr[khi]-m_arr[klo]
#         h2  = (h**2)*0.1666667
#         invh= 1/h
#     aa = (m_arr[khi]-ms)*invh
#     bb = (ms-m_arr[klo])*invh
#     a3 = (aa**3-aa)
#     b3 = (bb**3-bb)
#     sigma = aa*s_arr[klo]+bb*s_arr[khi]+(a3*s_2[klo]+b3*s_2[khi])*h2
#     alpha = aa*a_arr[klo]+bb*a_arr[khi]+(a3*a_2[klo]+b3*a_2[khi])*h2
#     return sigma,alpha

# print('sigma = ',spl_intp(3.61766217E+12)[0])

# scla  = 0.9/(spl_intp(3.61766217E+12)[0])
# print('scla = ',scla)
# sigma = spl_intp(mass*0.1825**3/0.25)[0]
# print('sigma_test = ',sigma*scla)
# print('alpha = ',spl_intp(mass*0.1825**3/0.25)[1])

# def sig(m,a):
#     z = 1/a-1
#     R = (3*m/(4*np.pi*rho_crit))**(1/3)
#     return cosmo.sigma(R,z,h_units=True)
# print(sig(mass))
# print('--------------------------')
# print('alpha = ',alpha(mass))

# lsig = lambda M: np.log(sigma_cdm(M))

# m_array = np.linspace(1e8,1e15,1000)

# # logSig = []
# # for i in range(len(m_array)):
# #     logSig.append(lsig(m_array[i]))

# # logM = np.log(m_array)

# # interpolation = intp.UnivariateSpline(logM,logSig)
# # alpha = interpolation.derivative(n=1)

# # print('here: ',np.exp(interpolation(np.log(mass))))
# # print('alpha = ', alpha(np.log(mass/2)))

# Sig = []
# for i in range(len(m_array)):
#     Sig.append(sigma_cdm(m_array[i]))
# Sig = np.array(Sig)
# sig_inter = intp.interp1d(np.log(m_array),np.log(Sig))
# print('sigma = ',np.exp(sig_inter(np.log(mass))))

# Sig_inter = []
# for m in m_array:
#     Sig_inter.append(np.exp(sig_inter(np.log(m))))
# Sig_inter = np.array(Sig_inter)
# # plt.plot(m_array,Sig,label='original',marker='.',lw=0)
# # plt.plot(m_array,Sig_inter,label='interpolated',marker='.',lw=0)
# plt.plot(m_array,abs(Sig-Sig_inter)/Sig,label='Difference',marker='.',lw=0)
# plt.xlabel('m')
# plt.ylabel(r'$\Delta\sigma/\sigma$')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.savefig('Sigma_comp.png')


# file_name = './Code_own/Data/pk_Mill.txt'
# pk_data   = np.loadtxt(file_name)

# h = 0.73
# z = np.array([0],dtype='float64')
# N_k = 1000
# k_array = np.zeros((N_k, 1, 1),dtype='float64')
# k_array[:,0,0] = np.logspace(-6,3,N_k)

# cosmo = Class()
# cosmo.set({'output':'mPk','P_k_max_h/Mpc':10000})
# cosmo.compute()
# Pk_0 = cosmo.get_pk(k_array*h,z,N_k,1,1)[:,0,0]
# k_0 = k_array[:,0,0]
# np.savetxt('pk_CLASS.txt',[k_0,Pk_0])

# plt.plot(pk_data[:,0],pk_data[:,1],label='FORTRAN')
# plt.plot(k_0,Pk_0*h**3,label='CLASS')
# plt.xlabel('k')
# plt.ylabel('P_k')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.savefig('P_k_comparison.png')


# m_arr = np.linspace(1e8,1e15)
# sig_CLASS = []
# sig_FORT  = []
# for i in range(len(m_arr)):
#     sig_CLASS.append(sigma_cdm(m_arr[i]))
#     sig_FORT.append(scla*spl_intp(m_arr[i]*0.1825**3/0.25)[0])

# plt.plot(m_arr,sig_CLASS,label='CLASS')
# plt.plot(m_arr,sig_FORT,label='FORTRAN')
# plt.xlabel('Mass')
# plt.ylabel('Sigma')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.savefig('Sigma_Test.png')

# alp_CLASS = []
# alp_FORT  = []
# for i in range(len(m_arr)):
#     alp_CLASS.append(alpha(m_arr[i]))
#     alp_FORT.append(spl_intp(m_arr[i]*0.1825**3/0.25)[1])

# plt.plot(m_arr,alp_CLASS,label='CLASS')
# plt.plot(m_arr,alp_FORT,label='FORTRAN')
# plt.xlabel('Mass')
# plt.ylabel('Alpha')
# plt.xscale('log')
# # plt.yscale('log')
# plt.legend()
# plt.savefig('Alpha_Test.png')

# z = np.logspace(-3,3,1000)

# J_u_intp = []
# J_u_og = []
# for i in z:
#     J_u_intp.append(J_unresolved(i))
#     J_u_og.append(J_pre(i))
# J_u_og = np.array(J_u_og)
# J_u_intp = np.array(J_u_intp)

# fig, (ax1,ax2) = plt.subplots(2,sharex=True)
# ax1.plot(z,J_u_intp,label='interpolated')
# ax1.plot(z,J_u_og,label='previous')
# ax2.plot(z,abs(J_u_og-J_u_intp)/J_u_og,marker='.',lw=0)
# plt.xlabel('z')
# ax1.set_ylabel('J(u)')
# ax2.set_ylabel('D_J(u)')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.savefig('J_unresolved.png')

# with h5py.File('./Code_own/Trees/tree_selftestfast_random_masses2.hdf5','a') as f:
#      grp = f.create_group('Header')
#      grp.attrs['LastSnapShotNr'] = 9
#      grp.attrs['Nhalos_ThisFile'] = 120000
#      grp.attrs['Nhalos_ThisFile'] = 1433267
#      grp.attrs['Nhalos_Total'] = 1433267
#      grp.attrs['Ntrees_ThisFile'] = 120000
#      grp.attrs['Ntrees_Total'] = 120000
#      grp.attrs['NumFiles'] = 1
# with h5py.File('./Code_own/Trees/tree_selftestfast_1e14_ytree1.h5','a',libver='latest') as f:
#      if 'Parameters' not in f:
#              f.create_group('Parameters')
#      f['Parameters'].attrs['HubbleParam'] = 0.73
#      f['Parameters'].attrs['Omega0'] = 0.25
#      f['Parameters'].attrs['OmegaLambda'] = 0.75
#      f['Parameters'].attrs['BoxSize'] = 479.0

from tqdm import tqdm
import pandas as pd
import seaborn as sns

# f = h5py.File('./Code_own/Trees/tree_selftestfast_1e14_ytree6.h5')

# time_level = f['TreeHalos/SnapNum'][:]
# masses_lev = f['TreeHalos/Mass'][:]
# masses_lev_1 = []
# time_level_1 = []
# for i in tqdm(range(len(f['TreeHalos/FirstProgenitor'][:]))):
#     index = f['TreeHalos/FirstProgenitor'][:][i]
#     if index != -1:
#         masses_lev_1.append(masses_lev[index])
#         time_level_1.append(time_level[index])
# masses_lev_1 = np.array(masses_lev_1)
# time_level_1 = np.array(time_level_1)

# plt.plot(time_level,masses_lev,marker='.',lw=0)
# plt.plot(time_level_1,masses_lev_1,marker='.',lw=0)
# plt.yscale('log')
# plt.savefig('tree.png')

# df = pd.DataFrame(
#     {
#         'Mass': masses_lev,
#         'level': time_level
#     }
# )
# for cat in df['level'].unique():
#     subset = df[df['level']== cat]
#     plt.hist(subset['Mass'],label=cat)
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')

# g = sns.FacetGrid(df, col='level', height=4)
# g.map(plt.hist, 'Mass',density=True)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('tree_hist.png')

import networkx as nx

def build_tree(h5_file, tree_id=0):
    G = nx.DiGraph()
    with h5py.File(h5_file, 'r') as f:
        # Load tree data (adjust paths as needed)
        descendant = f[f'TreeHalos/Descendant'][:]
        mass = f[f'TreeHalos/Mass'][:]
        
        # Create nodes
        for i in tqdm(range(len(descendant))):
            G.add_node(i, mass=mass[i])
        print('1. here')
        # Create edges
        for i, desc in enumerate(tqdm(descendant)):
            if desc >= 0:  # Valid descendant
                G.add_edge(i, desc)
        print('Then here')
    print('And finally here!')
    return G

# Visualize
G = build_tree('./Code_own/Trees/tree_selftestfast_1e14_ytree12.h5')
pos = nx.nx_agraph.graphviz_layout(G, prog='dot',args='-Grankdir=TB -Gnodesep=0.1')
# pos = nx.planar_layout(G)
# pos = nx.multipartite_layout(G)
# pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
edge_alphas = [0.3 if G.in_degree(v) > 1 else 1.0 for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, alpha=edge_alphas, width=0.3)
nx.draw_networkx_edges(G, pos, connectionstyle='arc3,rad=0.1', arrows=False)
node_sizes = [np.log10(data['mass'])  for _, data in G.nodes(data=True)]
nx.draw(G, pos, node_size=node_sizes, alpha=0.3)
plt.savefig('goodTree3.png')



# import json

# def export_to_d3(h5_file, output_json):
#     data = {"nodes": [], "links": []}
    
#     with h5py.File(h5_file, 'r') as f:
#         mass = f['TreeHalos/Mass'][:]
#         descendant = f['TreeHalos/Descendant'][:]
        
#         for i in range(len(mass)):
#             data["nodes"].append({"id": i, "mass": float(mass[i])})
#             if descendant[i] >= 0:
#                 data["links"].append({"source": i, "target": descendant[i]})
    
#     with open(output_json, 'w') as f:
#         json.dump(data, f)

# export_to_d3('./Code_own/Trees/tree_selftestfast_1e14_ytree7.h5', 'tree.json')
