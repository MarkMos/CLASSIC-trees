# from sigma_cdm_func import *
# from classy import Class
import numpy as np
# import scipy.interpolate as intp
import matplotlib.pyplot as plt
# from alpha_func import *
# from halo_velo_func import sig_halo_func
# from functions import *
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
# print('alpha = ', alpha(np.log(mass/2)))
from tqdm import tqdm

# m_array = np.logspace(8,15,3000)

# Sig = []
# for i in tqdm(range(len(m_array))):
#     Sig.append(sigma_cdm(m_array[i]))
# Sig = np.array(Sig)
# sig_inter = intp.interp1d(np.log(m_array),np.log(Sig))
# print('sigma = ',np.exp(sig_inter(np.log(mass))))

# cosmo = Class()
# cosmo.set({'output':'mPk','P_k_max_h/Mpc':10000,'h':0.67810})
# cosmo.compute()

# Sig_Class = []

# for m in m_array:
#     R = (3*m/(4*np.pi*rho_crit))**(1/3)
#     Sig_Class.append(cosmo.sigma(R,0,h_units=True))
# print(rho_critical,' = ',rho_crit)
# Sig_Class = np.array(Sig_Class)
# plt.figure(figsize=(15,10))
# plt.plot(m_array,Sig,label='original',marker='.',lw=0)
# plt.plot(m_array,Sig_Class,label='Class',marker='.',lw=0)
# plt.plot(m_array,abs(Sig-Sig_Class)/Sig,label='Difference',marker='.',lw=0)
# plt.xlabel('m')
# plt.ylabel(r'$\Delta\sigma/\sigma$')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.savefig('Sigma_comp_w_Class.png')
# m_rough = np.logspace(7,16,1500)
# logSig_Class = []
# for m in m_rough:
#     R = (3*m/(4*np.pi*rho_crit))**(1/3)
#     logSig_Class.append(np.log(cosmo.sigma(R,0,h_units=True)))
# log_M = np.log(m_rough)

# interp = intp.UnivariateSpline(log_M,logSig_Class,k=4,s=1)
# def alpha_cl(m):
#     alpha_ret = interp.derivative(n=1)
#     return alpha_ret(np.log(m))

# Alpha = []
# Alpha_Class = []
# for m in m_array:
#     Alpha.append(alpha(m))
#     Alpha_Class.append(alpha_cl(m))
# Alpha = np.array(Alpha)
# Alpha_Class = np.array(Alpha_Class)

# plt.plot(m_array,Alpha,label='original',marker='.',lw=0)
# plt.plot(m_array,Alpha_Class,label='Class',marker='.',lw=0)
# plt.plot(m_array,abs(Alpha-Alpha_Class)/abs(Alpha),label='Difference',marker='.',lw=0)
# plt.xlabel('m')
# plt.ylabel(r'$\Delta\alpha/\alpha$')
# plt.ylabel(r'$\alpha$')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.savefig('Alpha_comp_w_Class.png')


# file_name = './Code_own/Data/pk_Mill.txt'
# pk_data   = np.loadtxt(file_name)

# h = 0.73
# z = np.array([0],dtype='float64')
# N_k = 1000
# k_array = np.zeros((N_k, 1, 1),dtype='float64')
# k_array[:,0,0] = np.logspace(-6,3,N_k)

# cosmo = Class()
# cosmo.set({'output':'mPk','P_k_max_h/Mpc':10000,'h':h,'Omega_m':0.25})
# cosmo.compute()
# Pk_0 = cosmo.get_pk(k_array*h,z,N_k,1,1)[:,0,0]
# k_0 = k_array[:,0,0]
# np.savetxt('./CLASSIC-trees/pk_CLASS_h_73.txt',[k_0,Pk_0])
# print(k_0-pk_data[:,0])
# plt.plot(pk_data[:,0],pk_data[:,1],label='FORTRAN')
# plt.plot(k_0,Pk_0*h**3,label='CLASS')
# plt.plot(k_0,abs(pk_data[:,1]-Pk_0)/pk_data[:,1],marker='.',lw=0,label='Diff.')
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

# from tqdm import tqdm

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

# import networkx as nx

# def build_tree(h5_file, tree_id=0):
#     G = nx.DiGraph()
#     with h5py.File(h5_file, 'r') as f:
#         # Load tree data (adjust paths as needed)
#         descendant = f[f'TreeHalos/Descendant'][:]
#         mass = f[f'TreeHalos/Mass'][:]
        
#         # Create nodes
#         for i in tqdm(range(len(descendant))):
#             G.add_node(i, mass=mass[i])
#         print('1. here')
#         # Create edges
#         for i, desc in enumerate(tqdm(descendant)):
#             if desc >= 0:  # Valid descendant
#                 G.add_edge(i, desc)
#         print('Then here')
#     print('And finally here!')
#     return G

# Visualize
# G = build_tree('./Code_own/Trees/tree_selftestfast_1e10_ytree1.h5')
# pos = nx.nx_agraph.graphviz_layout(G, prog='dot',args='-Grankdir=TB -Gnodesep=0.1')
# pos = nx.planar_layout(G)
# pos = nx.multipartite_layout(G)
# pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
# edge_alphas = [0.3 if G.in_degree(v) > 1 else 1.0 for u, v in G.edges()]
# nx.draw_networkx_edges(G, pos, alpha=edge_alphas, width=0.3)
# nx.draw_networkx_edges(G, pos, connectionstyle='arc3,rad=0.1')
# node_sizes = [np.log10(data['mass'])  for _, data in G.nodes(data=True)]
# nx.draw(G, pos, node_size=node_sizes, alpha=0.3)
# plt.savefig('goodTree_1e10_longer.png')



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

# print(np.sqrt(sig_halo_func(1e15)))

import CLASSIC_trees as ct
# import numpy as np
from classic_trees import set_trees

file = 'TreeTest_FoF_random_20000_diffNsub.hdf5'
a = np.array([0.0625423207,0.0697435265,0.0770403364,0.0819350757,0.0871408012,0.0935597132,0.0999766069,0.10532572,0.110961031,0.116345263,0.121414094,0.126703761,0.132223883,0.137332233,0.142637941,0.14814863,0.153872219,0.160575993,0.166779702,0.172404243,0.178218468,0.183357906,0.188645553,0.194085686,0.1996827,0.20544112,0.2113656,0.216432969,0.222674431,0.229095883,0.235702516,0.24249967,0.249492839,0.256687676,0.262841616,0.269143094,0.275595646,0.282202894,0.288968547,0.295896403,0.30299035,0.31025437,0.317692541,0.325309039,0.333108137,0.341094214,0.349271753,0.357645343,0.366219686,0.374999594,0.383989995,0.393195935,0.402622583,0.412275229,0.422159292,0.434333459,0.442643994,0.455408895,0.466327063,0.477506988,0.488954945,0.50067736,0.512680813,0.524972043,0.537557948,0.550445592,0.56364221,0.57715521,0.590992177,0.605160876,0.619669262,0.634525479,0.649737864,0.665314958,0.681265504,0.697598455,0.714322979,0.731448464,0.748984523,0.770583627,0.789057929,0.807975142,0.827345884,0.847181028,0.867491709,0.888289327,0.909585556,0.93139235,0.953721949,0.976586888,1.0])
a = a[::-1][:10]


tree = ct.trees()
tree.set(pk_method='default') #,add_cosmo_params={'N_ncdm':1})#,cosmo_params={'h':0.8,'Omega_m':0.15,'Omega_Lambda':0.85})
set_trees(tree)
tree.compute_slow(mass=1e17,times=a,m_res=1e11)
# tree.compute_fast(random_mass='ST',times=a,file_name=file,n_part=500,n_tree=40)
# tree.compute_fast(mass=1e14,times=a,n_halo_max=100000,file_name=file,n_part=500,n_tree=20)


# f = h5py.File(file)

# mass = f['TreeHalos/SubhaloMass'][:]
# subID = f['TreeHalos/TreeDescendant'][:]
# TreeID_short = f['TreeTable/TreeID'][:]
# TreeID_long = f['TreeHalos/TreeID'][:]
# SnapNum = f['TreeHalos/SnapNum'][:]

# m_arr = []
# FoFs_arr = []

# for id in tqdm(TreeID_short):
#     indx = np.where(id==TreeID_long)[0]
#     if len(indx)>1:
#         sub1 = subID[indx]
#         mass1= mass[indx]
#         FoFs = len(np.where(0==sub1)[0])
#         m_arr.append(mass1[0])
#         FoFs_arr.append(FoFs)
#     else:
#         continue
# plt.scatter(m_arr,FoFs_arr,color='r')
# plt.xlabel(r'$M$')
# plt.ylabel(r'$N_{child}$')
# plt.grid()
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('N_FoFOMcen10000_2.png')




# tree2 = ct.trees()
# tree2.set(pk_method='default')
# print(np.all(abs(tree.Pk_0_np-tree2.Pk_0_np)/tree2.Pk_0_np<1e-2))

# import ytree
# a = ytree.load(file)
# p = ytree.TreePlot(a[0],dot_kwargs={'rankdir': 'LR', 'size': '"12,4"'})
# p.save('A_ytree1e11_withFortranOrderingLonger.png')


# f = h5py.File('TreeTest_FoF_1e12_20000.hdf5')
# # comp = [-1.9,-1.7,-1.5,-1.3,-1.1,-9e-1,-7e-1,-5e-1,-3e-1,-1e-1,1e-1,3e-1,5e-1,7e-1]

# # comp_y = 10**(np.array([-2.555,-1.627,-1.368,-1.274,-1.186,-1.083,-9.456e-1,-7.429e-1,-2.967e-1, 4.759e-1,-9.140e-1,-2.206,-3.438,-1.000e+1]))
# # indx = np.where(f['TreeHalos/SnapNum']==1)[0]
# coll = []
# for i in f['TreeTable/StartOffset'][:]:
#     first_1= f['TreeHalos/TreeFirstProgenitor'][i]
#     coll.append(np.log10(f['TreeHalos/SubhaloMass'][i+first_1]/f['TreeHalos/SubhaloMass'][i]))
# plt.hist(coll,bins=50,density=True,label='Our')
# # plt.plot(comp[:-2],comp_y[:-2])
# coll_fortran = np.log10(np.loadtxt('/home/markus/code/Fortran_MassOver1e12.txt'))
# plt.hist(coll_fortran,bins=50,density=True,label='Fortran')
# # plt.xscale('log')
# plt.yscale('log')
# plt.grid()
# plt.xlabel(r'$\log(M_1/M_2)$')
# plt.ylabel(r'$\log(f_{cmf})$')
# plt.legend(title=r'For $M_2 = 1e12$ at $z = 0.444$')
# plt.savefig('TreeTest_M1M2_1e12_3.png')


# plt.show()
# print(np.log10(f['TreeHalos/SubhaloMass'][first_1]/f['TreeHalos/SubhaloMass'][0]))
# from tqdm import tqdm

# f = h5py.File(file)

# TreeID = f['TreeHalos/TreeID'][:]
# FoF1_tot = f['TreeHalos/TreeFirstHaloInFOFgroup'][:]
# SnapNum = f['TreeHalos/SnapNum'][:]

# Nprog = []
# N_1pr = []
# for i in tqdm(f['TreeTable/TreeID'][:]):
#     indx = np.where(TreeID==i)
#     FoF1 = FoF1_tot[indx]
#     first_1 = FoF1[1]
#     indx1 = np.where(SnapNum[indx]==1)[0]
#     n_pr = len(indx1)
#     Nprog.append(n_pr)
#     indx2 = np.where(FoF1==first_1)[0]
#     n_1 = len(indx2)
#     N_1pr.append(n_1/n_pr)
# # print(len(N_1pr),' = ',len(Nprog))
# plt.scatter(Nprog,N_1pr)
# # plt.xscale('log')
# plt.yscale('log')
# plt.grid()
# plt.xlabel(r'$N_{prog}$')
# plt.ylabel(r'$N_{1 FoF}/N_{prog}$')
# plt.savefig('TreeTest_NFoF1Nprog_random_5.png')
# plt.show()


# f = h5py.File(file)

# mass = f['TreeHalos/SubhaloMass'][:]
# Vmax = f['TreeHalos/SubhaloVmax'][:]

# plt.hexbin(mass,Vmax,xscale='log',yscale='log',bins='log')
# plt.grid()
# # plt.xscale('log')
# # plt.yscale('log')
# plt.xlabel(r'$M$')
# plt.ylabel(r'$V_{max}$')
# plt.show()

# f = h5py.File('/home/markus/TreeTest_FoF_1e12_20000.hdf5')

# length = f['TreeTable/Length'][:]

# plt.hist(length,bins=50,density=True)
# plt.show()