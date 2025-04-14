# from Make_Tree import *
# from Tree_Node_and_Memory import *
# from Moving_in_Tree import *
# from Delta_crit import *
# from sigma_cdm_func import *
from classic_trees import functions, get_tree_vals
import numpy as np
# import h5py
import time

filename = './CLASSIC-trees/Data/flat.txt'
DELTA = functions(filename)

nth_run = False
def append_create_dataset(grp,name,data):
    if name in grp:
        dset = grp[name]
        curr_size = dset.shape[0]
        new_size  = curr_size + len(data)
        dset.resize(new_size,axis=0)
        dset[curr_size:] = data
    else:
        grp.create_dataset(name,data=data,maxshape=(None,)+data.shape[1:])


n_lev = 10
n_halo_lev = 10

mp_halo = 1e13
m_res   = 1e8
n_tree  = 1

G_0=0.57
gamma_1=0.38
gamma_2=-0.01
eps_1=0.1
eps_2=0.1

omega_0=0.25
lambda_0=0.75
h_0=0.73
omega_b=0.04
Gamma=omega_0*h_0

n_spec = 1
dn_dlnk = 0
k_ref = 1

sigma_8 = 0.9

i_err = 1
n_halo_max = int(1000000)
n_halo = 1
i_seed_0 = -8635
i_seed = i_seed_0

a_halo = 1
z_max  = 4

a_lev = []
for i_lev in range(1,n_lev+1):
    a_lev.append(1/(1 + z_max*(i_lev-1)/(n_lev-1)))
    #print(a_lev)
    d_c = DELTA.delta_crit(a_lev[i_lev-1])
    print('z = ',1/a_lev[i_lev-1]-1,' at which delta_crit = ',d_c)
a_lev = np.array(a_lev)
jp_halo = []
n_frag_max = 10
start_offset = 0
for i in range(n_tree):
    start = time.time()
    count,arr_mhalo,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc = get_tree_vals(i,i_seed_0,mp_halo,a_halo,m_res,a_lev,n_lev,n_halo_max,n_halo)
    print(count-len(np.where(arr_mhalo==0)[0]))
    end = time.time()
    print(f"Elapsed time: {end - start} seconds")