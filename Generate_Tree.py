# from Make_Tree import *
# from Tree_Node_and_Memory import *
# from Moving_in_Tree import *
# from Delta_crit import *
# from sigma_cdm_func import *
from classic_trees import functions, get_tree_vals
from random_masses import ppf_PS, ppf_ST
import numpy as np
import h5py
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

mp_halo = 1e12
m_res   = 1e8
n_tree  = 1

# u_PS = np.random.rand(n_tree)
# m_halos = ppf_PS(u_PS)
# print(type(m_halos))

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
w_lev = []
for i_lev in range(1,n_lev+1):
    a_lev.append(1/(1 + z_max*(i_lev-1)/(n_lev-1)))
    #print(a_lev)
    d_c = DELTA.delta_crit(a_lev[i_lev-1])
    w_lev.append(d_c)
    print('z = ',1/a_lev[i_lev-1]-1,' at which delta_crit = ',d_c)
a_lev = np.array(a_lev)
w_lev = np.array(w_lev)
jp_halo = []
n_frag_max = 10
start_offset = 0
start = time.time()
for i in range(n_tree):
    count,arr_mhalo,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc,arr_nextprog = get_tree_vals(i,i_seed_0,mp_halo,a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo)
    with h5py.File('./Code_own/Trees/tree_selftestfast_1e12_ytree.h5','a') as f:
        # Create or access groups of the merger tree file
        if 'TreeHalos' not in f:
            grp1 = f.create_group('TreeHalos')
        else:
            grp1 = f['TreeHalos']
            nth_run = True
        if 'TreeTable' not in f:
            grp2 = f.create_group('TreeTable')
        else:
            grp2 = f['TreeTable']
        
        if nth_run is False:
            grp3 = f.create_group('TreeTimes')
            d_red = grp3.create_dataset('Redshift',data=1/a_lev-1)
            d_time= grp3.create_dataset('Time',data=a_lev)
        
        append_create_dataset(grp1,'SnapNum',arr_time)
        append_create_dataset(grp1,'Mass',data=arr_mhalo)
        append_create_dataset(grp1,'Descendant',arr_desc)
        append_create_dataset(grp1,'FirstProgenitor',arr_1prog)
        append_create_dataset(grp1,'NextProgenitor',arr_nextprog)
        append_create_dataset(grp1,'TreeID',data=arr_treeid)
        append_create_dataset(grp1,'TreeIndex',data=arr_nodid)
        append_create_dataset(grp2,'Length',data=np.array([count]))
        append_create_dataset(grp2,'StartOffset',data=np.array([start_offset]))
        append_create_dataset(grp2,'TreeID',data=np.array([i]))
        grp = f.create_group('Header')
        grp.attrs['LastSnapShotNr'] = 9
        grp.attrs['Nhalos_ThisFile'] = 120000
        grp.attrs['Nhalos_ThisFile'] = 1433267
        grp.attrs['Nhalos_Total'] = 1433267
        grp.attrs['Ntrees_ThisFile'] = 120000
        grp.attrs['Ntrees_Total'] = 120000
        grp.attrs['NumFiles'] = 1
        if 'Parameters' not in f:
            f.create_group('Parameters')
            f['Parameters'].attrs['HubbleParam'] = 0.73
            f['Parameters'].attrs['Omega0'] = 0.25
            f['Parameters'].attrs['OmegaLambda'] = 0.75
            f['Parameters'].attrs['BoxSize'] = 479.0
    # with h5py.File('ytree_compatible.h5', 'w') as f:
    #     # Required groups
    #     tree = f.create_group('Tree0')
        
    #     # Mandatory fields (int32)
    #     tree.create_dataset('Descendant', data=arr_desc.astype('int32'))
    #     tree.create_dataset('FirstProgenitor', data=arr_1prog.astype('int32'))
    #     tree.create_dataset('NextProgenitor', data=arr_nextprog.astype('int32'))
    #     tree.create_dataset('SnapNum', data=arr_time.astype('int32'))
        
    #     # Physical fields
    #     tree.create_dataset('Mass', data=arr_mhalo.astype('float32'))
        
    #     # Header
    #     header = f.create_group('Header')
    #     header.attrs['BoxSize'] = 479.0  # Mpc/h
    #     header.attrs['HubbleParam'] = 0.73
    #     header.attrs['Omega0'] = 0.25
    #     header.attrs['OmegaLambda'] = 0.75
    start_offset += count
end = time.time()
print(f"Elapsed time: {end - start} seconds")