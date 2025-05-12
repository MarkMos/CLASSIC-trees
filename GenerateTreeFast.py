# from Make_Tree import *
# from Tree_Node_and_Memory import *
# from Moving_in_Tree import *
# from Delta_crit import *
# from sigma_cdm_func import *
from classic_trees import get_tree_vals, functions
from random_masses import ppf_ST, ppf_PS
import numpy as np
import h5py
from multiprocessing import Pool, Lock

lock = Lock()

filename = './CLASSIC-trees/Data/flat.txt'
DELTA = functions(filename)

def tree_process(i,i_seed_0,mp_halo,a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo):
    if mp_halo > 6e14:
        # Safety to ensure that the merger-tree can be calculated.
        n_halo_max=10000000
    count,arr_mhalo,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc,arr_nextprog = get_tree_vals(i,i_seed_0,mp_halo,a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo)
    count = np.array([count],dtype='int_')
    i = np.array([i],dtype='int_')

    return {
        "arr_mhalo": arr_mhalo,
        "arr_nodid": arr_nodid,
        "arr_treeid": arr_treeid,
        "arr_time": arr_time,
        "arr_1prog": arr_1prog,
        "arr_desc": arr_desc,
        "arr_nextprog": arr_nextprog,
        "count": count,
        "tree_index": i
    }

def append_create_dataset(grp,name,data):
    if name in grp:
        dset = grp[name]
        curr_size = dset.shape[0]
        new_size  = curr_size + len(data)
        dset.resize(new_size,axis=0)
        dset[curr_size:] = data
    else:
        grp.create_dataset(name,data=data,maxshape=(None,)+data.shape[1:])

# Parallel execution:
def parallel_exe(j,n_tree,i_seed_0,mp_halo,a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,nth_run,start_offset,file_name,omega_0,l_0,h_0,BoxSize):
    args_list = [(i,i_seed_0,mp_halo[i],a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo)
                  for i in range(j*n_tree,n_tree+j*n_tree)]
    with Pool() as pool:
        results = pool.starmap(tree_process, args_list)
    print('Here')
    with h5py.File(file_name,'a',libver='latest') as f:
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
        if 'Parameters' not in f:
            f.create_group('Parameters')
            f['Parameters'].attrs['HubbleParam'] = h_0 #0.6781
            f['Parameters'].attrs['Omega0'] = omega_0 #0.30988304304812053
            f['Parameters'].attrs['OmegaLambda'] = l_0 #0.6901169569518795
            f['Parameters'].attrs['BoxSize'] = BoxSize
        for result in results:
            append_create_dataset(grp1,'SnapNum',result['arr_time'])
            append_create_dataset(grp1,'SubhaloMass',data=result['arr_mhalo'])
            append_create_dataset(grp1,'TreeDescendant',result['arr_desc'])
            append_create_dataset(grp1,'TreeFirstProgenitor',result['arr_1prog'])
            append_create_dataset(grp1,'NextProgenitor',result['arr_nextprog'])
            append_create_dataset(grp1,'TreeID',data=result['arr_treeid'])
            append_create_dataset(grp1,'TreeIndex',data=result['arr_nodid'])
            append_create_dataset(grp2,'Length',data=result['count'])
            append_create_dataset(grp2,'StartOffset',data=np.array([start_offset]))
            append_create_dataset(grp2,'TreeID',data=result['tree_index'])
            start_offset += result['count']
        return start_offset

def compute_tree_fast(random_mass,
                      mass,
                      file_name,
                      omega_0,
                      l_0,
                      h_0,
                      BoxSize = 479.0,
                      n_tree = 30,
                      i_seed_0 = -8635,
                      a_halo = 1,
                      m_res = 1e8,
                      z_max  = 4,
                      n_lev = 10,
                      n_halo_max = 1000000,
                      n_halo = 1,
                      n_part = 40000):
    if random_mass=='PS':
        u_PS = np.random.rand(int(n_part*n_tree))
        mp_halo = ppf_PS(u_PS)
        mp_halo = np.sort(mp_halo)[::-1]
    elif random_mass=='ST':
        u_ST = np.random.rand(int(n_part*n_tree))
        mp_halo = ppf_ST(u_ST)
        mp_halo = np.sort(mp_halo)[::-1]
    else:
        mp_halo = mass*np.ones(int(n_part*n_tree))
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
    nth_run = False
    start_offset = 0
    for j in range(n_part):
        start_offset = parallel_exe(j,n_tree,i_seed_0,mp_halo,a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,nth_run,start_offset,file_name,omega_0, l_0, h_0,BoxSize)
    with h5py.File(file_name,'a') as f:
        grp = f.create_group('Header')
        grp.attrs['LastSnapShotNr'] = int(n_lev - 1)
        grp.attrs['Nhalos_ThisFile'] = start_offset
        grp.attrs['Nhalos_Total'] = start_offset
        grp.attrs['Ntrees_ThisFile'] = int(n_tree*n_part)
        grp.attrs['Ntrees_Total'] = int(n_tree*n_part)
        grp.attrs['NumFiles'] = 1