# from Make_Tree import *
# from Tree_Node_and_Memory import *
# from Moving_in_Tree import *
# from Delta_crit import *
# from sigma_cdm_func import *
from classic_trees import get_tree_vals, functions
# from random_masses import ppf_ST, ppf_PS
import numpy as np
import h5py
from multiprocessing import Pool, Lock

lock = Lock()

filename = './CLASSIC-trees/Data/flat.txt'
DELTA = functions(filename)

def tree_process(i,i_seed_0,mp_halo,a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,scaling):
    vel_base = np.random.lognormal(np.log(200),0.7,3)
    if mp_halo > 6e14:
        # Safety to ensure that the merger-tree can be calculated.
        n_halo_max=10000000
    count,arr_mhalo,arr_Vmax,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc,arr_nextprog,arr_pos,arr_velo,arr_spin = get_tree_vals(i,i_seed_0,mp_halo,a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,scaling)
    count = np.array([count],dtype='int_')
    i = np.array([i],dtype='int_')

    return {
        'arr_mhalo': arr_mhalo,
        'arr_Vmax': arr_Vmax,
        'arr_nodid': arr_nodid,
        'arr_treeid': arr_treeid,
        'arr_time': arr_time,
        'arr_1prog': arr_1prog,
        'arr_desc': arr_desc,
        'arr_nextprog': arr_nextprog,
        'arr_pos': arr_pos,
        'arr_velo': arr_velo,
        'arr_spin': arr_spin,
        'count': count,
        'tree_index': i
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
def parallel_exe(j,n_tree,i_seed_0,mp_halo,a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,nth_run,start_offset,file_name,omega_0,l_0,h_0,BoxSize,mode,pos_base,vel_base,scaling):
    args_list = [(i,i_seed_0,mp_halo[i],a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,scaling)
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
            append_create_dataset(grp1,'SubhaloVmax',result['arr_Vmax'])
            append_create_dataset(grp1,'TreeDescendant',result['arr_desc'])
            append_create_dataset(grp1,'TreeFirstProgenitor',result['arr_1prog'])
            append_create_dataset(grp1,'NextProgenitor',result['arr_nextprog'])
            append_create_dataset(grp1,'SubhaloPos',result['arr_pos'])
            append_create_dataset(grp1,'SubhaloVelo',result['arr_velo'])
            append_create_dataset(grp1,'SubhaloSpin',result['arr_spin'])
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
                      n_part = 40000,
                      times = 'equal a',
                      mode='FoF',
                      pos_base = np.array([0,0,0],dtype=np.float64),
                      vel_base = np.array([10,10,10],dtype=np.float64),
                      scaling = 0.5):
    '''
    Function to call the routines of classic_trees for huge numbers of trees to 
    compute. Ideally to produce large merger tree files.
    ----------------------
    Input:
        random_mass : Distribution to draw the mass of the base node(s) of merger tree(s)
        mass        : Mass of the base node of a merger tree (only if random_mass=None)
        file_name   : Name of hdf5-file
        omega_0     : Relative density of matter in the universe
        l_0         : Relative cosmological constant
        h_0         : Reduced Hubble-parameter
        BoxSize     : Size of the volume
        n_tree      : Number of trees that are computed in one Pool
        i_seed_0    : Used for seed to generate random numbers
        a_halo      : Value of scale factor today (default) or up to which time the tree is calculated
        m_res       : Mass resolution limit; minimum mass
        z_max       : Maximum redshift for lookback
        n_lev       : Number of time levels
        n_halo_max  : Maximum number of nodes per tree; used for preallocation 
        n_halo      : Start of counter of nodes inside the tree(s)
        n_part      : Number of runs of a Pool
        times       : Either equally spaced times in z or a, or a custom array of z or a
        mode    	: Defining the usage of the merger tree.
        pos_base    : Initial 3 position of base node
        velo_base   : Initial 3 velocity of base node
        scaling     : Factor of scattering of positional change over time
    ----------------------
    Output:
        hdf5-file with values of random or constant mass merger trees
    '''
    vel_base = np.random.lognormal(np.log(200),0.7,3)
    if random_mass=='PS':
        from random_masses import ppf_PS
        u_PS = np.random.rand(int(n_part*n_tree))
        mp_halo = ppf_PS(u_PS)
        mp_halo = np.sort(mp_halo)[::-1]
    elif random_mass=='ST':
        from random_masses import ppf_ST
        u_ST = np.random.rand(int(n_part*n_tree))
        mp_halo = ppf_ST(u_ST)
        mp_halo = np.sort(mp_halo)[::-1]
    else:
        mp_halo = mass*np.ones(int(n_part*n_tree))
    if type(times)==str and times=='equal z':
        a_lev = []
        w_lev = []
        for i_lev in range(n_lev):
            a_lev.append(1/(1 + z_max*(i_lev)/(n_lev-1)))
            d_c = DELTA.delta_crit(a_lev[i_lev])
            w_lev.append(d_c)
            print('z = ',1/a_lev[i_lev]-1,' at which delta_crit = ',d_c)
        a_lev = np.array(a_lev)
        w_lev = np.array(w_lev)
    elif type(times)==str and times=='equal a':
        a_lev = np.linspace(1,1/(z_max+1),n_lev)
        w_lev = []
        for i_lev in range(n_lev):
            # a_lev.append(1/(1+1/(z_max+1)*i_lev/(n_lev-1)))
            d_c = DELTA.delta_crit(a_lev[i_lev])
            w_lev.append(d_c)
            print('z = ',1/a_lev[i_lev]-1,' at which delta_crit = ',d_c)
        a_lev = np.array(a_lev)
        w_lev = np.array(w_lev)
    elif type(times)!=str and len(times)>1:
        n_lev = int(len(times))
        if np.any(times>1):
            a_lev = []
            w_lev = []
            for z in times:
                a_temp = 1/(z+1)
                a_lev.append(a_temp)
                d_c = DELTA.delta_crit(a_temp)
                w_lev.append(d_c)
                print('z = ',1/a_temp-1,' at which delta_crit = ',d_c)
            a_lev = np.array(a_lev)
            w_lev = np.array(w_lev)
        else:
            a_lev = times
            w_lev = []
            for a in a_lev:
                d_c = DELTA.delta_crit(a)
                w_lev.append(d_c)
                print('z = ',1/a-1,' at which delta_crit = ',d_c)
            a_lev = np.array(a_lev)
            w_lev = np.array(w_lev)
    a_lev = np.array(a_lev)
    w_lev = np.array(w_lev)
    nth_run = False
    start_offset = 0
    for j in range(n_part):
        start_offset = parallel_exe(j,n_tree,i_seed_0,mp_halo,a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,nth_run,start_offset,file_name,omega_0, l_0, h_0,BoxSize,mode,pos_base,vel_base,scaling)
    with h5py.File(file_name,'a') as f:
        grp = f.create_group('Header')
        grp.attrs['LastSnapShotNr'] = int(n_lev - 1)
        grp.attrs['Nhalos_ThisFile'] = start_offset
        grp.attrs['Nhalos_Total'] = start_offset
        grp.attrs['Ntrees_ThisFile'] = int(n_tree*n_part)
        grp.attrs['Ntrees_Total'] = int(n_tree*n_part)
        grp.attrs['NumFiles'] = 1