from mpi4py import MPI
import classic_trees as ct
import sys
import importlib.resources
import h5py
import numpy as np

n_tree = 40

m_min = 1e10
m_max = 1e14
m_res = 1e9

h_0 = 0.6781
omega_0 = 0.309974
l_0 = 0.690026
file_name = 'TestTree.hdf5'

a = np.array([0.0625423207,0.0697435265,0.0770403364,0.0819350757,0.0871408012,0.0935597132,0.0999766069,0.10532572,0.110961031,0.116345263,0.121414094,0.126703761,0.132223883,0.137332233,0.142637941,0.14814863,0.153872219,0.160575993,0.166779702,0.172404243,0.178218468,0.183357906,0.188645553,0.194085686,0.1996827,0.20544112,0.2113656,0.216432969,0.222674431,0.229095883,0.235702516,0.24249967,0.249492839,0.256687676,0.262841616,0.269143094,0.275595646,0.282202894,0.288968547,0.295896403,0.30299035,0.31025437,0.317692541,0.325309039,0.333108137,0.341094214,0.349271753,0.357645343,0.366219686,0.374999594,0.383989995,0.393195935,0.402622583,0.412275229,0.422159292,0.434333459,0.442643994,0.455408895,0.466327063,0.477506988,0.488954945,0.50067736,0.512680813,0.524972043,0.537557948,0.550445592,0.56364221,0.57715521,0.590992177,0.605160876,0.619669262,0.634525479,0.649737864,0.665314958,0.681265504,0.697598455,0.714322979,0.731448464,0.748984523,0.770583627,0.789057929,0.807975142,0.827345884,0.847181028,0.867491709,0.888289327,0.909585556,0.93139235,0.953721949,0.976586888,1.0])
times = a[::-1][:]

if float(sys.version[2:6])<9:
    filename = str(importlib.resources.path('classic_trees.Data', 'flat.txt'))
else:
    filename = str(importlib.resources.files('classic_trees.Data').joinpath('flat.txt'))
DELTA = ct.functions(filename)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

merger_tree = ct.forrest()
merger_tree.set(pk_method='file',file='./PowerSpectra/pk_class.txt',
                cosmo_params={'h':0.6781,'Omega_m':0.309974,'Omega_Lambda':0.690026})
comm.Barrier()
Boxsize = 50.0
# if rank==0:
#     a_lev = times
#     w_lev = []
#     for a in a_lev:
#         d_c = DELTA.delta_crit(a)
#         w_lev.append(d_c)
#         print('z = ',1/a-1,' at which delta_crit = ',d_c)
#     a_lev = np.array(a_lev)
#     w_lev = np.array(w_lev)

#     u_ST = np.random.rand(int(n_tree))
#     ppf_ST = ct.random_masses(w_lev[0]).random_ST(m_min,m_max)
#     mp_halo = ppf_ST(u_ST)
#     mp_halo = np.sort(mp_halo)[::-1]

def run_classic_trees(m,i,i_seed_0,n_lev):
    theta = 2*np.pi*np.random.uniform(0,1)
    u = 2*np.random.uniform(0,1)-1
    norm_vel = np.array([np.sqrt(1-u**2)*np.cos(theta),np.sqrt(1-u**2)*np.sin(theta),u])
    vel_base = np.random.lognormal(np.log(200),0.7*(m/1e4)**(-0.1))*norm_vel
    pos_base = np.random.uniform(0,Boxsize,3)
    count,arr_mhalo,arr_Vmax,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc,arr_nextprog,arr_1FoF,arr_nextFoF,arr_pos,arr_velo,arr_spin,arr_GroupMass,arr_sublen = ct.get_tree_vals_FoF(i,i_seed_0,m,1,m_min,m_res,w_lev,a_lev,n_lev,10000000,1,pos_base,vel_base,0.3)
    arr_vel_disp = np.zeros(np.sum(count),dtype=np.float32)
    count = np.array(count,dtype='int_')
    i = np.array([i],dtype='int_')

    return {
        'arr_mhalo': arr_mhalo/1e10,
        'arr_Vmax': arr_Vmax,
        'arr_nodid': arr_nodid,
        'arr_treeid': arr_treeid,
        'arr_time': n_lev-1-arr_time,
        'arr_1prog': arr_1prog,
        'arr_desc': arr_desc,
        'arr_nextprog': arr_nextprog,
        'arr_1FoF': arr_1FoF,
        'arr_nextFoF': arr_nextFoF,
        'arr_pos': arr_pos,
        'arr_velo': arr_velo,
        'arr_spin': arr_spin,
        'arr_GroupMass': arr_GroupMass/1e10,
        'arr_sublen': arr_sublen,
        'arr_vel_disp': arr_vel_disp,
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

if __name__ == "__main__":
    comm.Barrier()
    
    # Global variables for all ranks
    a_lev = None
    w_lev = None
    mp_halo = None
    
    if rank == 0:
        print("Rank 0: Computing cosmology data...")
        a_lev = times
        w_lev = np.array([DELTA.delta_crit(a) for a in a_lev])
        print("Rank 0: Generating halo masses...")
        u_ST = np.random.rand(n_tree)
        ppf_ST = ct.random_masses(w_lev[0]).random_ST(m_min, m_max)
        mp_halo = np.sort(ppf_ST(u_ST))[::-1]
    
    # Broadcast shared data
    a_lev = comm.bcast(a_lev, root=0)
    w_lev = comm.bcast(w_lev, root=0)
    mp_halo = comm.bcast(mp_halo, root=0)
    
    if rank == 0:
        print(f"Distributing {n_tree} trees across {size} ranks...")
    
    # Each rank gets its chunk of trees
    start_tree = int(rank * n_tree / size)
    end_tree = int((rank + 1) * n_tree / size)
    # CRITICAL: Rank 0 initializes file FIRST
    if rank == 0:
        with h5py.File(file_name, 'w', libver='latest') as f:
            f.create_group('TreeHalos')
            f.create_group('TreeTable') 
            f.create_group('TreeTimes')
            f.create_group('Parameters')
            f['Parameters'].attrs['HubbleParam'] = h_0
            f['Parameters'].attrs['Omega0'] = omega_0
            f['Parameters'].attrs['OmegaLambda'] = l_0
            f['Parameters'].attrs['BoxSize'] = Boxsize
            f['TreeTimes'].create_dataset('Redshift', data=1/a_lev-1)
            f['TreeTimes'].create_dataset('Time', data=a_lev)
    
    comm.Barrier()  # All ranks wait for file init

    local_offset = 0
    if rank > 0:
        prev_counts = comm.recv(source=rank-1, tag=0)
        local_offset = sum(prev_counts)

    with h5py.File(file_name, 'a', driver='mpio', comm=comm,libver='latest') as f:
        grp1 = f['TreeHalos']
        grp2 = f['TreeTable']
        
        local_tree_count = []
        # Each rank writes its trees sequentially
        for i in range(start_tree, end_tree):
            print(f"Rank {rank}: Tree {i}/{n_tree-1}")
            result = run_classic_trees(mp_halo[i], i, -8635, len(w_lev))
            
            # Write TreeHalos immediately
            append_create_dataset(grp1,'SnapNum',np.array(result['arr_time'],dtype=np.int32))
            append_create_dataset(grp1,'SubhaloMass',data=np.array(result['arr_mhalo'],dtype=np.float32))
            append_create_dataset(grp1,'SubhaloVmax',np.array(result['arr_Vmax'],dtype=np.float32))
            append_create_dataset(grp1,'TreeDescendant',np.array(result['arr_desc'],dtype=np.int32))
            append_create_dataset(grp1,'TreeFirstProgenitor',np.array(result['arr_1prog'],dtype=np.int32))
            append_create_dataset(grp1,'TreeNextProgenitor',np.array(result['arr_nextprog'],dtype=np.int32))
            append_create_dataset(grp1,'TreeFirstHaloInFOFgroup',np.array(result['arr_1FoF'],dtype=np.int32))
            append_create_dataset(grp1,'TreeNextHaloInFOFgroup',np.array(result['arr_nextFoF'],dtype=np.int32))
            append_create_dataset(grp1,'Group_M_Crit200',np.array(result['arr_GroupMass'],dtype=np.float32))
            append_create_dataset(grp1,'SubhaloPos',np.array(result['arr_pos'],dtype=np.float32))
            append_create_dataset(grp1,'SubhaloVel',np.array(result['arr_velo'],dtype=np.float32))
            append_create_dataset(grp1,'SubhaloSpin',np.array(result['arr_spin'],dtype=np.float32))
            append_create_dataset(grp1,'SubhaloLen',np.array(result['arr_sublen'],dtype=np.int32))
            append_create_dataset(grp1,'SubhaloVelDisp',result['arr_vel_disp'])
            append_create_dataset(grp1,'TreeID',data=result['arr_treeid'])
            append_create_dataset(grp1,'TreeIndex',data=np.array(result['arr_nodid'],dtype=np.int32))
            append_create_dataset(grp2, 'Length', np.array(result['count'],dtype=np.int32))
            append_create_dataset(grp2, 'StartOffset', np.array([local_offset]))
            append_create_dataset(grp2, 'TreeID', result['tree_index'])

            local_offset += np.sum(result['count'])
            local_tree_count.append(np.sum(result['count']))
            
        # Send count to next rank
        if rank < size - 1:
            comm.send(local_tree_count, dest=rank+1, tag=0)
    
    comm.Barrier()
    if rank == 0:
        print(f"✓ Generated {n_tree} UNIQUE trees across {size} ranks")
