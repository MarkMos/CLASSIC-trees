from .module import functions, get_tree_vals, get_tree_vals_FoF,random_masses
import numpy as np
import h5py

filename = './Data/flat.txt'
DELTA = functions(filename)

def append_create_dataset(grp,name,data):
    if name in grp:
        dset = grp[name]
        curr_size = dset.shape[0]
        new_size  = curr_size + len(data)
        dset.resize(new_size,axis=0)
        dset[curr_size:] = data
    else:
        grp.create_dataset(name,data=data,maxshape=(None,)+data.shape[1:])


def compute_tree(mass,
                 n_halo_max,
                 random_mass,
                 file_name,
                 omega_0,
                 l_0,
                 h_0,
                 BoxSize = 479.0,
                 n_lev = 10,
                 m_res = 1e8,
                 m_min = 1e11,
                 m_max = 1e16,
                 n_tree = 1,
                 n_halo = 1,
                 i_seed_0 = -8635,
                 a_halo = 1,
                 z_max = 4,
                 times = 'equal a',
                 mode = 'FoF',
                 pos_base = np.array([0,0,0],dtype=np.float64),
                 vel_base = np.array([0,0,0],dtype=np.float64),
                 scaling = 0.5,
                 verbose = 0):
    '''
    Function to call the routines of classic_trees for small numbers of trees to 
    compute. Ideally to see what different starting values yield.
    ----------------------
    Input:
        mass        : Mass of the base node of a merger tree (only if random_mass=None)
        random_mass : Distribution to draw the mass of the base node(s) of merger tree(s)
        file_name   : Name of hdf5-file (optional)
        omega_0     : Relative density of matter in the universe
        l_0         : Relative cosmological constant
        h_0         : Reduced Hubble-parameter
        BoxSize     : Size of the volume
        n_lev       : Number of time levels
        m_res       : Mass resolution limit; minimum mass
        n_tree      : Number of trees that are computed
        n_halo      : Start of counter of nodes inside the tree(s)
        i_seed_0    : Used for seed to generate random numbers
        a_halo      : Value of scale factor today (default) or up to which time the tree is calculated
        z_max       : Maximum redshift for lookback
        times       : Either equally spaced times in z or a, or a custom array of z or a
        mode    	: Defining the usage of the merger tree.
        pos_base    : Initial 3 position of base node
        velo_base   : Initial 3 velocity of base node
        scaling     : Factor of scattering of positional change over time
        verbose     : Level of output by the code
    ----------------------
    Output:
        hdf5-file if file_name is not None
    '''
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
    if random_mass=='PS':
        u_PS = np.random.rand(int(n_tree))
        ppf_PS = random_masses(w_lev[0]).random_PS(m_min,m_max)
        mp_halo = ppf_PS(u_PS)
        mp_halo = np.sort(mp_halo)[::-1]
    elif random_mass=='ST':
        u_ST = np.random.rand(int(n_tree))
        ppf_ST = random_masses(w_lev[0]).random_ST(m_min,m_max)
        mp_halo = ppf_ST(u_ST)
        mp_halo = np.sort(mp_halo)[::-1]
    else:
        mp_halo = mass            
    start_offset = 0
    nth_run = False
    if mode!='FoF':
        for i in range(n_tree):
            theta = 2*np.pi*np.random.uniform(0,1)
            u = 2*np.random.uniform(0,1)-1
            norm_vel = np.array([np.sqrt(1-u**2)*np.cos(theta),np.sqrt(1-u**2)*np.sin(theta),u])
            vel_base = np.random.lognormal(np.log(200),0.7*(mp_halo/1e4)**(-0.1))*norm_vel
            pos_base = np.random.uniform(0,BoxSize,3)
            if random_mass==None:
                count,arr_mhalo,arr_Vmax,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc,arr_nextprog,arr_pos,arr_velo,arr_spin,arr_sublen = get_tree_vals(i,i_seed_0,mp_halo,a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,scaling)
            else:
                count,arr_mhalo,arr_Vmax,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc,arr_nextprog,arr_pos,arr_velo,arr_spin,arr_sublen = get_tree_vals(i,i_seed_0,mp_halo[i],a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,scaling)
            if file_name!=None:    
                with h5py.File(file_name,'a') as f:
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
                    append_create_dataset(grp1,'SubhaloMass',data=arr_mhalo)
                    append_create_dataset(grp1,'SubhaloVmax',arr_Vmax)
                    append_create_dataset(grp1,'SubhaloPos',arr_pos)
                    append_create_dataset(grp1,'SubhaloVelo',arr_velo)
                    append_create_dataset(grp1,'SubhaloSpin',arr_spin)
                    append_create_dataset(grp1,'SubhaloLen',arr_sublen)
                    append_create_dataset(grp1,'TreeDescendant',arr_desc)
                    append_create_dataset(grp1,'FirstProgenitor',arr_1prog)
                    append_create_dataset(grp1,'NextProgenitor',arr_nextprog)
                    append_create_dataset(grp1,'TreeID',data=arr_treeid)
                    append_create_dataset(grp1,'TreeIndex',data=arr_nodid)
                    append_create_dataset(grp2,'Length',data=np.array([count]))
                    append_create_dataset(grp2,'StartOffset',data=np.array([start_offset]))
                    append_create_dataset(grp2,'TreeID',data=np.array([i]))
                    if 'Parameters' not in f:
                        f.create_group('Parameters')
                        f['Parameters'].attrs['HubbleParam'] = h_0
                        f['Parameters'].attrs['Omega0'] = omega_0
                        f['Parameters'].attrs['OmegaLambda'] = l_0
                        f['Parameters'].attrs['BoxSize'] = BoxSize
                    grp = f.create_group('Header')
                    grp.attrs['LastSnapShotNr'] = 19
                    grp.attrs['Nhalos_ThisFile'] = count
                    grp.attrs['Nhalos_Total'] = count
                    grp.attrs['Ntrees_ThisFile'] = n_tree
                    grp.attrs['Ntrees_Total'] = n_tree
                    grp.attrs['NumFiles'] = 1
                start_offset += count
    else:
        for i in range(n_tree):
            theta = 2*np.pi*np.random.uniform(0,1)
            u = 2*np.random.uniform(0,1)-1
            norm_vel = np.array([np.sqrt(1-u**2)*np.cos(theta),np.sqrt(1-u**2)*np.sin(theta),u])
            vel_base = np.random.lognormal(np.log(200),0.7*(mp_halo/1e4)**(-0.1))*norm_vel
            pos_base = np.random.uniform(0,BoxSize,3)
            if random_mass==None:
                count,arr_mhalo,arr_Vmax,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc,arr_nextprog,arr_1FoF,arr_nextFoF,arr_pos,arr_velo,arr_spin,arr_GroupMass,arr_sublen = get_tree_vals_FoF(i,i_seed_0,mp_halo,a_halo,m_min,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,scaling)
            else:
                count,arr_mhalo,arr_Vmax,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc,arr_nextprog,arr_1FoF,arr_nextFoF,arr_pos,arr_velo,arr_spin,arr_GroupMass,arr_sublen = get_tree_vals_FoF(i,i_seed_0,mp_halo[i],a_halo,m_min,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,scaling)
            if file_name!=None:
                with h5py.File(file_name,'a') as f:
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
                    append_create_dataset(grp1,'SubhaloMass',data=arr_mhalo)
                    append_create_dataset(grp1,'Group_MCrit_200',data=arr_GroupMass)
                    append_create_dataset(grp1,'SubhaloPos',arr_pos)
                    append_create_dataset(grp1,'SubhaloVelo',arr_velo)
                    append_create_dataset(grp1,'SubhaloSpin',arr_spin)
                    append_create_dataset(grp1,'SubhaloVmax',arr_Vmax)
                    append_create_dataset(grp1,'SubhaloLen',arr_sublen)
                    append_create_dataset(grp1,'TreeDescendant',arr_desc)
                    append_create_dataset(grp1,'TreeFirstProgenitor',arr_1prog)
                    append_create_dataset(grp1,'TreeNextProgenitor',arr_nextprog)
                    append_create_dataset(grp1,'TreeFirstHaloInFOFgroup',arr_1FoF)
                    append_create_dataset(grp1,'TreeNextHaloInFOFgroup',arr_nextFoF)
                    append_create_dataset(grp1,'TreeID',data=arr_treeid)
                    append_create_dataset(grp1,'TreeIndex',data=arr_nodid)
                    append_create_dataset(grp2,'Length',data=np.array([count]))
                    append_create_dataset(grp2,'StartOffset',data=np.array([start_offset]))
                    append_create_dataset(grp2,'TreeID',data=np.array([i]))
                    if 'Parameters' not in f:
                        f.create_group('Parameters')
                        f['Parameters'].attrs['HubbleParam'] = h_0 #0.6781
                        f['Parameters'].attrs['Omega0'] = omega_0 #0.30988304304812053
                        f['Parameters'].attrs['OmegaLambda'] = l_0 #0.6901169569518795
                        f['Parameters'].attrs['BoxSize'] = BoxSize
                    grp = f.create_group('Header')
                    grp.attrs['LastSnapShotNr'] = 19
                    grp.attrs['Nhalos_ThisFile'] = count
                    grp.attrs['Nhalos_Total'] = count
                    grp.attrs['Ntrees_ThisFile'] = n_tree
                    grp.attrs['Ntrees_Total'] = n_tree
                    grp.attrs['NumFiles'] = 1
                start_offset += count
    return arr_mhalo,arr_time,1/a_lev-1