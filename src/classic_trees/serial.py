from .module import functions, get_tree_vals, get_tree_vals_FoF,random_masses
import numpy as np
import h5py
import importlib.resources
import os
import sys

if float(sys.version[2:6])<9:
    filename = str(importlib.resources.path('classic_trees.Data', 'flat.txt'))
else:
    filename = str(importlib.resources.files('classic_trees.Data').joinpath('flat.txt'))
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
                 scaling = 0.3,
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
    np.random.seed(abs(i_seed_0+1))
    if type(times)==str and times=='equal_z':
        a_lev = []
        w_lev = []
        for i_lev in range(n_lev):
            a_lev.append(1/(1 + z_max*(i_lev)/(n_lev-1)))
            d_c = DELTA.delta_crit(a_lev[i_lev])
            w_lev.append(d_c)
            if verbose>0:
                print('z = ',1/a_lev[i_lev]-1,' at which delta_crit = ',d_c)
        a_lev = np.array(a_lev)
        w_lev = np.array(w_lev)
    elif type(times)==str and times=='equal_a':
        a_lev = np.linspace(1,1/(z_max+1),n_lev)
        w_lev = []
        for i_lev in range(n_lev):
            d_c = DELTA.delta_crit(a_lev[i_lev])
            w_lev.append(d_c)
            if verbose>0:
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
                if verbose>0:
                    print('z = ',1/a_temp-1,' at which delta_crit = ',d_c)
            a_lev = np.array(a_lev)
            w_lev = np.array(w_lev)
        else:
            a_lev = times
            w_lev = []
            for a in a_lev:
                d_c = DELTA.delta_crit(a)
                w_lev.append(d_c)
                if verbose>0:
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
        mp_halo = mass*np.ones(n_tree)          
    start_offset = 0
    mass_arr = []
    j_arr = []
    nth_run = False
    if mode!='FoF':
        for i in range(n_tree):
            theta = 2*np.pi*np.random.uniform(0,1)
            u = 2*np.random.uniform(0,1)-1
            norm_vel = np.array([np.sqrt(1-u**2)*np.cos(theta),np.sqrt(1-u**2)*np.sin(theta),u])
            vel_base = np.random.lognormal(np.log(200),0.7*(mp_halo[i]/1e4)**(-0.1))*norm_vel
            pos_base = np.random.uniform(0,BoxSize,3)
            if random_mass==None:
                count,arr_mhalo,arr_Vmax,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc,arr_nextprog,arr_pos,arr_velo,arr_spin,arr_sublen = get_tree_vals(i,i_seed_0,mp_halo[i],a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,scaling)
            else:
                count,arr_mhalo,arr_Vmax,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc,arr_nextprog,arr_pos,arr_velo,arr_spin,arr_sublen = get_tree_vals(i,i_seed_0,mp_halo[i],a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,scaling)
            mass_arr += np.ndarray.tolist(arr_mhalo)
            j_arr += np.ndarray.tolist(arr_time)
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
                    
                    append_create_dataset(grp1,'SnapNum',np.array(n_lev-1-arr_time,dtype=np.int32))
                    append_create_dataset(grp1,'SubhaloMass',data=np.array(arr_mhalo/1e10,dtype=np.float32))
                    append_create_dataset(grp1,'SubhaloVmax',np.array(arr_Vmax,dtype=np.float32))
                    append_create_dataset(grp1,'SubhaloPos',np.array(arr_pos,dtype=np.float32))
                    append_create_dataset(grp1,'SubhaloVelo',np.array(arr_velo,dtype=np.float32))
                    append_create_dataset(grp1,'SubhaloSpin',np.array(arr_spin,dtype=np.float32))
                    append_create_dataset(grp1,'SubhaloLen',np.array(arr_sublen,dtype=np.int32))
                    append_create_dataset(grp1,'TreeDescendant',np.array(arr_desc,dtype=np.int32))
                    append_create_dataset(grp1,'TreeFirstProgenitor',np.array(arr_1prog,dtype=np.int32))
                    append_create_dataset(grp1,'TreeNextProgenitor',np.array(arr_nextprog,dtype=np.int32))
                    append_create_dataset(grp1,'TreeID',data=arr_treeid)
                    append_create_dataset(grp1,'TreeIndex',data=np.array(arr_nodid,dtype=np.int32))
                    append_create_dataset(grp2,'Length',data=np.array([count],dtype=np.int32))
                    append_create_dataset(grp2,'StartOffset',data=np.array([start_offset]))
                    append_create_dataset(grp2,'TreeID',data=np.array([i]))
                    if 'Parameters' not in f:
                        f.create_group('Parameters')
                        f['Parameters'].attrs['HubbleParam'] = h_0
                        f['Parameters'].attrs['Omega0'] = omega_0
                        f['Parameters'].attrs['OmegaLambda'] = l_0
                        f['Parameters'].attrs['BoxSize'] = BoxSize
                start_offset += count
    else:
        for i in range(n_tree):
            theta = 2*np.pi*np.random.uniform(0,1)
            u = 2*np.random.uniform(0,1)-1
            norm_vel = np.array([np.sqrt(1-u**2)*np.cos(theta),np.sqrt(1-u**2)*np.sin(theta),u])
            vel_base = np.random.lognormal(np.log(200),0.7*(mp_halo[i]/1e4)**(-0.1))*norm_vel
            pos_base = np.random.uniform(0,BoxSize,3)
            if random_mass==None:
                count,arr_mhalo,arr_Vmax,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc,arr_nextprog,arr_1FoF,arr_nextFoF,arr_pos,arr_velo,arr_spin,arr_GroupMass,arr_sublen = get_tree_vals_FoF(i,i_seed_0,mp_halo[i],a_halo,m_min,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,scaling,'ST')
            else:
                count,arr_mhalo,arr_Vmax,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc,arr_nextprog,arr_1FoF,arr_nextFoF,arr_pos,arr_velo,arr_spin,arr_GroupMass,arr_sublen = get_tree_vals_FoF(i,i_seed_0,mp_halo[i],a_halo,m_min,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,scaling,random_mass)
            mass_arr += np.ndarray.tolist(arr_mhalo)
            j_arr += np.ndarray.tolist(arr_time)
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
                        d_red = grp3.create_dataset('Redshift',data=1/a_lev[::-1]-1)
                        d_time= grp3.create_dataset('Time',data=a_lev[::-1])
                    
                    append_create_dataset(grp1,'SnapNum',np.array(n_lev-1-arr_time,dtype=np.int32))
                    append_create_dataset(grp1,'SubhaloMass',data=np.array(arr_mhalo/1e10,dtype=np.float32))
                    append_create_dataset(grp1,'Group_M_Crit200',data=np.array(arr_GroupMass/1e10,dtype=np.float32))
                    append_create_dataset(grp1,'SubhaloPos',np.array(arr_pos,dtype=np.float32))
                    append_create_dataset(grp1,'SubhaloVelo',np.array(arr_velo,dtype=np.float32))
                    append_create_dataset(grp1,'SubhaloSpin',np.array(arr_spin,dtype=np.float32))
                    append_create_dataset(grp1,'SubhaloVmax',np.array(arr_Vmax,dtype=np.float32))
                    append_create_dataset(grp1,'SubhaloLen',np.array(arr_sublen,dtype=np.int32))
                    append_create_dataset(grp1,'TreeDescendant',np.array(arr_desc,dtype=np.int32))
                    append_create_dataset(grp1,'TreeFirstProgenitor',np.array(arr_1prog,dtype=np.int32))
                    append_create_dataset(grp1,'TreeNextProgenitor',np.array(arr_nextprog,dtype=np.int32))
                    append_create_dataset(grp1,'TreeFirstHaloInFOFgroup',np.array(arr_1FoF,dtype=np.int32))
                    append_create_dataset(grp1,'TreeNextHaloInFOFgroup',np.array(arr_nextFoF,dtype=np.int32))
                    append_create_dataset(grp1,'SubhaloVelDisp',np.zeros(np.sum(count),dtype=np.float32))
                    append_create_dataset(grp1,'TreeID',data=arr_treeid)
                    append_create_dataset(grp1,'TreeIndex',data=np.array(arr_nodid,dtype=np.int32))
                    append_create_dataset(grp2,'Length',data=np.array([int(np.sum(count))],dtype=np.int32))
                    append_create_dataset(grp2,'StartOffset',data=np.array([start_offset]))
                    append_create_dataset(grp2,'TreeID',data=np.array([i]))
                    if 'Parameters' not in f:
                        f.create_group('Parameters')
                        f['Parameters'].attrs['HubbleParam'] = h_0
                        f['Parameters'].attrs['Omega0'] = omega_0
                        f['Parameters'].attrs['OmegaLambda'] = l_0
                        f['Parameters'].attrs['BoxSize'] = BoxSize
                start_offset += sum(count)
    if file_name!=None:
        with h5py.File(file_name,'a') as f:
            grp = f.create_group('Header')
            grp.attrs['LastSnapShotNr'] = np.int32(n_lev-1)
            grp.attrs['Nhalos_ThisFile'] = start_offset
            grp.attrs['Nhalos_Total'] = start_offset
            grp.attrs['Ntrees_ThisFile'] = n_tree
            grp.attrs['Ntrees_Total'] = n_tree
            grp.attrs['NumFiles'] = np.int32(1)
            if mode=='FoF':
                grp1 = f['TreeHalos']
                arr_SubhaloNr = np.zeros(start_offset,dtype='int_')
                arr_MostBoundID = np.array([i for i in range(start_offset)],dtype=np.uint32)
                SnapNum = f['TreeHalos/SnapNum'][:]
                masses = f['TreeHalos/Group_M_Crit200'][:]
                for i in range(n_lev-1,0,-1):
                    indx_lev = np.where(SnapNum==i)[0]
                    if len(indx_lev)!=0:
                        mass_lev = masses[indx_lev]
                        temp_arr = []
                        m_temp = 0
                        for indx,j in enumerate(indx_lev):
                            if mass_lev[indx]>0:
                                m_temp = mass_lev[indx]
                                temp_arr.append([mass_lev[indx],j])
                            else:
                                temp_arr.append([m_temp,j])
                        sorted(temp_arr,key = lambda x:x[0],reverse=True)
                        c = 0
                        for j in range(len(temp_arr)):
                            arr_SubhaloNr[temp_arr[j][1]] = c
                            c += 1
                append_create_dataset(grp1,'SubhaloNr',arr_SubhaloNr)
                append_create_dataset(grp1,'SubhaloIDMostbound',arr_MostBoundID)
    return np.array(mass_arr),np.array(j_arr),1/a_lev-1