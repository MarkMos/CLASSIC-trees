# from Make_Tree import *
# from Tree_Node_and_Memory import *
# from Moving_in_Tree import *
# from walk_the_tree import walk_tree
import random
# from Delta_crit import *
from classic_trees import functions, get_tree_vals#node_vals_and_counter, make_tree
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

# merger_tree = tree_memory_arrays_passable().merger_tree
# merger_tree_aux = tree_memory_arrays().merger_tree_aux

n_lev = 10
n_halo_lev = 10

mp_halo = 1e10
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
    # iter = 1
    # start = time.time()
    # while i_err != 0 or iter == 1:
    #     if iter == 1:
    #         i_seed_0 -= 19
    #     i_seed = i_seed_0
    #     random.seed(i_seed)
    #     '''
    #     for j in range(n_halo_max):
    #         merger_tree_aux[j].index = j
    #     merger_tree = merger_tree_aux'
    #     '''
    #     print('Making a tree...',i)
    #     my_tree = make_tree(mp_halo,a_halo,m_res,a_lev,n_lev,n_halo_max,n_halo)
    #     # print(my_tree)
    #     i_err = my_tree[0]
    #     merger_tree = my_tree[1]
    #     iter +=1
    #     #print(merger_tree)
    # print('Made a tree',i)
    '''
    for i in range(len(merger_tree)):
        print('We have the following values for mhalo in node ',i,' as:')
        print(merger_tree[i].mhalo)
    '''
    # end = time.time()
    # print(f"Elapsed time in make_tree: {end - start} seconds")
    # this_node = merger_tree[0]
    # print('m_halo = ',this_node.mhalo)
    # arr_mhalo = np.zeros(n_halo_max)-1
    # arr_nodid = np.zeros(n_halo_max,dtype='int_')-1
    # arr_treeid= np.zeros(n_halo_max,dtype='int_')-1
    # arr_time  = np.zeros(n_halo_max,dtype='int_')-1
    # arr_1prog = np.zeros(n_halo_max,dtype='int_')-1
    # arr_desc  = np.zeros(n_halo_max,dtype='int_')-1
    # while this_node is not None:
    #     node_ID = count
    #     arr_nodid[node_ID] = node_ID
    #     arr_mhalo[node_ID] = this_node.mhalo
    #     arr_treeid[node_ID]= i
    #     arr_time[node_ID]  = this_node.jlevel
    #     if this_node.child is not None:
    #         arr_1prog[node_ID] = merger_tree.index(this_node.child)
    #     else:
    #         arr_1prog[node_ID] = -1
    #     if this_node.parent is not None:
    #         arr_desc[node_ID] = merger_tree.index(this_node.parent)
    #     else:
    #         arr_desc[node_ID] = -1
    #     count +=1
    #     this_node = walk_tree(this_node)
    #     # print(this_node)
    #     # print('index = ',this_node.mhalo)
    #     # print('m_halo = ',this_node.mhalo)
    #     # print('count = ',count)
    #     # break

    # arr_mhalo = arr_mhalo[0:count]
    # arr_nodid = arr_nodid[0:count]
    # arr_treeid= arr_treeid[0:count]
    # arr_time  = arr_time[0:count]
    # arr_1prog = arr_1prog[0:count]
    # arr_desc  = arr_desc[0:count]
    start = time.time()
    count,arr_mhalo,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc = get_tree_vals(i,i_seed_0,mp_halo,a_halo,m_res,a_lev,n_lev,n_halo_max,n_halo)

    # print('Number of nodes in tree',i+1,'is',count)
    
    # print('Example information from tree:')
    # this_node = merger_tree[0]
    # print('Base node: \n  mass =',this_node.mhalo,' z= ',1/a_lev[this_node.jlevel]-1,' number of progenitors ',this_node.nchild)
    # this_node = this_node.child
    # print('First progenitor: \n  mass =',this_node.mhalo,' z= ',1/a_lev[this_node.jlevel]-1)
    # this_node = this_node.sibling
    # print('  mass =',this_node.mhalo,' z= ',1/a_lev[this_node.jlevel]-1)
    # print(arr_mhalo)
    with h5py.File('./Code_own/Trees/tree_selftest1e10.hdf5','a') as f:
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
        append_create_dataset(grp1,'TreeDescendant',arr_desc)
        append_create_dataset(grp1,'TreeFirstProgenitor',arr_1prog)
        append_create_dataset(grp1,'TreeID',data=arr_treeid)
        append_create_dataset(grp1,'TreeIndex',data=arr_nodid)
        append_create_dataset(grp2,'Length',data=np.array([count]))
        append_create_dataset(grp2,'StartOffset',data=np.array([start_offset]))
        append_create_dataset(grp2,'TreeID',data=np.array([i]))

    start_offset += count


    end = time.time()
    print(f"Elapsed time: {end - start} seconds")