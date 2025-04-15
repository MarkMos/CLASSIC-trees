# from Make_Tree import *
# from Tree_Node_and_Memory import *
# from Moving_in_Tree import *
# from Delta_crit import *
# from sigma_cdm_func import *
from classic_trees import get_tree_vals, functions
import numpy as np
import h5py
import time
from multiprocessing import Pool, Lock

start = time.time()

lock = Lock()

filename = './CLASSIC-trees/Data/flat.txt'
DELTA = functions(filename)

def tree_process(i,i_seed_0,mp_halo,a_halo,m_res,a_lev,n_lev,n_halo_max,n_halo):
    # iter = 1
    # i_err= 1 #Initialising error

    # while i_err != 0 or iter == 1:
    #     if iter == 1:
    #         i_seed_0 -= 19*(1+i)
    #     i_seed = i_seed_0
    #     random.seed(i_seed)
    #     print('Making a tree...',i)

    #     # Make a merger-tree using the make_tree function
    #     my_tree = make_tree(mp_halo,a_halo,m_res,a_lev,n_lev,n_halo_max,n_halo)
    #     i_err = my_tree[0]
    #     merger_tree = my_tree[-2]
    #     iter +=1
    # print('Made a tree',i)
    count,arr_mhalo,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc = get_tree_vals(i,i_seed_0,mp_halo,a_halo,m_res,a_lev,n_lev,n_halo_max,n_halo)
    # Initialising arrays to save important quantaties of the merger-tree and
    # walking through tree to count the number of nodes using walk_tree function
    # this_node = merger_tree[0]
    # count = 0
    # arr_mhalo = np.zeros(n_halo_max)-1              # array for halo masses
    # arr_nodid = np.zeros(n_halo_max,dtype='int_')-1 # array for the node id's
    # arr_treeid= np.zeros(n_halo_max,dtype='int_')-1 # array for the tree id's
    # arr_time  = np.zeros(n_halo_max,dtype='int_')-1 # array for the scale factor of the nodes
    # arr_1prog = np.zeros(n_halo_max,dtype='int_')-1 # array for first progenitors of nodes
    # arr_desc  = np.zeros(n_halo_max,dtype='int_')-1 # array for the descandant of nodes
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

    # # Trim arrays to actual size
    # arr_mhalo = arr_mhalo[0:count]
    # arr_nodid = arr_nodid[0:count]
    # arr_treeid= arr_treeid[0:count]
    # arr_time  = arr_time[0:count]
    # arr_1prog = arr_1prog[0:count]
    # arr_desc  = arr_desc[0:count]
    count = np.array([count],dtype='int_')
    i = np.array([i],dtype='int_')

    # print('Number of nodes in tree',i+1,'is',count)
    
    # print('Example information from tree:')
    # this_node = merger_tree[0]
    # print('Base node: \n  mass =',this_node.mhalo,' z= ',1/a_lev[this_node.jlevel]-1,' number of progenitors ',this_node.nchild)
    # this_node = this_node.child
    # print('First progenitor: \n  mass =',this_node.mhalo,' z= ',1/a_lev[this_node.jlevel]-1)
    # this_node = this_node.sibling
    # print('  mass =',this_node.mhalo,' z= ',1/a_lev[this_node.jlevel]-1)
    return {
        "arr_mhalo": arr_mhalo,
        "arr_nodid": arr_nodid,
        "arr_treeid": arr_treeid,
        "arr_time": arr_time,
        "arr_1prog": arr_1prog,
        "arr_desc": arr_desc,
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
def parallel_exe(j,n_tree,i_seed_0,mp_halo,a_halo,m_res,a_lev,n_lev,n_halo_max,n_halo,nth_run,start_offset):
    args_list = [(i,i_seed_0,mp_halo,a_halo,m_res,a_lev,n_lev,n_halo_max,n_halo)
                  for i in range(j*n_tree,n_tree+j*n_tree)]
    with Pool() as pool:
        results = pool.starmap(tree_process, args_list)
    print('Here')
    with h5py.File('./Code_own/Trees/tree_selftestfast_r45_1e14.hdf5','a',libver='latest') as f:
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
        for result in results:
            append_create_dataset(grp1,'SnapNum',result['arr_time'])
            append_create_dataset(grp1,'SubhaloMass',data=result['arr_mhalo'])
            append_create_dataset(grp1,'TreeDescendant',result['arr_desc'])
            append_create_dataset(grp1,'TreeFirstProgenitor',result['arr_1prog'])
            append_create_dataset(grp1,'TreeID',data=result['arr_treeid'])
            append_create_dataset(grp1,'TreeIndex',data=result['arr_nodid'])
            append_create_dataset(grp2,'Length',data=result['count'])
            append_create_dataset(grp2,'StartOffset',data=np.array([start_offset]))
            append_create_dataset(grp2,'TreeID',data=result['tree_index'])
            start_offset += result['count']
        return start_offset

if __name__ == '__main__':
    n_tree = 15
    i_seed_0 = -8635
    mp_halo = 1e14
    a_halo = 1
    m_res = 1e8
    z_max  = 4
    n_lev = 10
    n_halo_max = 1000000
    n_halo = 1

    n_part = 3

    a_lev = []
    for i_lev in range(1,n_lev+1):
        a_lev.append(1/(1 + z_max*(i_lev-1)/(n_lev-1)))
        #print(a_lev)
        d_c = DELTA.delta_crit(a_lev[i_lev-1])
        print('z = ',1/a_lev[i_lev-1]-1,' at which delta_crit = ',d_c)
    a_lev = np.array(a_lev)
    nth_run = False
    start_offset = 0
    for j in range(n_part):
        start_offset = parallel_exe(j,n_tree,i_seed_0,mp_halo,a_halo,m_res,a_lev,n_lev,n_halo_max,n_halo,nth_run,start_offset)


end = time.time()
print(f"Elapsed time: {end - start} seconds")