from classic_trees import functions,speed_test,set_trees
import numpy as np
from multiprocessing import Pool, Lock

lock = Lock()

# import CLASSIC_trees as ct

# tree = ct.trees()
# tree.set(pk_method='default') #,add_cosmo_params={'N_ncdm':1})#,cosmo_params={'h':0.8,'Omega_m':0.15,'Omega_Lambda':0.85})
# set_trees(tree)

filename = './CLASSIC-trees/Data/flat.txt'
DELTA = functions(filename)

def tree_speed(j,n_tree,i_seed_0,mp_halo,a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,scaling):
    speed_test(j,n_tree,i_seed_0,mp_halo,a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,scaling)

def parallel_speed(n_part,n_tree,i_seed_0,mp_halo,a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,scaling):
    arg_list = [(j,n_tree,i_seed_0,mp_halo,a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,scaling) for j in range(n_part)]
    with Pool() as pool:
        pool.starmap(tree_speed, arg_list)


def compute_speed():

    z_max = 4
    n_lev = 10

    m_res = 1e8
    mp_halo = 1e12

    i_seed_0 = -8635
    a_halo = 1
    n_halo = 1
    n_halo_max = 5000

    pos_base = np.array([0,0,0],dtype=np.float64)
    vel_base = np.array([100,100,100],dtype=np.float64)


    a_lev = []
    w_lev = []

    for i_lev in range(n_lev):
        a_lev.append(1/(1+z_max*(i_lev)/(n_lev-1)))
        d_c = DELTA.delta_crit(a_lev[i_lev])
        w_lev.append(d_c)
        print('z = ',1/a_lev[i_lev]-1,' at which delta_crit = ',d_c)
    a_lev = np.array(a_lev)
    w_lev = np.array(w_lev)

    n_part = 75
    n_tree = 100

    # parallel_speed(n_tree,i_seed_0,mp_halo,a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,0.3)

    parallel_speed(n_part,n_tree,i_seed_0,mp_halo,a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,0.3)
        # speed_test(i,n_tree,i_seed_0,mp_halo,a_halo,m_res,w_lev,a_lev,n_lev,n_halo_max,n_halo,pos_base,vel_base,0.3)