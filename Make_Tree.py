from classic_trees import Tree_Node
from classic_trees import Make_Siblings
from classic_trees import functions
# from classic_trees import split
import numpy as np
# from Tree_Node_and_Memory import *
# from make_tree_arrays_and_modules import *
from split_function import *
# from Delta_crit import *
import bisect

filename = './CLASSIC-trees/Data/flat.txt'
DELTA = functions(filename)

def make_tree(m_0,a_0,m_min,a_lev,n_lev,n_frag_max,n_frag_tot=0):
    # print('in make_tree function')
    merger_tree = []
    n_v = 20000
    # merger_tree.append([])
    # merger_tree.append([])
    # merger_tree.append([])
    # #print(node)
    i_err = 0
    child_ref = np.zeros(n_frag_max) #[0]*n_frag_max
    j_index = np.zeros(n_frag_max,dtype='int_') #[0]*n_frag_max
    m_tr = np.zeros(n_frag_max) #[0]*n_frag_max
    i_par= np.zeros(n_frag_max,dtype='int_') #[0]*n_frag_max
    i_sib = np.zeros(n_frag_max,dtype='int_') #[0]*n_frag_max
    i_child = np.zeros(n_frag_max,dtype='int_') #[0]*n_frag_max
    w_lev = [0]*n_lev
    n_frag_lev = [-1]*n_lev
    jp_frag = [-1]*n_lev
    m_left = np.zeros(n_v) #[0]*n_v
    m_right = np.zeros(n_v) #[0]*n_v
    w_node = np.zeros(n_v) #[0]*n_v
    l_node = np.zeros(n_v,dtype='bool') #[False]*n_v

    for i_frag in range(int(n_frag_max)):
        node = Tree_Node()
        node.mhalo = 0.0
        node.jlevel = 0
        node.nchild = 0
        node.parent = None
        node.child  = None
        node.sibling= None
        merger_tree.append(node)
        
    a_lev[0] = a_0
    #print('a_lev = ',a_lev)
    for i_lev in range(n_lev):
        w_lev[i_lev] = DELTA.delta_crit(a_lev[i_lev])
    #print(w_lev)
    #w_lev = np.loadtxt('./Code_own/delta_crit_values.txt')
    i_node = 0
    w_fin = w_lev[n_lev-1]
    
    i_frag_lev = [-1]*n_lev
    #print('m_0 = ',m_0)
    m_left[0] = m_0
    m_right[0] = -1
    m = m_0
    w_node[0] = w_lev[0]
    w = w_lev[0]
    i_lev = 1
    i_frag_lev[0] = 0

    i_frag = 0
    m_tr[0] = m_0
    m_minlast= 0
    i_par[0] = -1
    i_sib[0] = -1
    i_child[0] = -1
    #print('m = ',m)
    count = 0
    while m_left[i_node] > 0.0:
        count += 1
        # print('count: ',count)
        dw_max = w_lev[i_lev] - w
        #print('i_lev = ',i_lev)
        #print('dw_max = ',dw_max)
        dw,n_prog,m_prog,m_minlast = split(m, w, m_min, dw_max, 0.1, 0.1,m_minlast)
        #print('dw = ',dw)
        w += dw
        #print('m_prog = ',m_prog)
        if n_prog==2:
            i_node += 1
            w_node[i_node] = w
            l_node[i_node] = False
            m_left[i_node] = m_prog[0]
            m_right[i_node] = m_prog[1]
            m = m_left[i_node]
        elif n_prog==1:
            m = m_prog[0]
        elif n_prog==0:
            m = 0.0
            m_left[i_node] = -1
        if w == w_lev[i_lev] and  n_prog > 0.0:
            #print('In here?---------------------?')
            if i_frag+2 > n_frag_max:
                i_err = 1
                return

            i_frag += 1
            #print('i_lev = ',i_lev)
            m_tr[i_frag] = m_prog[0]
            i_par[i_frag] = i_frag_lev[i_lev-1]
            i_sib[i_frag] = -1
            i_child[i_frag] = -1

            i_frag_prev = i_frag_lev[i_lev]
            if i_frag_prev > 0:
                i_par_prev = i_par[i_frag_prev]
                if i_par_prev == i_par[i_frag]:
                    i_sib[i_frag_prev] = i_frag
                else:
                    i_child[i_par[i_frag]] = i_frag
            else:
                i_child[i_par[i_frag]] = i_frag
            
            i_frag_lev[i_lev] = i_frag

            if n_prog == 2:
                i_frag += 1
                m_tr[i_frag] = m_prog[1]
                i_par[i_frag] = i_frag_lev[i_lev-1]
                i_sib[i_frag] = -1
                i_child[i_frag] = -1
                i_sib[i_frag-1] = i_frag
                l_node[i_node] = True
                if i_lev == n_lev-1:
                    i_frag_lev[i_lev] = i_frag
            if i_lev < n_lev-1:
                i_lev +=1
            
        if w >= w_fin:
            if n_prog == 2:
                m_left[i_node] = -1
                m_right[i_node] = -1
            else:
                m_left[i_node] = -1
            
        while m_left[i_node] < 0.0 and i_node > 0:
            if m_right[i_node] > 0.0:
                m_left[i_node] = m_right[i_node]
                m_right[i_node] = -1
                w = w_node[i_node]
                m = m_left[i_node]
                # print('w = ',w)
                # print('w[i_lev-1] = ',w_lev[i_lev-1])
                # print('w[i_lev] = ',w_lev[i_lev])
                if w < w_lev[i_lev-1] or w >= w_lev[i_lev]:
                    iw = bisect.bisect_left(w_lev,w) #locate(w_lev, n_lev, w)
                    i_lev = iw
                    #print('i_lev = ',i_lev)
                    #print(w_lev[i_lev],' = ',w)
                    if w_lev[i_lev] == w:
                        #print('briefly here --------------!')
                        i_lev += 1
                #print('l_node[i_node] = ',l_node[i_node])
                if l_node[i_node]:
                    i_frag_lev[i_lev-1] += 1
            else:
                i_node -= 1
                m_left[i_node] = -1

    n_frag_tot = i_frag
    # c_sib = 0
    # for sib in i_par:
    #     if sib != 0:
    #         c_sib += 1
    #         print(sib)
    # print('number of non-zero elements in i_par: ',c_sib)
    j_frag = -1
    for i_lev_wk in range(n_lev):
        #print('Or here?---------------------?')
        i_lev = 0
        i_frag  = 0
        while i_frag >= 0:
            while i_lev < i_lev_wk and i_child[i_frag] > 0:
                i_frag = i_child[i_frag]
                i_lev += 1
            if i_lev == i_lev_wk:
                if jp_frag[i_lev_wk]<0:
                    jp_frag[i_lev_wk] = j_frag +1
                while i_frag >= 0:
                    #print('Or here?---------------------?')
                    j_frag += 1
                    # print('j_frag = ',j_frag)
                    j_index[i_frag] = j_frag
                    n_frag_lev[i_lev_wk] += 1
                    merger_tree[j_frag].mhalo = m_tr[i_frag]
                    merger_tree[j_frag].jlevel = i_lev_wk

                    if i_lev > 0:
                        merger_tree[j_frag].parent = merger_tree[j_index[i_par[i_frag]]]
                    else:
                        merger_tree[j_frag].parent = None
                    i_frag_prev = i_frag
                    i_frag = i_sib[i_frag]
                i_frag = i_frag_prev
                if i_lev > 0:
                    i_lev -=1
                    i_frag = i_par[i_frag]
        
            while i_sib[i_frag] < 0 and i_lev > 0:
                i_lev -= 1
                i_frag = i_par[i_frag]
            if i_lev > 0:
                i_frag = i_sib[i_frag]
            else:
                i_frag = -1
    for i_frag in range(n_frag_tot):
        j_frag = j_index[i_frag]
        i_frag_c = i_child[i_frag]
        if i_frag_c >= 0:
            child_ref[j_frag] = j_index[i_frag_c]

            merger_tree[j_frag].child = merger_tree[j_index[i_frag_c]]
            n_ch = 0
            while i_frag_c >= 0:
                n_ch += 1
                i_frag_c = i_sib[i_frag_c]
            # print('number of children here: ',n_ch)
            merger_tree[j_frag].nchild = n_ch

    merger_tree = Make_Siblings().build_sibling(merger_tree,n_frag_tot)
    
    i_err = 0
    return i_err, n_frag_lev, jp_frag, m_tr, merger_tree, node