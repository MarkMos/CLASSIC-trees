from Tree_Node_and_Memory import *
from make_tree_arrays_and_modules import *
from split_function import *
from index_functions import *
from locate_function import *
from Delta_crit import *
import bisect

#n_prog = 0
#m_prog = [0.0, 0.0]

#merger_tree = tree_memory_arrays_passable().merger_tree

#tree_index = tree_memory_arrays_passable().tree_index

def make_tree(m_0,a_0,m_min,a_lev,n_lev,n_frag_max,n_frag_tot=0):
    # print('in make_tree function')
    merger_tree = [] # tree_memory_arrays_passable().merger_tree
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

    for i_frag in range(int(n_frag_max/5)):
        node = Tree_Node()
        node.mhalo = 0.0
        node.jlevel = 0
        node.nchild = 0
        node.parent = None
        node.child  = None
        node.sibling= None
        merger_tree.append(node)
        node = Tree_Node()
        node.mhalo = 0.0
        node.jlevel = 0
        node.nchild = 0
        node.parent = None
        node.child  = None
        node.sibling= None
        merger_tree.append(node)
        node = Tree_Node()
        node.mhalo = 0.0
        node.jlevel = 0
        node.nchild = 0
        node.parent = None
        node.child  = None
        node.sibling= None
        merger_tree.append(node)
        node = Tree_Node()
        node.mhalo = 0.0
        node.jlevel = 0
        node.nchild = 0
        node.parent = None
        node.child  = None
        node.sibling= None
        merger_tree.append(node)
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
        w_lev[i_lev] = delta_crit(a_lev[i_lev])
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
        #print('count: ',count)
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
            
    # k_frag = 0
    # j_index[0] = 0
    # i_sib = [0]*n_frag_max
    # i_child = [0]*n_frag_max
    # m_tr = [0]*n_frag_max
    #print('now here: ',m_tr)
    # merger_tree[0].mhalo = m_tr[0]
    # print('I am here with ',m_tr[0])
    '''
    for j in range(n_frag_tot):
        merger_tree[j].mhalo = m_tr[j]
    
    for i_lev in range(n_lev-1):
        # print('start point range: ',jp_frag[i_lev])
        # print('end point range: ',jp_frag[i_lev]+n_frag_lev[i_lev])
        for k_frag_p in range(jp_frag[i_lev],jp_frag[i_lev]+n_frag_lev[i_lev]):
            #print('k_frag_p = ',k_frag_p)
            j_frag_p = j_index[k_frag_p]
            n_child1 = merger_tree[j_frag_p].nchild
            i_sib[k_frag_p] = n_child1
            j_child1 = child_ref[j_frag_p]
            if n_child1==1:
                #print('indx_ch = ',indx_ch)
                indx_ch[0] = 0
            elif n_child1 >= 2:
                if n_child1 > n_ch_max:
                    print('make_tree: n_child1 > n_ch_max increase n_ch_max \n n_child1 = ',n_child1)
                #length = len(merger_tree[j_child1:j_child1+n_child1])
                indx_ch = indexxx(n_child1,[merger_tree[i+j_child1].mhalo for i in range(n_child1)],indx_ch)
            if n_child1 >= 1:
                i_child[k_frag_p] = k_frag + 1
            else:
                i_child[k_frag_p] = -1
            for k_child in range(n_child1-1,-1,-1):
                j_frag_c = indx_ch[k_child] + j_child1 - 1  # Adjust for Python's 0-based indexing
                k_frag += 1
                j_index[k_frag] = j_frag_c  # Adjust for Python's 0-based indexing
                #merger_tree[k_frag].parent = merger_tree[k_frag_p]  # Assign parent as a reference
                #m_tr[k_frag] = merger_tree[j_frag_c].mhalo  # Store mhalo value in mtr array
    '''
    # for k_frag in range(jp_frag[n_lev-1],jp_frag[n_lev-1]+n_frag_lev[n_lev-1]+1):
    #     i_sib[k_frag] = 0
    #     i_child[k_frag] = -1
    #m_tr.sort(reverse=True)
    # for j in range(n_frag_tot):
    #     merger_tree[j].mhalo = m_tr[j]
    # child_ref[0:n_frag_tot] = i_child[0:n_frag_tot]
    # for i_frag in range(n_frag_tot):
    #     if i_child[i_frag] > 0:
    #         merger_tree[i_frag].child = merger_tree[i_child[i_frag]]
    #     else:
    #         merger_tree[i_frag].child = None
    
    # for k in range(n_frag_tot):
    #    merger_tree[k].nchild = i_sib[k]

    merger_tree = build_sibling(merger_tree,n_frag_tot)
    
    i_err = 0
    # c_m = 0
    # for m in merger_tree:
    #     if m.mhalo != 0:
    #         c_m += m.mhalo
            # print(m.mhalo)
    # print('sum of masses: ',c_m)
    # print('len(merger_tree) = ',len(merger_tree))
    # print('n_frag_tot = ',n_frag_tot)
    # print(i_child)
    #print(node.mhalo,'here')
    #print('i_sib = ',i_sib)
    return i_err, n_frag_lev, jp_frag, m_tr, merger_tree, node