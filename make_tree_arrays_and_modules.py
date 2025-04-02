from Tree_Node_and_Memory import *

#merger_tree = tree_memory_arrays_passable().merger_tree

class make_tree_arrays:
    def __init__(self):
        self.m_left = []
        self.m_right = []
        self.w_node = []
        self.l_node = []

def next_sibling(child_node, parent_node):
    '''
    Get the next sibling of the child node.
    '''
    if child_node == parent_node:
        if parent_node.child is not None:
            Sibling = parent_node.child
        else:
            raise ValueError('next_sibling(): Parent has no children.')
    else:
        if child_node.sibling is not None:
            Sibling = child_node.sibling
        else:
            raise ValueError('next_sibling(): Child has no siblings.')
    sibs_left = Sibling.sibling
    return Sibling, sibs_left

def associated_siblings(this_node,merger_tree):
    if this_node.nchild > 1:
        child_index = tree_index(this_node.child,merger_tree)
        #print('from ',child_index,' to ',child_index+this_node.nchild-1)
        sib_nodes = []
        merger_temp = []
        # indices = []
        for i_frag in range(child_index, child_index + this_node.nchild -1):
            sib_nodes.append([merger_tree[i_frag+1],i_frag])
            # indices.append(i_frag)
            # merger_tree[i_frag].sibling = merger_tree[i_frag + 1]
            # print('halo mass = ',merger_tree[i_frag+1].mhalo)
        sib_sorted = sorted(sib_nodes,key=lambda x:x[0].mhalo,reverse=True)
        # print(sib_nodes)
        # print(sib_sorted)
        n = len(sib_sorted)
        for i in range(n):
            merger_temp.append(merger_tree[sib_sorted[i][1]+1])
        for k in range(n):
            merger_tree[sib_nodes[k][1]+1] = merger_temp[k]
        for j in range(n):
            merger_tree[sib_nodes[j][1]].sibling = merger_tree[sib_nodes[j][1]+1]
            #print('Tree node: ',sib_sorted[i][0])
            # print('halo mass = ',sib_sorted[i][0].mhalo)
        # print('i = ',i)
    return merger_tree

def build_sibling(merger_tree,n_frag_tot):
    for i in range(n_frag_tot):
        merger_tree = associated_siblings(merger_tree[i],merger_tree)
    return merger_tree