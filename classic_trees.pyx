import cython
import numpy as np
cimport numpy as np
import bisect
#from CLASSIC_trees import trees
#from values import omega_0, l_0, h_0, gamma_1, gamma_2, G_0, eps_1,eps_2, file_name_pk
from scipy.integrate import simpson
from astropy.constants import G, M_sun, pc
from libc.math cimport sqrt, log, exp, pi, cos, sin, fabs, acosh, sinh, cosh
from scipy.interpolate import UnivariateSpline, CubicSpline
from libc.stdlib cimport malloc, free, rand, srand

ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

trees = None
def set_trees(obj):
    global trees
    trees = obj

cdef double G_0=0.57
cdef double gamma_1=0.38
cdef double gamma_2=-0.01
cdef double eps_1=0.1
cdef double eps_2=0.1


cdef double* c_logspace(double start, double stop,int n):
    '''
    Function to make an logspaced array in C
    ----------------------
    Input:
        start: Start of the array (here: e.g. -8 -> 10^-8)
        stop : End of the array (here: e.g. 3 -> 10^3)
        n    : Number of steps between start and stop
    ----------------------
    Output:
        c_arr: C-array in logspace
    '''
    cdef double* c_arr = <double*>malloc(n*sizeof(double))
    if not c_arr:
        raise MemoryError()
    cdef double step = (stop-start)/(n-1)
    for i in range(n):
        c_arr[i] = 10**(start + i*step)
    return c_arr

cdef struct cspline:
    int n
    double * x
    double *y
    double *b
    double *c
    double *d

cdef cspline* cspline_alloc(int n, double * x, double * y) nogil:
    cdef cspline* s = <cspline*>malloc(sizeof(cspline))
    cdef int i

    # allocate already n, x and y
    s.n = n
    s.x = x
    s.y = y
    # and prepare the allocation of b, c and d
    s.b = <double*>malloc(n*sizeof(double))
    s.c = <double*>malloc((n-1)*sizeof(double))
    s.d = <double*>malloc((n-1)*sizeof(double))

    # derivative calculation and allocation
    cdef double* dx = <double*>malloc((n-1)*sizeof(double))
    cdef double* dydx = <double*>malloc((n-1)*sizeof(double))
    for i in range(n-1):
        dx[i] = x[i+1] - x[i]
        dydx[i] = (y[i+1] - y[i])/dx[i]
    
    cdef double* D = <double*>malloc(n*sizeof(double))
    cdef double* Q = <double*>malloc((n-1)*sizeof(double))
    cdef double* B = <double*>malloc(n*sizeof(double))
    D[0] = 2
    for i in range(n-2):
        D[i+1] = 2*dx[i]/dx[i+1]+2
    D[n-1] = 2

    Q[0] = 1
    for i in range(n-2):
        Q[i+1] = dx[i]/dx[i+1]
    for i in range(n-2):
        B[i+1] = 3*(dydx[i]+dydx[i+1]*dx[i]/dx[i+1])
    B[0] = 3*dydx[0]
    B[n-1] = 3*dydx[n-2]

    for i in range(1,n):
        D[i] -= Q[i-1]/D[i-1]
        B[i] -= B[i-1]/D[i-1]
    
    s.b[n-1] = B[n-1]/D[n-1]

    for i in range(n-2,0,-1):
        s.b[i] = (B[i] - Q[i]*s.b[i+1])/D[i]
    for i in range(n-1):
        s.c[i] = (-2*s.b[i] - s.b[i+1] + 3*dydx[i])/dx[i]
        s.d[i] = (s.b[i] + s.b[i+1] - 2*dydx[i])/dx[i]/dx[i]
    return s

cdef double cspline_eval(cspline * s, double z):
    # function to evaluate a spline
    cdef int n = s.n
    if not (n>1 and z>=s.x[0] and z<=s.x[n-1]):
        raise ValueError('z does not lay in range of x')
    cdef int i = 0
    cdef int j = n-1
    cdef int m
    while j-i>1:
        m = (i+j)//2
        if z > s.x[m]:
            i = m
        else:
            j = m
    return s.y[i] + s.b[i]*(z-s.x[i]) + s.c[i]*(z-s.x[i])*(z-s.x[i]) + s.d[i]*(z-s.x[i])*(z-s.x[i])*(z-s.x[i])

cdef double cspline_deriv(cspline * s, double z):
    # function to calculate the derivative of a spline
    cdef int n = s.n
    if not (n>1 and z>=s.x[0] and z<=s.x[n-1]):
        raise ValueError('z does not lay in range of x')
    cdef int i = 0
    cdef int j = n-1
    cdef int m
    while j-i>1:
        m = (i+j)//2
        if z > s.x[m]:
            i = m
        else:
            j = m
    return s.b[i] + 2*s.c[i]*(z-s.x[i]) + 3*s.d[i]*(z-s.x[i])*(z-s.x[i])

cdef double cspline_int(cspline * s, double z):
    # function to calculate the integral of a spline
    cdef int n = s.n
    if not (n>1 and z>=s.x[0] and z<=s.x[n-1]):
        raise ValueError('z does not lay in range of x')
    cdef int i = 0
    cdef int j = n-1
    cdef int m
    while j-i>1:
        m = (i+j)//2
        if z > s.x[m]:
            i = m
        else:
            j = m
    cdef double intsum = 0
    cdef int k
    cdef double deltax
    for k in range(i):
        deltax = s.x[k+1] - s.x[k]
        intsum += s.y[k]*deltax + 0.5*s.b[k]*deltax*deltax + s.c[k]*deltax*deltax*deltax/3.0 + s.d[k]*deltax*deltax*deltax*deltax/4.0
    intsum += s.y[i]*(z-s.x[i]) + 0.5*s.b[i]*(z-s.x[i])*(z-s.x[i]) + s.c[i]*(z-s.x[i])*(z-s.x[i])*(z-s.x[i])/3.0 + s.d[i]*(z-s.x[i])*(z-s.x[i])*(z-s.x[i])*(z-s.x[i])/4.0
    return intsum

cdef void cspline_free(cspline * s):
    free(s.x)
    free(s.y)
    free(s.b)
    free(s.c)
    free(s.d)
    free(s)

cdef struct Tree_Node:
    Tree_Node* child
    Tree_Node* sibling
    Tree_Node* parent
    int jlevel
    int nchild
    int index
    double mhalo


cdef Tree_Node* walk_tree(Tree_Node* this_node):
    '''
    Function to walk through the entire merger tree.
    ----------------------
    Input:
        this_node: Node from where to go to its child, sibling or into the next branch of the tree
                   untill the whole tree is walked through
    ----------------------
    Output:
        next_node: Either child, sibling or node in the next branch. NULL at the end of the tree
    '''
    cdef Tree_Node* next_node = this_node
    if next_node.child is not NULL:
        next_node = next_node.child
    else:
        if next_node.sibling is not NULL:
            next_node = next_node.sibling
        else:
            while next_node.sibling is NULL and next_node.parent is not NULL:
                next_node = next_node.parent
            if next_node.sibling is not NULL:
                next_node = next_node.sibling
            else:
                next_node = NULL
    return next_node

cdef node_vals_and_counter(int i,Tree_Node* this_node,int n_halo_max,Tree_Node** merger_tree):
    '''
    Function to count the number of nodes in a merger tree and get values out of it.
    ----------------------
    Input:
        i          : the level of the tree worked with in this call
        this_node  : the node where to start counting (best use to start with the base node)
        n_halo_max : maximum size of the merger tree
        merger_tree: list of the nodes of the given merger tree
    ----------------------
    Output:
        count     : the number of nodes inside the tree
        arr_halo  : the masses of the different nodes
        arr_nodid : the nodes id
        arr_treeid: the id of the tree
        arr_time  : the time level of the nodes
        arr_1prog : first progenitor of the nodes (-1 is no first progenitor)
        arr_desc  : descandant of the nodes (-1 is no descandant)
    '''
    cdef:
        int count = 0
        double* arr_mhalo = NULL
        int* arr_nodid = NULL
        int* arr_treeid = NULL
        int* arr_time = NULL
        int* arr_1prog = NULL
        int* arr_desc = NULL
        int* arr_nextprog = NULL
        int node_ID
        bint malloc_failed = 0 # checks if the allocation of the above pointers worked

    arr_mhalo = <double*>malloc(n_halo_max*sizeof(double))
    arr_nodid = <int*>malloc(n_halo_max*sizeof(int))
    arr_treeid = <int*>malloc(n_halo_max*sizeof(int))
    arr_time = <int*>malloc(n_halo_max*sizeof(int))
    arr_1prog = <int*>malloc(n_halo_max*sizeof(int))
    arr_desc = <int*>malloc(n_halo_max*sizeof(int))
    arr_nextprog = <int*>malloc(n_halo_max*sizeof(int))
    if not (arr_mhalo and arr_nodid and arr_treeid and arr_time and arr_1prog and arr_desc and arr_nextprog):
        malloc_failed = 1
    if malloc_failed:
        raise MemoryError('Failed to allocate memory for arrays in node_vals_counter function!')
    while this_node is not NULL:
        node_ID = this_node.index
        arr_nodid[node_ID] = node_ID
        arr_mhalo[node_ID] = this_node.mhalo
        arr_treeid[node_ID]= i
        arr_time[node_ID]  = this_node.jlevel
        if this_node.child is not NULL:
            if this_node.nchild > 1:
                arr_nextprog[node_ID] = this_node.child.sibling.index
            else:
                arr_nextprog[node_ID] = -1
            arr_1prog[node_ID] = this_node.child.index
        else:
            arr_1prog[node_ID] = -1
            arr_nextprog[node_ID] = -1
        if this_node.parent is not NULL:
            arr_desc[node_ID] = this_node.parent.index
        else:
            arr_desc[node_ID] = -1
        count +=1
        this_node = walk_tree(this_node)
    
    np_arr_mhalo = np.array([arr_mhalo[j] for j in range(count)])
    np_arr_nodid = np.array([arr_nodid[j] for j in range(count)],dtype='int_')
    np_arr_treeid= np.array([arr_treeid[j] for j in range(count)],dtype='int_')
    np_arr_time  = np.array([arr_time[j] for j in range(count)],dtype='int_')
    np_arr_1prog = np.array([arr_1prog[j] for j in range(count)],dtype='int_')
    np_arr_desc  = np.array([arr_desc[j] for j in range(count)],dtype='int_')
    np_arr_nextprog = np.array([arr_nextprog[j] for j in range(count)],dtype='int_')
    
    return (count,np_arr_mhalo,np_arr_nodid,np_arr_treeid,np_arr_time,np_arr_1prog,np_arr_desc,np_arr_nextprog)

cdef int tree_index(Tree_Node* node):
    '''
    Function to get the index of a certain node in a merger tree
    ----------------------
    Input:
        node: Node of which the index is needed
    ----------------------
    Output:
        index: Index of node inside the tree
    '''
    cdef int index
    if node is NULL:
        raise ValueError('Node is not associated with any tree')
    index = node.index
    return index

# class of the build up of the siblings from a certain Tree Node

#cpdef next_sibling(Tree_Node child_node,Tree_Node parent_node):
    '''
    Get the next sibling of the child node.
    '''
#    if child_node == parent_node:
#        if parent_node.child is not NULL:
#           Sibling = parent_node.child
#        else:
#            raise ValueError('next_sibling(): Parent has no children.')
#    else:
#        if child_node.sibling is not NULL:
#            Sibling = child_node.sibling
#        else:
#            raise ValueError('next_sibling(): Child has no siblings.')
#    sibs_left = Sibling.sibling
#    return Sibling, sibs_left

cdef Tree_Node** associated_siblings(Tree_Node* this_node, Tree_Node** merger_tree,int i):
    '''
    Function to associate the siblings of a certain node and as well sort them by their mass
    in descending order.
    ----------------------
    Input:
        this_node  : Node to which siblings should be associated
        merger_tree: Merger tree in which the other nodes are
    ----------------------
    Output:
        merger_tree: Updated merger tree
    '''
    cdef:
        int child_index, i_frag, k, j
        Tree_Node* temp = <Tree_Node*>malloc(sizeof(Tree_Node)) # temporary node to store the ones to change for mass ordering
    
    if this_node.nchild > 1:
        child_index = tree_index(this_node.child)
        # 1. Sort siblings in-place using mhalo
        for i_frag in range(child_index, child_index + this_node.nchild - 2):
            for j in range(child_index, child_index + this_node.nchild - 2 - i_frag):
                if merger_tree[j + 1].mhalo < merger_tree[j + 2].mhalo:
                    # Swap pointers
                    temp = merger_tree[j + 1]
                    merger_tree[j + 1] = merger_tree[j + 2]
                    merger_tree[j + 2] = temp
        # 2. Update sibling pointers
        for k in range(child_index, child_index + this_node.nchild - 1):
            merger_tree[k].sibling = merger_tree[k + 1]
    return merger_tree
'''
cdef Tree_Node** associated_siblings(Tree_Node* this_node,Tree_Node** merger_tree,int i):
    cdef int child_index, i_frag
    #print(this_node.nchild)
    if this_node.nchild > 1:
        #print('In here')
        child_index = tree_index(this_node.child)
        for i_frag in range(child_index, child_index + this_node.nchild -1):
            merger_tree[i_frag].sibling = merger_tree[i_frag + 1]
    return merger_tree
'''
cdef Tree_Node** build_sibling(Tree_Node** merger_tree,int n_frag_tot):
    '''
    Function to go through the whole merger tree and build the siblings of each node.
    ----------------------
    Input:
        merger_tree: Merger tree with all the nodes
        n_frag_tot : Number of nodes inside the tree
    ----------------------
    Output:
        merger_tree: Updated merger tree
    '''
    cdef int i
    for i in range(n_frag_tot):
        merger_tree = associated_siblings(merger_tree[i],merger_tree,i)
    return merger_tree

cdef class functions:
    # class to define the different functions used in the making of the tree
    cdef int n_table, n_v, n_sum
    cdef float a, a_min, delta_c
    cdef np.ndarray data
    cdef double l_0, omega_0
    def __init__(self,str filename):
        self.data = np.loadtxt(filename)
    def delta_crit(self,double a):
        '''
        Function to compute the critical delta.
        '''
        l_0 = trees.l_0
        omega_0 = trees.omega_0
        cdef:
            double[:] omega_flat= self.data[:,0]
            double[:] delflat   = self.data[:,-1] 
            int n_table = 200
            int n_v = 1000
            double a_min = 0.1
            double eps_om = 1e-5
            int n_sum = 2000
            double* delta_flat = <double*>malloc(n_table*sizeof(double))
            double* a_flat = <double*>malloc(n_table*sizeof(double))
            double omega_0_save = 0
            double l_0_save = 0
            int i_o
            double x_p, summ
        if fabs(1-omega_0) <= eps_om:
            del_c0 = 3*(12*pi)**(2/3)/20
            delta_c = del_c0/a
        elif (1-omega_0) > eps_om and l_0 < eps_om:
            eta_0 = acosh(2/omega_0-1)
            sh_0  = sinh(eta_0)
            ch_0  = sinh(eta_0)
            t_omega = 2*pi/(sh_0-eta_0)
            d_0   = 3*sh_0*(sh_0-eta_0)/(ch_0-1)**2-2

            ch = a*(ch_0-1)+1
            eta= acosh(ch)
            sh = sinh(eta)
            t  = (sh-eta)/(sh_0-eta_0)
            delta_c = 1.5*d_0*(1+(t_omega/t)**(2/3))

        else:
            if omega_0 != omega_0_save or l_0 != l_0_save:
                omega_0_save = omega_0
                l_0_save = l_0
                if fabs(omega_0+l_0-1) > eps_om:
                    raise ValueError('ERROR: omega_0+l_0+1 != 1')
                x_0 = (2*(1/omega_0-1))**0.333333
                summ= 0
                dxp = x_0/float(n_sum)
                for i_s in range(1,n_sum+1):
                    x_p = x_0*(float(i_s)-0.5)/float(n_sum)
                    if x_p < 0:
                        x_p = 0
                    summ += ((x_p/(x_p**3+2))**1.5)*dxp
                dlin_0 = summ*sqrt(x_0**3+2)/sqrt(x_0**3)
                
                for i in range(1,n_table+1):
                    aa = a_min+(1-a_min)*float(i-1)/float(n_table-1)
                    a_flat[i-1] = aa
                    l = l_0/(l_0+(1-omega_0-l_0)/aa**2+omega_0/aa**3)
                    omega = omega_0*l/(l_0*aa**3)
                    x = x_0*aa
                    summ = 0
                    dxp = x/float(n_sum)
                    for i_s in range(1,n_sum+1):
                        x_p = x*(float(i_s)-0.5)/float(n_sum)
                        if x_p<0:
                            x_p = 0
                        summ += ((x_p/(x_p**3+2))**1.5)*dxp
                    dlin = (summ*sqrt(x**3+2)/sqrt(x**3))/dlin_0
                    i_o = bisect.bisect_left(omega_flat,omega)
                    if i_o < n_v:
                        h = (omega_flat[i_o]-omega)/(omega_flat[i_o]-omega_flat[i_o-1])
                        delta_flat[i-1]=(delflat[i_o-1]*h+delflat[i_o]*(1-h))/dlin
                    else:
                        delta_flat[i-1]=delflat[n_v-1]/dlin
            if a > a_min and a <= 1.001:
                i = 1+int((a-a_min)*float(n_table-1)/(1-a_min))
                i = min(n_table-2,i)-1
                h = (a_flat[i+1]-a)/(a_flat[i+1]-a_flat[i])
                delta_c = delta_flat[i]*h+delta_flat[i+1]*(1-h)
            elif a <= a_min:
                delta_c = delta_flat[0]*a_min/a
            else:
                raise ValueError('delta_crit(): FATAL - look up tale only for a<1 \n a=',a)
        #free(delta_flat)
        #free(a_flat)
        return delta_c
cdef class sig_alph:
    cdef object trees
    cdef double G_used
    cdef np.ndarray k_0_np
    cdef np.ndarray Pk_0_np
    cdef double[:] Pk_0
    cdef double[:] k_0
    cdef double* log_m_r
    cdef double* sigma_temp
    cdef double* alpha_temp
    cdef cspline* sigma_spline
    cdef cspline* alpha_spline
    cdef int num_points
    cdef np.ndarray m_rough
    cdef np.ndarray m_array
    cdef np.ndarray log_m
    cdef np.ndarray Sig

    def __init__(self, object trees_obj):
        self.trees = trees_obj
        self.G_used = G.value * M_sun.value / (1e6 * pc.value * (1e3**2))
        self.k_0_np = np.asarray(self.trees.k_0_np, dtype=np.float64)
        self.Pk_0_np = np.asarray(self.trees.Pk_0_np, dtype=np.float64)
        self.k_0 = self.k_0_np
        self.Pk_0 = self.Pk_0_np
        self.num_points = 2000
        self.m_rough = np.geomspace(1e7, 1e17, self.num_points)
        self.m_array = np.geomspace(1e8, 1e17, self.num_points)
        self.Sig = np.zeros_like(self.m_rough)
        self.log_m = np.zeros(self.num_points)
        self.log_m_r = <double*>malloc(self.num_points * sizeof(double))
        self.sigma_temp = <double*>malloc(self.num_points * sizeof(double))
        self.alpha_temp = <double*>malloc(self.num_points * sizeof(double))
        self._precompute_sigma()
        self._precompute_alpha()

    # Helper function for integrand calculation
    cdef double my_int(self,double R, double k, double Pk) nogil:
        return 9.0 * (k * R * cos(k * R) - sin(k * R))**2.0 * Pk / k**4.0 / R**6.0 / (2.0 * pi**2.0)

    # Function to compute the sigma using simpson
    def sig_int(self,double m):
        h_0 = self.trees.h_0
        cdef double H_0100 = 100 * h_0
        cdef double rho_crit = 3 * H_0100**2 / (8 * pi * self.G_used)
        cdef double R = (3 * m / (4 * pi * rho_crit))**(1/3)
        cdef np.ndarray[double, ndim=1] my_integrand = np.empty_like(self.k_0)
        cdef int i
        for i in range(self.k_0.shape[0]):
            my_integrand[i] = self.my_int(R, self.k_0[i], self.Pk_0[i])
        return sqrt(simpson(my_integrand, self.k_0))

    # Sigma function
    cdef void _precompute_sigma(self):
        cdef int i
        for i in range(self.num_points):
            self.log_m_r[i] = np.log(self.m_rough[i])
            self.sigma_temp[i] = self.sig_int(self.m_rough[i])
            self.Sig[i] = log(self.sig_int(self.m_array[i]))
            self.log_m[i] = log(self.m_array[i])
        self.sigma_spline = cspline_alloc(self.num_points,self.log_m_r,self.sigma_temp)
    cdef void _precompute_alpha(self):
        cdef i
        for i in range(self.num_points):
            self.Sig[i] = log(self.sigma_cdm(self.m_array[i]))
            self.log_m[i] = log(self.m_array[i])
        interp = UnivariateSpline(self.log_m, self.Sig, k=4, s=0.1)
        for i in range(self.num_points):
            self.alpha_temp[i] = interp(log(self.m_rough[i]))
            self.log_m_r[i] = log(self.m_rough[i])
        self.alpha_spline = cspline_alloc(self.num_points,self.log_m_r,self.alpha_temp)
        
    # Interpolated sigma function
    cpdef double sigma_cdm(self,double m):
        return cspline_eval(self.sigma_spline, log(m))

    cpdef double alpha(self,double m):
        return cspline_deriv(self.alpha_spline,log(m))


cdef double eps = 1e-5
cdef double z_max = 10
cdef int N_TAB = 10000

cdef double J_pre(double z):
    cdef float J_tab[10000]
    cdef float z_tab[10000]
    cdef int i_first = 0
    cdef double h, J_un
    cdef int i
    cdef double dz = z_max/N_TAB
    cdef double inv_dz = 1.0/dz
    if fabs(gamma_1) > eps:
        if i_first == 0:
            if fabs(1.0-gamma_1) > eps:
                J_tab[0] = dz**(1.0-gamma_1)/(1.0-gamma_1)
            else:
                J_tab[0] = log(dz)
            z_tab[0] = dz
            for i in range(1,N_TAB):
                z_tab[i] = (i+1)*dz
                J_tab[i] = (J_tab[i-1]
                +(1.0+1.0/z_tab[i]**2)**(0.5*gamma_1)*0.5*dz
                +(1.0+1.0/z_tab[i-1]**2)**(0.5*gamma_1)*0.5*dz)
            i_first = 1
        i = int(z*inv_dz) - 1
        if i < 1:
            if fabs(1.0-gamma_1) > eps:
                J_un = (z**(1.0-gamma_1))/(1.0-gamma_1)
            else:
                J_un = log(z)
        elif i >= N_TAB-1:
            J_un = J_tab[N_TAB-1]+z-z_tab[N_TAB-1]
        else:
            h = (z-z_tab[i])*inv_dz
            J_un = J_tab[i]*(1.0-h)+J_tab[i+1]*h
    else:
        J_un = z
    return J_un

cdef int n = 400
cdef double* compute_J_arr(double* z_arr):
    cdef int i
    cdef double* J_arr = <double*>malloc(n*sizeof(double*))
    for i in range(n):
        J_arr[i] = log(J_pre(z_arr[i]))
    return J_arr
cdef double* z_arr = c_logspace(-3.0,4.0,n)
cdef double* J_arr = compute_J_arr(z_arr)
cdef double* log_z_arr = <double*>malloc(n*sizeof(double*))
for i in range(n):
    log_z_arr[i] = log(z_arr[i])

cdef cspline* J_spline = cspline_alloc(n,log_z_arr,J_arr)
cdef double J_unresolved(double z):
    log_J_used = cspline_eval(J_spline,log(z))
    return exp(log_J_used)

cdef double SQRT2OPI = 1.0/sqrt(pi/2.0)

cdef double min3(double a,double b,double c):
    if a<b and a<c:
        return a
    elif b<a and b<c:
        return b
    else:
        return c
cdef double min2(double a,double b):
    if a<b:
        return a
    else:
        return b

cdef extern from 'stdlib.h':
    int RAND_MAX

cdef struct split_result:
    double dw
    int n_prog
    double* m_prog
    double m_min_last

cdef split_result split(
    sig_alph sig,
    double m_2,
    double w,
    double m_min,
    double dw_max,
    double eps_1,
    double eps_2,
    double m_min_last):
    '''
    Function to calculate the time-step, number of progenitors and mass of the current node in
    the merger tree.
    ----------------------
    Input:
        m_2       : Mass of the current node
        w         : Current time
        m_min     : Minimum mass to resolve
        dw_max    : Maximum time-step dw
        eps_1     : Used inside formula of split
        eps_2     : Used inside formula of split
        m_min_last: Needed to get the correct sigma_cdm(m_min)
    ----------------------
    Output:
        dw        : Time-step
        n_prog    : Number of progenitors with a mass m > m_min
        m_prog    : Array containing those masses
        m_min_last: Needed to get the correct sigma_cdm(m_min)
    '''

    cdef:
        split_result res
        double* m_prog = <double*>malloc(2*sizeof(double))
        double eps_eta = 1e-6
        double eps_q = 6e-6
        
        double sig_q_min, sigsq_q_min, sigma_m2, sigsq_m2
        double sig_hf, sigsq_hf, alpha_hf, q_min, g_fac0
        double diff_q_min, diff12_q_min, diff32_q_min, v_q_min
        double diff_hf, diff12_hf, diff32_hf, v_hf, s_fac
        double beta, two_pow_beta, b, eta, half_pow_eta
        double eta_inv, q_min_exp, f_fac, dn_dw
        double dw_eps2, dw, n_av, z, f, mu
        int n_prog = 0

    if fabs(m_min - m_min_last) > eps_eta:
        sig_q_min = sig.sigma_cdm(m_min)
        sigsq_q_min = sig_q_min**2
        m_min_last = m_min
    else:
        sig_q_min = sig.sigma_cdm(m_min)
        sigsq_q_min = sig_q_min**2
        
    sigma_m2 = sig.sigma_cdm(m_2)
    sigsq_m2 = sigma_m2**2
    sig_hf = sig.sigma_cdm(0.5*m_2)
    sigsq_hf = sig_hf**2
    alpha_hf = -sig.alpha(0.5*m_2)
    q_min = m_min/m_2 # Minimum mass ratio

    g_fac0 = G_0*((w/sigma_m2)**gamma_2)

    if q_min < (0.5-eps_q):
        if gamma_1 <= 0:
            mu = -log(sig_q_min/sig_hf)/log(2*q_min)
        else:
            mu = alpha_hf
        g_fac1 = g_fac0*((sig_hf/sigma_m2)**gamma_1)/(2**(mu*gamma_1))

        diff_q_min = sigsq_q_min - sigsq_m2
        diff12_q_min = sqrt(diff_q_min)
        diff32_q_min = diff_q_min*diff12_q_min

        v_q_min = sigsq_q_min/diff32_q_min

        diff_hf = sigsq_hf - sigsq_m2
        diff12_hf = sqrt(diff_hf)
        diff32_hf = diff_hf*diff12_hf

        v_hf = sigsq_hf/diff32_hf
        s_fac = sqrt(2)*diff12_hf

        beta = log(v_q_min/v_hf)/log(2*q_min)
        if beta <= 0:
            raise ValueError('split(): beta <= 0')
        
        two_pow_beta = 2**beta
        b = v_hf*two_pow_beta
        eta = beta - 1 - gamma_1*mu

        half_pow_eta = 2**(-eta)
        if fabs(eta) > eps_eta:
            eta_inv = 1/eta
            q_min_exp = q_min**eta
            f_fac = half_pow_eta-q_min_exp
            dn_dw = SQRT2OPI*alpha_hf*b*eta_inv*f_fac*g_fac1
        else:
            dn_dw = -SQRT2OPI*alpha_hf*b*log(2*q_min)*g_fac1

        if dn_dw > 0:
            dw_eps2 = eps_2/dn_dw
        else:
            dw_eps2 = dw_max
        dw = min3(eps_1*s_fac,dw_eps2,dw_max)
        n_av = dn_dw*dw

        z = sigma_m2/diff12_q_min
        f = SQRT2OPI*dw*g_fac0*J_unresolved(z)/sigma_m2
        randy = rand()/float(RAND_MAX)
        if randy <= n_av:
            randy = rand()/float(RAND_MAX)
            if fabs(eta) > eps_eta:
                q_pow_eta = q_min_exp + randy*f_fac
                q = q_pow_eta**eta_inv
            else:
                q = q_min*(2*q_min)**(-randy)
            sig_q = sig.sigma_cdm(q*m_2)
            sigsq_q = sig_q**2
            alpha_q = -sig.alpha(q*m_2)
            diff12_q = sqrt(sigsq_q - sigsq_m2)
            v_q = sigsq_q/diff12_q**3

            R_q = (alpha_q/alpha_hf)*((sig_q*(2*q)**mu/sig_hf)**gamma_1)*v_q/(b*q**beta)

            if R_q > 1.00001:
                raise ValueError('split(): R_q > 1')
            randy = rand()/float(RAND_MAX)
            if randy >= R_q:
                q = 0
        else:
            q = 0
        m_prog[1] = q*m_2
        if m_prog[1] <= m_min:
            n_prog = 1
        else:
            n_prog = 2
        m_prog[0] =(1-q-f)*m_2
        
        if m_prog[0] <= m_min:
            n_prog -= 1
            m_prog[0] = m_prog[1]
            m_prog[1] = 0
        elif m_prog[0] < m_prog[1]:
            m_temp = m_prog[0]
            m_prog[0] = m_prog[1]
            m_prog[1] = m_temp
        
    else:
        diff_hf = sigsq_hf - sigsq_m2
        diff12_hf = sqrt(diff_hf)
        s_fac = sqrt(2)*diff12_hf
        dw = min2(eps_1*s_fac,dw_max)

        diff_q_min = sigsq_q_min - sigsq_m2
        if diff_q_min < 0:
            diff_q_min = 0
        diff12_q_min = sqrt(diff_q_min)
        if diff12_q_min > SQRT2OPI*dw:
            z = sigma_m2/diff12_q_min
            f = SQRT2OPI*dw*g_fac0*J_unresolved(z)/sigma_m2
        else:
            f = 1
        m_prog[0] = (1-f)*m_2
        if m_prog[0] > m_min:
            n_prog = 1
        else:
            n_prog = 0
            m_prog[0] = 0
        m_prog[1] = 0
    res.dw = dw
    res.n_prog = n_prog
    res.m_prog = m_prog
    res.m_min_last = m_min_last
    return res


cdef locate(double[:] xx,int n,double x):
    # Function to locate an element x inside a pointer xx of size n
    cdef int jl = 0
    cdef int ju = n+1
    cdef int jm
    while ju-jl > 1:
        jm = (ju+jl)//2
        if ((xx[n-1] > xx[0]) == (x > xx[jm-1])):
            jl = jm
        else:
            ju = jm
    return int(jl-1)

cdef int n_switch = 60
cdef double tiny = 1e-5
cdef double aln2I = 1.0/log(2.0)

cdef int* indexxx(int n,double* arr,int* indx):
    if n > n_switch:
        indx = indexx(n, arr, indx)
    else:
        indx = indexsh(n, arr, indx)
    return indx

cdef int* indexx(int n,double* arr,int* indx):
    '''
    Indexing or array arr[n] using numerical recipes heapsort.
    '''
    cdef int l,ir,idnxt
    cdef double q
    if n <= 1:
        return indx
    l = n//2
    ir = n-1
    while True:
        if l > 0:
            l -= 1
            indxt = indx[l]
            q = arr[indxt]
        else:
            indxt = indx[ir]
            q = arr[indxt]
            indx[ir] = indx[0]
            ir -= 1
            if ir == 0:
                indx[0] = indxt
                break
        i = l
        j = 2*l + 1
        while j <= ir:
            if j < ir and arr[indx[j]] < arr[indx[j+1]]:
                j += 1
            if q < arr[indx[j]]:
                indx[i] = indx[j]
                i = j
                j = 2*j+1
            else:
                break
        indx[i] = indxt
    return indx
 
cdef int* indexsh(int n,double* arr,int* indx):
    '''
    Indexing or array arr[n] using shell sort.
    '''
    cdef int m,log_nb2,k,i,j,t_index,n_n
    for i in range(n):
        indx[i] = i
    if n >= 2:
        log_nb2 = int(log(n)*aln2I+tiny)
        m = n
        for n_n in range(log_nb2):
            m = m//2
            k = n - m
            i = 0
            for j in range(int(k)):
                i=j
                done3 = False
                while done3 is False:
                    l = i + m
                    if arr[indx[l]] < arr[indx[i]]:
                        t_index = indx[i]
                        indx[i] = indx[l]
                        indx[l] = t_index
                        i = i - m
                        if i < 0:
                            done3 = True
                    else:
                        done3 = True
    return indx

cdef Tree_Node** make_tree(double m_0,double a_0,double m_min,double[:] w_lev,int n_lev,int n_frag_max,int n_frag_tot=0):
    '''
    Function to make the merger tree of a certain mass.
    ----------------------
    Input:
        m_0       : Mass of the halo at the beginning of the current tree (also base node mass)
        a_0       : Value of scale factor today or up to which time the tree should be calculated
        m_min     : Minimum mass; scale at which the mass is not resolveable
        a_lev     : Array of the different times for which to take snapshots of the tree
        n_lev     : Number of time levels
        n_frag_max: Maximum number of halos in one tree
        n_frag_tot: Start of counter of nodes inside the tree
    ----------------------
    Output:
        merger_tree: Merger tree of given mass, with parents, siblings, children and mass
    '''
    cdef int n_v
    n_v = int(n_frag_max/5)
    cdef:
        split_result split_fct
        Tree_Node** merger_tree = <Tree_Node**>malloc(n_frag_max*sizeof(Tree_Node*))
        Tree_Node* node
        sig_alph Sig
        int i_err = 0
        int j_frag = -1
        int n_ch
        int k_child
        int j_frag_c
        int i_frag
        int i_frag_prev, i_par_prev
        int k, j, iw, i
        int* i_frag_lev = <int*>malloc(n_lev*sizeof(int))
        double dw, m
        int n_prog
        double* m_prog = <double*>malloc(2*sizeof(double))
        double w, w_fin, m_min_last
        int i_lev, i_lev_wk, i_node
        int* child_ref = <int*>malloc(n_frag_max*sizeof(int))
        int* j_index = <int*>malloc(n_frag_max*sizeof(int))
        double* m_tr = <double*>malloc(n_frag_max*sizeof(double))
        int* i_par = <int*>malloc(n_frag_max*sizeof(int))
        int* i_sib = <int*>malloc(n_frag_max*sizeof(int))
        int* i_child = <int*>malloc(n_frag_max*sizeof(int))
        int* n_frag_lev = <int*>malloc(n_lev*sizeof(int))
        int* jp_frag = <int*>malloc(n_lev*sizeof(int))
        double* m_left = <double*>malloc(n_v*sizeof(double))
        double* m_right = <double*>malloc(n_v*sizeof(double))
        double* w_node = <double*>malloc(n_v*sizeof(double))
        int* l_node = <int*>malloc(n_v*sizeof(int))
    n_frag_tot = 0
    Sig = sig_alph(trees)
    for i_frag in range(n_frag_max):
        node = <Tree_Node*>malloc(sizeof(Tree_Node))
        node.mhalo = 0.0
        node.jlevel = 0
        node.nchild = 0
        node.index  = i_frag
        node.parent = NULL
        node.child  = NULL
        node.sibling= NULL
        merger_tree[i_frag] = node
        m_tr[i_frag] = 0

    for i_lev in range(n_lev):
        i_frag_lev[i_lev] = -1
        jp_frag[i_lev] = -1
        n_frag_lev[i_lev] = 0

    for i in range(n_v):
        m_left[i] = 0       
        m_right[i]= 0
        w_node[i] = 0
        l_node[i] = -1
    i_node = 0
    w_fin = w_lev[n_lev-1]
    
    m_left[0] = m_0
    m_right[0] = -1
    m = m_0
    w_node[0] = w_lev[0]
    w = w_lev[0]
    i_lev = 1
    i_frag_lev[0] = 0

    i_frag = 0
    m_tr[0] = m_0
    m_minlast= 0.0
    i_par[0] = -1
    i_sib[0] = -1
    i_child[0] = -1
    while m_left[i_node] > 0.0:
        dw_max = w_lev[i_lev] - w
        split_fct = split(Sig, m, w, m_min, dw_max, eps_1, eps_2,m_minlast)
        dw = split_fct.dw
        n_prog = split_fct.n_prog
        m_prog = split_fct.m_prog
        m_minlast = split_fct.m_min_last
        w += dw
        # Now setting some of the values to according to the number of progenitors
        # n_prog (=2,1,0)
        if n_prog==2:
            i_node += 1
            w_node[i_node] = w
            l_node[i_node] = -1
            m_left[i_node] = m_prog[0]
            m_right[i_node] = m_prog[1]
            m = m_left[i_node]
        elif n_prog==1:
            m = m_prog[0]
        elif n_prog==0:
            m = 0.0
            m_left[i_node] = -1
        # Asign indices to parents, siblings and children at the right point
        if w == w_lev[i_lev] and  n_prog > 0:
            if i_frag+2 > n_frag_max:
                print('Here!!!')
                print(i_frag)
                i_err = 1
                return NULL

            i_frag += 1     
            m_tr[i_frag] = m_prog[0]
            i_par[i_frag] = i_frag_lev[i_lev-1]
            i_sib[i_frag] = -1
            i_child[i_frag] = -1
        
            i_frag_prev = i_frag_lev[i_lev]
            if i_frag_prev >= 0:
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
                l_node[i_node] = 1
                if i_lev == n_lev-1:
                    i_frag_lev[i_lev] = i_frag
            if i_lev < n_lev-1:
                i_lev +=1
        # Drop out values that are out of the chosen time-window
        if w >= w_fin:
            if n_prog == 2:
                m_left[i_node] = -1
                m_right[i_node] = -1
            else:
                m_left[i_node] = -1
        # Reorder the masses to have the non-negative in the pointer for the 
        # left masses or both =-1  
        while m_left[i_node] < 0.0 and i_node > 0:
            if m_right[i_node] > 0.0:
                m_left[i_node] = m_right[i_node]
                m_right[i_node] = -1
                w = w_node[i_node]
                m = m_left[i_node]
                if w < w_lev[i_lev-1] or w >= w_lev[i_lev]:
                    iw = locate(w_lev, n_lev, w)
                    i_lev = iw + 1
                    if w_lev[i_lev] == w:
                        i_lev += 1
                if l_node[i_node]==1:
                    i_frag_lev[i_lev-1] += 1
            else:
                i_node -= 1
                m_left[i_node] = -1

    n_frag_tot = i_frag # Number of nodes in merger tree - 1
    
    # Asigning the parents of nodes in the merger tree, as well as masses
    # and time-level
    for i_lev_wk in range(n_lev):
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
                    j_frag += 1
                    j_index[i_frag] = j_frag
                    n_frag_lev[i_lev_wk] += 1
                    merger_tree[j_frag].mhalo = m_tr[i_frag]
                    merger_tree[j_frag].jlevel = i_lev_wk

                    if i_lev > 0:
                        merger_tree[j_frag].parent = merger_tree[j_index[i_par[i_frag]]]
                    else:
                        merger_tree[j_frag].parent = NULL
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
    # Asigning children and the number of children of certain nodes
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

            merger_tree[j_frag].nchild = n_ch
    '''
    cdef:
        int k_frag_p, n_child1, j_child1
        int k_frag = 0
        int n_ch_max = 100000
        int* indx_ch = <int*>malloc(n_ch_max*sizeof(int))
        double* temp
    j_index[0] = 0
    m_tr[0] = merger_tree[0].mhalo
    for i_lev in range(n_lev-1):
        #print('start point range: ',jp_frag[i_lev])
        #print('end point range: ',jp_frag[i_lev]+n_frag_lev[i_lev])
        for k_frag_p in range(jp_frag[i_lev],jp_frag[i_lev]+n_frag_lev[i_lev]):
            #print('k_frag_p = ',k_frag_p)
            j_frag_p = j_index[k_frag_p]
            n_child1 = merger_tree[j_frag_p].nchild
            #print(n_child1)
            i_sib[k_frag_p] = n_child1
            j_child1 = child_ref[j_frag_p]
            if n_child1==1:
                #print('In here')
                indx_ch[0] = 0
                #print('After this')
            elif n_child1 >= 2:
                if n_child1 > n_ch_max:
                    print('make_tree: n_child1 > n_ch_max increase n_ch_max \n n_child1 = ',n_child1)
                temp = <double*>malloc(n_child1*sizeof(double))
                for i in range(n_child1):
                    temp[i] = merger_tree[i+j_child1].mhalo
                    #print(temp[i])
                indx_ch = indexxx(n_child1,temp,indx_ch)
                free(temp)
            if n_child1 >= 1:
                #print('Made it till here')
                i_child[k_frag_p] = k_frag + 1
            else:
                i_child[k_frag_p] = -1
            #print('We are here')
            for k_child in range(n_child1-1,-1,-1):
                j_frag_c = indx_ch[k_child] + j_child1  # Adjust for Python's 0-based indexing
                #print('j_frag_c = ',j_frag_c)
                k_frag += 1
                j_index[k_frag] = j_frag_c  # Adjust for Python's 0-based indexing
                merger_tree[k_frag].parent = merger_tree[k_frag_p]  # Assign parent as a reference
                m_tr[k_frag] = merger_tree[j_frag_c].mhalo  # Store mhalo value in mtr array
    for k_frag in range(jp_frag[n_lev-1],jp_frag[n_lev-1]+n_frag_lev[n_lev-1]):
        i_sib[k_frag] = 0
        i_child[k_frag] = -1
    for j in range(n_frag_tot):
        merger_tree[j].mhalo = m_tr[j]
        #print(merger_tree[j].mhalo)
    for i_frag in range(n_frag_tot):
        if i_child[i_frag] > 0:
            merger_tree[i_frag].child = merger_tree[i_child[i_frag]]
        else:
            merger_tree[i_frag].child = NULL    
    for k in range(n_frag_tot):
       merger_tree[k].nchild = i_sib[k]
       #print(merger_tree[k].nchild)
    '''
    
    # Build the siblings, also in decreasing mass order        
    merger_tree = build_sibling(merger_tree,n_frag_tot)
    free(i_par)
    free(i_sib)
    free(i_child)
    free(node)

    i_err = 0
    return merger_tree

def get_tree_vals(
    int i,
    int i_seed_0,
    double m_0,
    double a_0,
    double m_min,
    double[:] w_lev,
    double[:] a_lev,
    int n_lev,
    int n_frag_max,
    int n_frag_tot=0):
    '''
    Function that builds the merger tree and returns the data that is needed for a later analysis.
    ---------------------
    Input:
        i         : Number of tree produced now
        i_seed_0  : Used for seed to generate random numbers
        m_0       : Mass of the halo at the beginning of the current tree (also base node mass)
        a_0       : Value of scale factor today or up to which time the tree should be calculated
        m_min     : Minimum mass; scale at which the mass is not resolveable
        a_lev     : Array of the different times for which to take snapshots of the tree
        n_lev     : Number of time levels
        n_frag_max: Maximum number of halos in one tree
        n_frag_tot: Start of counter of nodes inside the tree
    ----------------------
    Output:
        count     : Counter that counts the length/number of nodes in the tree
        arr_mhalo : Array of masses of the different halos inside the tree
        arr_nodid : Array of node ID's inside the tree
        arr_treeid: Array of the tree's ID
        arr_time  : Array of the time levels of the different nodes
        arr_1prog : Array of the first progenitor of the different nodes
        arr_desc  : Array of the descandents of the different nodes
    '''

    i_seed_0 -=19*(1+i)
    cdef:
        int i_seed = i_seed_0
        Tree_Node** merger_tree
        Tree_Node* this_node
        int count = 0
    srand(i_seed)
    #print('Going into make_tree')
    merger_tree = make_tree(m_0,a_0,m_min,w_lev,n_lev,n_frag_max,n_frag_tot)

    #print('Made a tree ',i+1)
    this_node = merger_tree[0]
    count,arr_mhalo,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc,arr_nextprog = node_vals_and_counter(i,this_node,n_frag_max,merger_tree)

    print('Number of nodes in tree',i+1,'is',count)
    
    print('Example information from tree:')
    this_node = merger_tree[0]
    print('Base node: \n  mass =',this_node.mhalo,' z= ',1/a_lev[this_node.jlevel]-1,' number of progenitors ',this_node.nchild)
    if count>1:
        this_node = this_node.child
        print('First progenitor: \n  mass =',this_node.mhalo,' z= ',1/a_lev[this_node.jlevel]-1)
    else:
        print('No Progenitors.')
    free(merger_tree)
    free(this_node)
    return count,arr_mhalo,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc,arr_nextprog