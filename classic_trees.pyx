import cython
import numpy as np
cimport numpy as np
import bisect
import random
#from libc.stdlib cimport rand, srand
from scipy.integrate import simpson
from astropy.constants import G, M_sun, pc
from libc.math cimport sqrt, log, exp, pi, cos, sin, fabs, acosh, sinh, cosh
from scipy.interpolate import UnivariateSpline, CubicSpline
from libc.stdlib cimport malloc, free

ctypedef np.float64_t DTYPE_t
DTYPE = np.float64


cdef double* c_logspace(double start, double stop,int n):
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

cdef class Tree_Node:
    # definition of the class of Tree Nodes
    cdef public Tree_Node child, sibling, parent
    cdef public int jlevel, nchild
    cdef public float mhalo

    def __init__(self):
        # starting values of the different elements of Tree Nodes
        self.mhalo = 0.0
        self.jlevel = 0
        self.nchild = 0
    
        self.child = None
        self.parent = None
        self.sibling = None
'''

cdef struct Tree_Node:
    Tree_Node* child
    Tree_Node* sibling
    Tree_Node* parent
    int jlevel
    int nchild
    int index
    double mhalo
'''

cdef walk_tree(Tree_Node this_node):
    '''
    Walk through the entire merger tree.
    '''
    cdef Tree_Node next_node = this_node
    if next_node.child is not None:
        next_node = next_node.child
    else:
        if next_node.sibling is not None:
            next_node = next_node.sibling
        else:
            while next_node.sibling is None and next_node.parent is not None:
                next_node = next_node.parent
            if next_node.sibling is not None:
                next_node = next_node.sibling
            else:
                next_node = None
    return next_node

def node_vals_and_counter(int i,Tree_Node this_node,int n_halo_max,list merger_tree):
    '''
    Function to count the number of nodes in a merger tree and get values out of it.
    Input:
        i          : the level of the tree worked with in this call
        this_node  : the node where to start counting (best use to start with the base node)
        n_halo_max : maximum size of the merger tree
        merger_tree: list of the nodes of the given merger tree
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
        double* arr_mhalo = <double*>malloc(n_halo_max*sizeof(double))
        int* arr_nodid = <int*>malloc(n_halo_max*sizeof(int))
        int* arr_treeid = <int*>malloc(n_halo_max*sizeof(int))
        int* arr_time = <int*>malloc(n_halo_max*sizeof(int))
        int* arr_1prog = <int*>malloc(n_halo_max*sizeof(int))
        int* arr_desc = <int*>malloc(n_halo_max*sizeof(int))
        int node_ID
    while this_node is not None:
        node_ID = count
        arr_nodid[node_ID] = node_ID
        arr_mhalo[node_ID] = this_node.mhalo
        arr_treeid[node_ID]= i
        arr_time[node_ID]  = this_node.jlevel
        if this_node.child is not None:
            arr_1prog[node_ID] = merger_tree.index(this_node.child)
        else:
            arr_1prog[node_ID] = -1
        if this_node.parent is not None:
            arr_desc[node_ID] = merger_tree.index(this_node.parent)
        else:
            arr_desc[node_ID] = -1
        count +=1
        this_node = walk_tree(this_node)
    cdef int j
    np_arr_mhalo = np.array([arr_mhalo[j] for j in range(count)])
    np_arr_nodid = np.array([arr_nodid[j] for j in range(count)],dtype='int_')
    np_arr_treeid= np.array([arr_treeid[j] for j in range(count)],dtype='int_')
    np_arr_time  = np.array([arr_time[j] for j in range(count)],dtype='int_')
    np_arr_1prog = np.array([arr_1prog[j] for j in range(count)],dtype='int_')
    np_arr_desc  = np.array([arr_desc[j] for j in range(count)],dtype='int_')
    
    return count,np_arr_mhalo,np_arr_nodid,np_arr_treeid,np_arr_time,np_arr_1prog,np_arr_desc

cdef class Tree_Values:
    # class of different operations on the merger trees,
    # first some definitions of variables:
    cdef Tree_Node node
    cdef list merger_tree
    def tree_index(self,node,merger_tree):
        # finding the Tree Node index of inside a merger tree
        if node is None:
            raise ValueError('Node is not associated with any tree')
        index = merger_tree.index(node)
        return index

# class of the build up of the siblings from a certain Tree Node

cpdef next_sibling(Tree_Node child_node,Tree_Node parent_node):
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

cdef list associated_siblings(Tree_Node this_node,list merger_tree):
    if this_node.nchild > 1:
        child_index = Tree_Values().tree_index(this_node.child,merger_tree)
        sib_nodes = []
        merger_temp = []
        for i_frag in range(child_index, child_index + this_node.nchild -1):
            sib_nodes.append([merger_tree[i_frag+1],i_frag])
        sib_sorted = sorted(sib_nodes,key=lambda x:x[0].mhalo,reverse=True)
        n = len(sib_sorted)
        for i in range(n):
            merger_temp.append(merger_tree[sib_sorted[i][1]+1])
        for k in range(n):
            merger_tree[sib_nodes[k][1]+1] = merger_temp[k]
        for j in range(n):
            merger_tree[sib_nodes[j][1]].sibling = merger_tree[sib_nodes[j][1]+1]
    return merger_tree
'''
cdef Tree_Node** associated_siblings(Tree_Node* this_node, Tree_Node** merger_tree) nogil:
    cdef:
        int child_index, i_frag, k, j
        Tree_Node* temp
    
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
cdef list build_sibling(list merger_tree,int n_frag_tot):
    for i in range(n_frag_tot):
        merger_tree = associated_siblings(merger_tree[i],merger_tree)
    return merger_tree

cdef class functions:
    # class to define the different functions used in the making of the tree
    cdef int n_table, n_v, n_sum
    cdef float a, a_min, delta_c
    cdef np.ndarray data
    def __init__(self,str filename):
        self.data = np.loadtxt(filename)
    def delta_crit(self,double a,double l_0=0.75,double omega_0=0.25):
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
            double x_p
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

# Constants
cdef double G_used = G.value * M_sun.value / (1e6 * pc.value * (1e3**2))
cdef double H_0100 = 100 * 0.73
cdef double rho_crit = 3 * H_0100**2 / (8 * pi * G_used)

# Load data from file
cdef np.ndarray pk_data = np.loadtxt('./CLASSIC-trees/pk_CLASS.txt')
cdef np.ndarray k_0_np = pk_data[0]
cdef np.ndarray Pk_0_np = pk_data[1] * 0.73**3
cdef int n_Pk_k = len(pk_data[0])

cdef double[:] Pk_0 = Pk_0_np
cdef double[:] k_0 = k_0_np
cdef int i_k
#for i_k in range(n_Pk_k):
#    k_0[i_k] = k_0_np[i_k]
#    Pk_0[i_k] = Pk_0_np[i_k]
# Precompute sigma values for interpolation
cdef np.ndarray m_rough = np.geomspace(1e7, 1e15, 500)
cdef np.ndarray Sig = np.zeros_like(m_rough)

# Helper function for integrand calculation
cdef double my_int(double R, double k, double Pk) nogil:
    return 9.0 * (k * R * cos(k * R) - sin(k * R))**2.0 * Pk / k**4.0 / R**6.0 / (2.0 * pi**2.0)
    

def sig_int(double m):
    cdef double R = (3 * m / (4 * pi * rho_crit))**(1/3)
    cdef np.ndarray[double, ndim=1] my_integrand = np.empty_like(k_0)
    cdef int i
    for i in range(k_0.shape[0]):
        my_integrand[i] = my_int(R, k_0[i], Pk_0[i])
    return sqrt(simpson(my_integrand, k_0))

# Sigma function
'''
cdef double sigma(double m):
    cdef double sigma_val
    cdef double* sigma_temp = <double*>malloc(500*sizeof(double))
    cdef double* m_rough = c_logspace(7.0,15.0,500)
    cdef int i
    cdef double* log_m_r = <double*>malloc(500*sizeof(double))
    for i in range(500):
        log_m_r[i] = log(m_rough[i])
    cdef cspline* sigma_spline
    try:
        for i in range(500):
            sigma_temp[i] = log(sig_int(m))
        sigma_spline = cspline_alloc(500,log_m_r,sigma_temp)
        try:
            sigma_val = cspline_eval(sigma_spline,log(m))
        finally:
            cspline_free(sigma_spline)
        return sigma_val
    finally:
        free(sigma_temp)
        free(m_rough)
        free(log_m_r)
'''
cdef double* log_m_r = <double*>malloc(500*sizeof(double))
cdef double* sigma_temp = <double*>malloc(500*sizeof(double))
cdef double* alpha_temp = <double*>malloc(500*sizeof(double))
cdef int i
for i in range(m_rough.shape[0]):
    log_m_r[i] = np.log(m_rough[i])
    sigma_temp[i] = sig_int(m_rough[i])
    Sig[i] = sig_int(m_rough[i])
cdef cspline* sigma_spline = cspline_alloc(500,log_m_r,sigma_temp)
# Interpolated sigma function
def sigma_cdm(double m):
    return cspline_eval(sigma_spline, log(m))#exp(np.interp(log(m),np.log(m_rough),np.log(Sig)))


cdef double m_min=1e8
cdef double m_max=1e15
cdef int num_points=500
cdef np.ndarray m_array = np.geomspace(m_min, m_max, num_points, dtype=DTYPE)
cdef log_sig = compute_log_sig(m_array)
log_m = np.log(m_array)
interp = UnivariateSpline(log_m, log_sig, k=4, s=0.1)
deriv = interp.derivative(n=1)

cdef compute_log_sig(np.ndarray m_array):
    cdef int i
    cdef int n = m_array.shape[0]
    cdef double log_sig[500]
    
    for i in range(n):
        log_sig[i] = log(sigma_cdm(m_array[i]))
    return np.asarray(log_sig)
for i in range(m_rough.shape[0]):
    alpha_temp[i] = interp(log(m_rough[i]))
cdef cspline* alpha_spline = cspline_alloc(500,log_m_r,alpha_temp)
def alpha(double m):
    return cspline_deriv(alpha_spline,log(m))#float(deriv(log(m)))


cdef double eps = 1e-5
cdef double z_max = 10
cdef int N_TAB = 10000
cdef double gamma_1=0.38

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
cdef double* z_arr = c_logspace(-3.0,3.0,n)
cdef double* J_arr = compute_J_arr(z_arr)
cdef double* log_z_arr = <double*>malloc(n*sizeof(double*))
for i in range(n):
    log_z_arr[i] = log(z_arr[i])
#compute_J_arr(z_arr, J_arr)

#log_J_used = CubicSpline(np.log(z_arr),J_arr,bc_type='natural') 
cdef cspline* J_spline = cspline_alloc(n,log_z_arr,J_arr)
def J_unresolved(double z):
    log_J_used = cspline_eval(J_spline,log(z))
    #return J_pre(z)
    return exp(log_J_used)

cdef double SQRT2OPI = 1.0/sqrt(pi/2.0)

cpdef split(
    double m_2,
    double w,
    double m_min,
    double dw_max,
    double eps_1,
    double eps_2,
    double m_min_last):

    cdef:
        double m_prog[2]
        double eps_eta = 1e-6
        double eps_q = 6e-6
        double gamma_1 = 0.38
        double gamma_2 = -0.01
        double G_0 = 0.57

        double sig_q_min, sigsq_q_min, sigma_m2, sigsq_m2
        double sig_hf, sigsq_hf, alpha_hf, q_min, g_fac0
        double diff_q_min, diff12_q_min, diff32_q_min, v_q_min
        double diff_hf, diff12_hf, diff32_hf, v_hf, s_fac
        double beta, two_pow_beta, b, eta, half_pow_eta
        double eta_inv, q_min_exp, f_fac, dn_dw
        double dw_eps2, dw, n_av, z, f
        int n_prog = 0

    if fabs(m_min - m_min_last) > eps_eta:
        sig_q_min = sigma_cdm(m_min)
        sigsq_q_min = sig_q_min**2
        m_min_last = m_min
    else:
        sig_q_min = sigma_cdm(m_min)
        sigsq_q_min = sig_q_min**2
        
    sigma_m2 = sigma_cdm(m_2)
    sigsq_m2 = sigma_m2**2
    sig_hf = sigma_cdm(0.5*m_2)
    sigsq_hf = sig_hf**2
    alpha_hf = -alpha(0.5*m_2)
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
        dw = min(eps_1*s_fac,dw_eps2,dw_max)
        n_av = dn_dw*dw

        z = sigma_m2/diff12_q_min
        f = SQRT2OPI*dw*g_fac0*J_unresolved(z)/sigma_m2
        randy = random.random()
        if randy <= n_av:
            randy = random.random()
            if fabs(eta) > eps_eta:
                q_pow_eta = q_min_exp + randy*f_fac
                q = q_pow_eta**eta_inv
            else:
                q = q_min*(2*q_min)**(-randy)
            sig_q = sigma_cdm(q*m_2)
            sigsq_q = sig_q**2
            alpha_q = -alpha(q*m_2)
            diff12_q = sqrt(sigsq_q - sigsq_m2)
            v_q = sigsq_q/diff12_q**3

            R_q = (alpha_q/alpha_hf)*((sig_q*(2*q)**mu/sig_hf)**gamma_1)*v_q/(b*q**beta)

            if R_q > 1.00001:
                raise ValueError('split(): R_q > 1')
            randy = random.random()
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
        dw = min(eps_1*s_fac,dw_max)

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
    return dw, n_prog, np.asarray(m_prog),m_min_last


cdef locate(double* xx,int n,double x):
    cdef int jl = 0
    cdef int ju = n+1
    while ju-jl > 1:
        jm = (ju+jl)//2
        if ((xx[n-1] > xx[0]) == (x > xx[jm-1])):
            jl = jm
        else:
            ju = jm
    return int(jl-1)

cdef str filename = './CLASSIC-trees/Data/flat.txt'
cdef DELTA = functions(filename)

cdef list make_tree(double m_0,double a_0,double m_min,double[:] a_lev,int n_lev,int n_frag_max,int n_frag_tot=0):
    cdef:
        list merger_tree = [None]*n_frag_max #<Tree_Node**>malloc(n_frag_max*sizeof(Tree_Node*))
        Tree_Node node
        int n_v = 20000
        int i_err = 0
        int i_lev
        np.ndarray child_ref = np.zeros(n_frag_max) #[0]*n_frag_max
        np.ndarray j_index = np.zeros(n_frag_max,dtype='int_') #[0]*n_frag_max
        np.ndarray m_tr = np.zeros(n_frag_max) #[0]*n_frag_max
        np.ndarray i_par = np.zeros(n_frag_max,dtype='int_') #[0]*n_frag_max
        np.ndarray i_sib = np.zeros(n_frag_max,dtype='int_') #[0]*n_frag_max
        np.ndarray i_child = np.zeros(n_frag_max,dtype='int_') #[0]*n_frag_max
        list w_lev = [0]*n_lev
        list n_frag_lev = [-1]*n_lev
        list jp_frag = [-1]*n_lev
        np.ndarray m_left = np.zeros(n_v) #[0]*n_v
        np.ndarray m_right = np.zeros(n_v) #[0]*n_v
        np.ndarray w_node = np.zeros(n_v) #[0]*n_v
        np.ndarray l_node = np.zeros(n_v,dtype='bool') #[False]*n_v
    
    for i_frag in range(n_frag_max):
        node = Tree_Node()
        node.mhalo = 0.0
        node.jlevel = 0
        node.nchild = 0
        node.parent = None
        node.child  = None
        node.sibling= None
        merger_tree[i_frag] = node
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
                return None

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
            merger_tree[j_frag].nchild = n_ch

    merger_tree = build_sibling(merger_tree,n_frag_tot)
    
    i_err = 0
    return merger_tree

def get_tree_vals(
    int i,
    int i_seed_0,
    double m_0,
    double a_0,
    double m_min,
    double[:] a_lev,
    int n_lev,
    int n_frag_max,
    int n_frag_tot=0):

    i_seed_0 -=19
    cdef:
        int i_seed = i_seed_0
        list merger_tree
        Tree_Node this_node
        int count = 0
    random.seed(i_seed)
    merger_tree = make_tree(m_0,a_0,m_min,a_lev,n_lev,n_frag_max,n_frag_tot)

    print('Made a tree ',i+1)
    this_node = merger_tree[0]
    count,arr_mhalo,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc = node_vals_and_counter(i,this_node,n_frag_max,merger_tree)

    print('Number of nodes in tree',i+1,'is',count)
    
    print('Example information from tree:')
    this_node = merger_tree[0]
    print('Base node: \n  mass =',this_node.mhalo,' z= ',1/a_lev[this_node.jlevel]-1,' number of progenitors ',this_node.nchild)
    this_node = this_node.child
    print('First progenitor: \n  mass =',this_node.mhalo,' z= ',1/a_lev[this_node.jlevel]-1)
    this_node = this_node.sibling
    print('  mass =',this_node.mhalo,' z= ',1/a_lev[this_node.jlevel]-1)
    return count,i_seed,arr_mhalo,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc