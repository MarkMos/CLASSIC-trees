import cython
import numpy as np
cimport numpy as np
import bisect
import random
#from libc.stdlib cimport rand, srand
from scipy.integrate import simpson
from astropy.constants import G, M_sun, pc
from libc.math cimport sqrt, log, exp, pi, cos, sin, fabs
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
    assert(n>1 and z>=s.x[0] and z<=s.x[n-1])
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
    assert(n>1 and z>=s.x[0] and z<=s.x[n-1])
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
    assert(n>1 and z>=s.x[0] and z<=s.x[n-1])
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

def node_vals_and_counter(int count,Tree_Node this_node,int n_halo_max,list merger_tree):
    cdef:
        np.ndarray arr_mhalo = np.zeros(n_halo_max)-1
        np.ndarray arr_nodid = np.zeros(n_halo_max,dtype='int_')-1
        np.ndarray arr_treeid= np.zeros(n_halo_max,dtype='int_')-1
        np.ndarray arr_time  = np.zeros(n_halo_max,dtype='int_')-1
        np.ndarray arr_1prog = np.zeros(n_halo_max,dtype='int_')-1
        np.ndarray arr_desc  = np.zeros(n_halo_max,dtype='int_')-1
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
    arr_mhalo = arr_mhalo[0:count]
    arr_nodid = arr_nodid[0:count]
    arr_treeid= arr_treeid[0:count]
    arr_time  = arr_time[0:count]
    arr_1prog = arr_1prog[0:count]
    arr_desc  = arr_desc[0:count]

    return count,arr_mhalo,arr_nodid,arr_treeid,arr_time,arr_1prog,arr_desc

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

cdef class Make_Siblings:
    # class of the build up of the siblings from a certain Tree Node
    cdef Tree_Node this_node, child_node, parent_node, Sibling, sibs_left
    cdef list sib_nodes, sib_sorted
    cdef int n_frag_tot, child_index, i
    cdef public list merger_tree
    def next_sibling(self,child_node, parent_node):
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

    def associated_siblings(self,this_node,merger_tree):
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

    def build_sibling(self,merger_tree,n_frag_tot):
        for i in range(n_frag_tot):
            merger_tree = Make_Siblings().associated_siblings(merger_tree[i],merger_tree)
        return merger_tree

cdef class functions:
    # class to define the different functions used in the making of the tree
    cdef int n_table, n_v, n_sum
    cdef float a, a_min, delta_c
    cdef np.ndarray data
    def __init__(self,str filename):
        self.data = np.loadtxt(filename)
    def delta_crit(self,a,l_0=0.75,omega_0=0.25):
        omega_flat= self.data[:,0]
        delflat   = self.data[:,-1] 
        n_table = 200
        n_v = 1000
        a_min = 0.1
        eps_om = 1e-5
        n_sum = 2000
        delta_flat = [0]*n_table
        a_flat     = []
        omega_0_save = 0
        l_0_save = 0
        if fabs(1-omega_0) <= eps_om:
            del_c0 = 3*(12*pi)**(2/3)/20
            delta_c = del_c0/a
        elif (1-omega_0) > eps_om and l_0 < eps_om:
            eta_0 = np.acosh(2/omega_0-1)
            sh_0  = np.sinh(eta_0)
            ch_0  = np.cosh(eta_0)
            t_omega = 2*pi/(sh_0-eta_0)
            d_0   = 3*sh_0*(sh_0-eta_0)/(ch_0-1)**2-2

            ch = a*(ch_0-1)+1
            eta= np.acosh(ch)
            sh = np.sinh(eta)
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
                    a_flat.append(aa)
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

cdef double* Pk_0 = <double*>malloc(n_Pk_k*sizeof(double*))
cdef double* k_0 = <double*>malloc(n_Pk_k*sizeof(double*))
cdef int i_k
for i_k in range(n_Pk_k):
    k_0[i_k] = k_0_np[i_k]
    Pk_0[i_k] = Pk_0_np[i_k]
# Precompute sigma values for interpolation
cdef double* m_rough = c_logspace(7.0,15.0,200) #np.geomspace(1e7, 1e15, 200)
#cdef np.ndarray Sig = np.zeros_like(m_rough)

# Helper function for integrand calculation
cdef double my_int(double R, double k, double Pk) nogil:
    return 9 * (k * R * cos(k * R) - sin(k * R))**2 * Pk / k**4 / R**6 / (2 * pi**2)

# Sigma function
cdef double sigma(double m):
    cdef double R = (3 * m / (4 * pi * rho_crit))**(1/3)
    cdef double* my_integrand = <double*>malloc(n_Pk_k*sizeof(double*))
    cdef int i
    cdef cspline* sigma_spline
    
    for i in range(n_Pk_k):
        my_integrand[i] = my_int(R, k_0[i], Pk_0[i])
    sigma_spline = cspline_alloc(n_Pk_k,k_0,Pk_0)
    sigma_val = cspline_int(sigma_spline,m)
    cspline_free(sigma_spline)
    return sigma_val
#cdef int i

#for i in range(m_rough.shape[0]):
#    Sig[i] = sigma(m_rough[i])

# Interpolated sigma function
def sigma_cdm(double m):
    return sigma(m)


cdef double m_min=1e8
cdef double m_max=1e15
cdef int num_points=200
cdef np.ndarray m_array = np.geomspace(m_min, m_max, num_points, dtype=DTYPE)
cdef log_sig = compute_log_sig(m_array)
log_m = np.log(m_array)
interp = UnivariateSpline(log_m, log_sig, k=4, s=0.1)
deriv = interp.derivative(n=1)

cdef compute_log_sig(np.ndarray m_array):
    cdef int i
    cdef int n = m_array.shape[0]
    cdef double log_sig[200]
    
    for i in range(n):
        log_sig[i] = log(sigma_cdm(m_array[i]))
    return np.asarray(log_sig)

def alpha(double m):
    return float(deriv(log(m)))


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
cdef cspline* spline = cspline_alloc(n,log_z_arr,J_arr)

#log_J_used = CubicSpline(np.log(z_arr),J_arr,bc_type='natural') 

def J_unresolved(double z):
    #return J_pre(z)
    log_J_used = cspline_eval(spline,log(z))
    
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
        np.ndarray m_prog = np.zeros(2)
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
    return dw, n_prog, m_prog,m_min_last