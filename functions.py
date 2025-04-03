import numpy as np
import scipy.interpolate as intp
from numba import njit, prange


@njit(fastmath=True)
def J_pre(z,gamma_1=0.38):
    i_first = 0
    eps = 1e-5
    z_max = 10
    n_tab = 10000
    J_tab = np.zeros(n_tab) #[0]*n_tab
    z_tab = np.zeros(n_tab) #[0]*n_tab
    if abs(gamma_1) > eps:
        if i_first == 0:
            dz = z_max/n_tab
            inv_dz = 1/dz
            if abs(1-gamma_1) > eps:
                J_tab[0] = dz**(1-gamma_1)/(1-gamma_1)
            else:
                J_tab[0] = np.log(dz)
            z_tab[0] = dz
            for i in range(1,n_tab):
                z_tab[i] = (i+1)*dz
                J_tab[i] = J_tab[i-1]+(1+1/z_tab[i]**2)**(0.5*gamma_1)*0.5*dz+(1+1/z_tab[i-1]**2)**(0.5*gamma_1)*0.5*dz
            i_first = 1
        i = int(z*inv_dz) - 1
        if i < 1:
            if abs(1-gamma_1) > eps:
                J_un = (z**(1-gamma_1))/(1-gamma_1)
            else:
                J_un = np.log(z)
        elif i >= n_tab-1:
            J_un = J_tab[n_tab-1]+z-z_tab[n_tab-1]
        else:
            h = (z-z_tab[i])*inv_dz
            J_un = J_tab[i]*(1-h)+J_tab[i+1]*h
    else:
        J_un = z
    return J_un
z_arr = np.logspace(-3,3,400)
J_arr = np.zeros_like(z_arr)
# for i in range(len(z_arr)):
#     J_arr[i] = np.log(J_pre(z_arr[i]))

@njit(parallel=True, fastmath=True)
def compute_J_arr(z_arr, J_arr):
    for i in prange(len(z_arr)):
        J_arr[i] = np.log(J_pre(z_arr[i]))
compute_J_arr(z_arr, J_arr)

log_J_used = intp.CubicSpline(np.log(z_arr),J_arr,bc_type='natural')

def J_unresolved(z):
    return np.exp(log_J_used(np.log(z)))