import numpy as np
import scipy.integrate as itg
from sigma_cdm_func import rho_crit
import matplotlib.pyplot as plt
from values import h_0, omega_0, file_name_pk

# if file_name_pk==None:
#     from values import k_0_np,Pk_0_np
#     k_0  = k_0_np
#     Pk_0 = Pk_0_np
# else:
#     file_name = file_name_pk
#     pk_data   = np.loadtxt(file_name)
#     k_0  = pk_data[0]
#     Pk_0 = pk_data[1]*h_0**3

# def sig_j_func(m,j,k=k_0,Pk=Pk_0):
#     R = (3*m/(4*np.pi*rho_crit))**(1/3)
#     my_integrand = k**(2+2*j)*9*(k*R*np.cos(k*R) - np.sin(k*R))**2 * Pk / k**6 / R**6 / 2 / np.pi**2
#     sig_j = np.sqrt(itg.simpson(my_integrand,k))
#     return sig_j

# def sig_halo_func(m,omega_0=omega_0,H0=100*h_0):
#     return H0*omega_0**(0.6)*sig_j_func(m,-1)*np.sqrt(1-sig_j_func(m,0)**4/(sig_j_func(m,1)**2*sig_j_func(m,-1)**2))
# n = 100
# masses   = np.logspace(11,15,n)
# sig_halo = np.zeros(n)

# for i in range(n):
#     sig_halo[i] = sig_halo_func(masses[i])

# plt.plot(masses,sig_halo)
# plt.grid()
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('m')
# plt.ylabel(r'$\sigma$')
# plt.savefig('sigma_halo.png')


def halo_pos(BoxSize,halo_type,cen_pos=np.array([0,0,0]),mean_linking_length=1,b=0.2):
    if halo_type=='cen':
        return BoxSize*np.random.random(size=3)
    else:
        pos = mean_linking_length*b*np.random.uniform(-1,1,3)
        while np.sqrt(sum(pos**2))>mean_linking_length*b:
            pos = mean_linking_length*b*np.random.uniform(-1,1,3)
        return cen_pos+pos