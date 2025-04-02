from numpy import pi
import numpy as np
import scipy.integrate as itg
from astropy.constants import G,M_sun,pc
from classy import Class # type: ignore
from numba import njit
#from classy import DTYPE_t

# z = np.array([0],dtype='float64')#np.ndarray.tolist(np.linspace(0,4))
# N_k = 500
# k_array = np.zeros((N_k, 1, 1),dtype='float64')
# k_array[:,0,0] = np.logspace(-6,2,N_k)
# #print(k_array.ndim)
# #print(z)
# cosmo = Class()
# cosmo.set({'output':'mPk','P_k_max_h/Mpc':1})
# cosmo.compute()
#print(cosmo.get_pk(k_array,z,N_k,1,1))
#Pk_0 = cosmo.get_pk(k_array*0.73,z,N_k,1,1)[:,0,0]*0.73**3
#k_0 = cosmo.get_pk_and_k_and_z(nonlinear=False,only_clustering_species=True,h_units=True)[1]
# k_0 = k_array[:,0,0]
#print(len(Pk_0))
#print('Pk_0 = ',Pk_0)
#print('k_0 = ',k_0)

G_used = G.value*M_sun.value/(1e6*pc.value*(1e3**2))
#print('G_used = ',G_used)
H_0100 = 100*0.73
rho_crit = 3*H_0100**2/(8*pi*G_used)
#print(rho_crit)
m_8crit  = rho_crit*4*pi*8**3/3

# file_name = './Code_own/Data/pk_Mill.txt'
# pk_data   = np.loadtxt(file_name)

file_name = './Code_own/pk_CLASS.txt'
pk_data   = np.loadtxt(file_name)
k_0  = pk_data[0]
Pk_0 = pk_data[1]*0.73**3

@njit
def my_int(R,k,Pk):
    my_integrand = 9*(k*R*np.cos(k*R) - np.sin(k*R))**2 * Pk / k**4 / R**6 / 2 / np.pi**2
    return my_integrand

def sigma_cdm(m,k=k_0,Pk=Pk_0):
    #print('in sigma_cdm function')
    R = (3*m/(4*np.pi*rho_crit))**(1/3)
    # z_comp = 1/a - 1
    # z = np.array([z_comp],dtype='float64')
    #k  = pk_data[:,0] #k_array[:,0,0]
    #Pk = pk_data[:,1] #cosmo.get_pk(k_array,z,N_k,1,1)[:,0,0]
    #print(R)
    my_integrand = my_int(R,k,Pk)
    #print(my_integrand)
    sigma_cdm = np.sqrt(itg.simpson(my_integrand,k))
    #print(sigma_cdm)
    return sigma_cdm
'''
def sigma_cdm(m,a):
    z = 1/a-1
    R = (3*m/(4*np.pi*rho_crit))**(1/3)
    return cosmo.sigma(R,z,h_units=True)
'''
