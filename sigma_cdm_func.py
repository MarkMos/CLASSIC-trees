from numpy import pi
import numpy as np
import scipy.integrate as itg
from astropy.constants import G,M_sun,pc
from classy import Class # type: ignore
from numba import njit, float64, prange
import scipy.interpolate as intp
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

file_name = './CLASSIC-trees/pk_CLASS.txt'
pk_data   = np.loadtxt(file_name)
k_0  = pk_data[0]
Pk_0 = pk_data[1]*0.73**3

# @njit
# def my_int(R,k,Pk):
#     my_integrand = 9*(k*R*np.cos(k*R) - np.sin(k*R))**2 * Pk / k**4 / R**6 / 2 / np.pi**2
#     return my_integrand

# def sigma(m,k=k_0,Pk=Pk_0):
#     #print('in sigma_cdm function')
#     R = (3*m/(4*np.pi*rho_crit))**(1/3)
#     # z_comp = 1/a - 1
#     # z = np.array([z_comp],dtype='float64')
#     #k  = pk_data[:,0] #k_array[:,0,0]
#     #Pk = pk_data[:,1] #cosmo.get_pk(k_array,z,N_k,1,1)[:,0,0]
#     #print(R)
#     my_integrand = my_int(R,k,Pk)
#     #print(my_integrand)
#     sigma = np.sqrt(itg.simpson(my_integrand,k))
#     #print(sigma_cdm)
#     return sigma

@njit(float64(float64, float64, float64), cache=True)
def my_int(R, k, Pk):
    """Numba-optimized integrand calculation"""
    if k == 0 or R == 0:
        return 0.0
    kR = k * R
    trig_part = kR * np.cos(kR) - np.sin(kR)
    scale = 9.0 / (2 * np.pi**2 * R**6)
    return scale * (trig_part**2) * Pk / k**4

@njit(float64[:](float64, float64[:], float64[:]), parallel=True)
def compute_integrand(R, k, Pk):
    """Vectorized integrand computation"""
    result = np.empty_like(k)
    for i in prange(len(k)):
        result[i] = my_int(R, k[i], Pk[i])
    return result

def sigma(m, k=k_0, Pk=Pk_0):
    """Optimized σ(m) calculation"""
    R = (3 * m / (4 * np.pi * rho_crit))**(1/3)
    integrand = compute_integrand(R, k, Pk)
    return np.sqrt(itg.simpson(integrand, k))

m_rough = np.geomspace(1e7,1e15,200)
Sig = np.zeros_like(m_rough)
for i in range(len(m_rough)):
    Sig[i]=sigma(m_rough[i])
# Sig = np.array(Sig)
# sig_inter = intp.interp1d(np.log(m_rough),np.log(Sig))
@njit(fastmath=True)
def sigma_cdm(m):
    return np.exp(np.interp(np.log(m),np.log(m_rough),np.log(Sig)))
    # return np.exp(sig_inter(np.log(m)))
'''
def sigma_cdm(m,a):
    z = 1/a-1
    R = (3*m/(4*np.pi*rho_crit))**(1/3)
    return cosmo.sigma(R,z,h_units=True)
'''
