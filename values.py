import numpy as np
from classy import Class

'''
File to set cosmological parameters initially and add different cases.

For adding a file:
    - filestyle needs to be data[0] = k-values and data[1] = values of the 
    Powerspectrum (not in h-units!!!)
    - choose the right omega_0, l_0 and h_0 (see if-statements below)
'''

# Choose here if the mass input should be generated using 'PS' = Press&Schechter,
# 'ST' = Sheth&Tormen or None = constant
# random_mass = 'PS'
# random_mass = 'ST'
random_mass = None

# Choose here from two precomputed files or None and adjust yourself the cosmology
# you want in the elif-statement in the following.

pk_method = 'class'
# pk_method = 'file'

if pk_method=='class':
    file_name_pk = None
elif pk_method=='file':
    file_name_pk = './CLASSIC-trees/pk_CLASS_default.txt'
    # file_name_pk = './CLASSIC-trees/pk_CLASS_h_73.txt'

if file_name_pk=='./CLASSIC-trees/pk_CLASS_default.txt':
    cosmo = Class()
    cosmo.set({'output':'mPk','P_k_max_h/Mpc':100})
    cosmo.compute()
    omega_0=cosmo.Omega_m()
    l_0=1-omega_0
    h_0=cosmo.h()
elif file_name_pk==None:
    h_0 = 0.73
    # Choose omega_0 and l_0 such that it adds up to exact 1.0!
    omega_0 = 0.25
    l_0 = 0.75
    z = np.array([0],dtype='float64')
    N_k = 1000
    k_array = np.zeros((N_k, 1, 1),dtype='float64')
    k_array[:,0,0] = np.logspace(-6,3,N_k)

    cosmo = Class()
    cosmo.set({'output':'mPk','P_k_max_h/Mpc':1000,'h':h_0,'Omega_m':omega_0,'Omega_Lambda':l_0})
    cosmo.compute()
    Pk_0_np = cosmo.get_pk(k_array*h_0,z,N_k,1,1)[:,0,0]
    k_0_np = k_array[:,0,0]
else:
    omega_0=0.25
    l_0=0.75
    h_0=0.73

# Some constants that are used for the calculations.
G_0=0.57
gamma_1=0.38
gamma_2=-0.01
eps_1=0.1
eps_2=0.1
omega_b=0.04
Gamma=omega_0*h_0