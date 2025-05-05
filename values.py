from classy import Class

# file_name_pk = './CLASSIC-trees/pk_CLASS_default.txt'
file_name_pk = './CLASSIC-trees/pk_CLASS_h_73.txt'
# file_name_pk = './CLASSIC-trees/pk_CLASS.txt'
# file_name_pk = None

if file_name_pk=='./CLASSIC-trees/pk_CLASS_default.txt':
    cosmo = Class()
    cosmo.set({'output':'mPk','P_k_max_h/Mpc':100})
    cosmo.compute()
    omega_0=cosmo.Omega_m()
    l_0=1-omega_0
    h_0=cosmo.h()
elif file_name_pk==None:
    cosmo = Class()
    cosmo.set({'output':'mPk','P_k_max_h/Mpc':10000})
    cosmo.compute()
    omega_0=cosmo.Omega_m()
    l_0=1-omega_0
    h_0=cosmo.h()
else:
    omega_0=0.25
    l_0=0.75
    h_0=0.73
G_0=0.57
gamma_1=0.38
gamma_2=-0.01
eps_1=0.1
eps_2=0.1
omega_b=0.04
Gamma=omega_0*h_0