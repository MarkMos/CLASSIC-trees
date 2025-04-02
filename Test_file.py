from sigma_cdm_func import *
from classy import Class
import numpy as np
import scipy.interpolate as intp
import matplotlib.pyplot as plt
from alpha_func import *
from functions import *

a_here = 1
mass = 1e10 #4/3*np.pi*rho_crit*8**3
#print('m_8 = ',mass)
print(sigma_cdm(mass))

file_3 = './Code_own/Data/sigmacdm_1.00_+0.000_1.00.txt'
data_sp= np.loadtxt(file_3,skiprows=1)
# print(data_sp)
def spl_intp(ms):
    kplo = 200
    kphi = 1
    i_mod= 0
    nmod = 2
    m_arr= data_sp[:,0]
    s_arr= data_sp[:,1]
    s_2  = data_sp[:,2]
    a_arr= data_sp[:,3]
    a_2  = data_sp[:,4]
    i_mod= 1+i_mod%nmod
    klo = kplo-1
    khi = kphi-1
    if m_arr[khi] < ms or m_arr[klo] > ms:
        klo = 0
        khi = kplo-1
        while khi-klo > 1:
            k = int((khi+klo)/2)
            if m_arr[k] > ms:
                khi=k
            else:
                klo=k
        h   = m_arr[khi]-m_arr[klo]
        h2  = (h**2)*0.1666667
        invh= 1/h
    aa = (m_arr[khi]-ms)*invh
    bb = (ms-m_arr[klo])*invh
    a3 = (aa**3-aa)
    b3 = (bb**3-bb)
    sigma = aa*s_arr[klo]+bb*s_arr[khi]+(a3*s_2[klo]+b3*s_2[khi])*h2
    alpha = aa*a_arr[klo]+bb*a_arr[khi]+(a3*a_2[klo]+b3*a_2[khi])*h2
    return sigma,alpha

print('sigma = ',spl_intp(3.61766217E+12)[0])

scla  = 0.9/(spl_intp(3.61766217E+12)[0])
print('scla = ',scla)
sigma = spl_intp(mass*0.1825**3/0.25)[0]
print('sigma_test = ',sigma*scla)
print('alpha = ',spl_intp(mass*0.1825**3/0.25)[1])

# def sig(m,a):
#     z = 1/a-1
#     R = (3*m/(4*np.pi*rho_crit))**(1/3)
#     return cosmo.sigma(R,z,h_units=True)
# print(sig(mass))
print('--------------------------')
print('alpha = ',alpha(mass))

# lsig = lambda M: np.log(sigma_cdm(M))

# m_array = np.linspace(1e8,1e15,1000)

# # logSig = []
# # for i in range(len(m_array)):
# #     logSig.append(lsig(m_array[i]))

# # logM = np.log(m_array)

# # interpolation = intp.UnivariateSpline(logM,logSig)
# # alpha = interpolation.derivative(n=1)

# # print('here: ',np.exp(interpolation(np.log(mass))))
# # print('alpha = ', alpha(np.log(mass/2)))

# Sig = []
# for i in range(len(m_array)):
#     Sig.append(sigma_cdm(m_array[i]))
# Sig = np.array(Sig)
# sig_inter = intp.interp1d(np.log(m_array),np.log(Sig))
# print('sigma = ',np.exp(sig_inter(np.log(mass))))

# Sig_inter = []
# for m in m_array:
#     Sig_inter.append(np.exp(sig_inter(np.log(m))))
# Sig_inter = np.array(Sig_inter)
# # plt.plot(m_array,Sig,label='original',marker='.',lw=0)
# # plt.plot(m_array,Sig_inter,label='interpolated',marker='.',lw=0)
# plt.plot(m_array,abs(Sig-Sig_inter)/Sig,label='Difference',marker='.',lw=0)
# plt.xlabel('m')
# plt.ylabel(r'$\Delta\sigma/\sigma$')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.savefig('Sigma_comp.png')


# file_name = './Code_own/Data/pk_Mill.txt'
# pk_data   = np.loadtxt(file_name)

# h = 0.73
# z = np.array([0],dtype='float64')
# N_k = 1000
# k_array = np.zeros((N_k, 1, 1),dtype='float64')
# k_array[:,0,0] = np.logspace(-6,3,N_k)

# cosmo = Class()
# cosmo.set({'output':'mPk','P_k_max_h/Mpc':10000})
# cosmo.compute()
# Pk_0 = cosmo.get_pk(k_array*h,z,N_k,1,1)[:,0,0]
# k_0 = k_array[:,0,0]
# np.savetxt('pk_CLASS.txt',[k_0,Pk_0])

# plt.plot(pk_data[:,0],pk_data[:,1],label='FORTRAN')
# plt.plot(k_0,Pk_0*h**3,label='CLASS')
# plt.xlabel('k')
# plt.ylabel('P_k')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.savefig('P_k_comparison.png')


# m_arr = np.linspace(1e8,1e15)
# sig_CLASS = []
# sig_FORT  = []
# for i in range(len(m_arr)):
#     sig_CLASS.append(sigma_cdm(m_arr[i]))
#     sig_FORT.append(scla*spl_intp(m_arr[i]*0.1825**3/0.25)[0])

# plt.plot(m_arr,sig_CLASS,label='CLASS')
# plt.plot(m_arr,sig_FORT,label='FORTRAN')
# plt.xlabel('Mass')
# plt.ylabel('Sigma')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.savefig('Sigma_Test.png')

# alp_CLASS = []
# alp_FORT  = []
# for i in range(len(m_arr)):
#     alp_CLASS.append(alpha(m_arr[i]))
#     alp_FORT.append(spl_intp(m_arr[i]*0.1825**3/0.25)[1])

# plt.plot(m_arr,alp_CLASS,label='CLASS')
# plt.plot(m_arr,alp_FORT,label='FORTRAN')
# plt.xlabel('Mass')
# plt.ylabel('Alpha')
# plt.xscale('log')
# # plt.yscale('log')
# plt.legend()
# plt.savefig('Alpha_Test.png')

# z = np.logspace(-3,3,1000)

# J_u_intp = []
# J_u_og = []
# for i in z:
#     J_u_intp.append(J_unresolved(i))
#     J_u_og.append(J_pre(i))
# J_u_og = np.array(J_u_og)
# J_u_intp = np.array(J_u_intp)

# fig, (ax1,ax2) = plt.subplots(2,sharex=True)
# ax1.plot(z,J_u_intp,label='interpolated')
# ax1.plot(z,J_u_og,label='previous')
# ax2.plot(z,abs(J_u_og-J_u_intp)/J_u_og,marker='.',lw=0)
# plt.xlabel('z')
# ax1.set_ylabel('J(u)')
# ax2.set_ylabel('D_J(u)')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.savefig('J_unresolved.png')

