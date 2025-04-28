import numpy as np
from alpha_func import alpha
from sigma_cdm_func import sigma_cdm, rho_crit
from Delta_crit import delta_crit
# from classy import Class
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous
from scipy.integrate import simpson, cumulative_trapezoid
from scipy.interpolate import interp1d

n = 500
p = 0.3
q = 0.75
A_p = 0.3222
delta_c = delta_crit(1)

rho_bar = rho_crit
masses = np.geomspace(1e8,2e15,n)
alph_m = np.zeros(n)
sigm_m = np.zeros(n)

for i in range(n):
    alph_m[i] = -alpha(masses[i])
    sigm_m[i] = sigma_cdm(masses[i])

nu = delta_c**2/(sigm_m)**2
dln_nu_dln_m = 4*np.log(delta_c)*alph_m

nu_f_PS = np.sqrt(nu/(2*np.pi))*np.exp(-nu/2)
nu_f_ST = A_p*(1+(q*nu)**(-p))*np.sqrt((q*nu)/(2*np.pi))*np.exp(-(q*nu)/2)

n_PS = rho_bar/(masses**2)*nu_f_PS*dln_nu_dln_m
n_ST = rho_bar/(masses**2)*nu_f_ST*dln_nu_dln_m

summ_nPS = simpson(n_PS,masses)
summ_nST = simpson(n_ST,masses)

n_PS = n_PS/summ_nPS
n_ST = n_ST/summ_nST

# plt.plot(masses,n_PS,label='Press & Schechter')
# plt.plot(masses,n_ST,label='Sheth & Tormen')
# plt.grid()
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('m')
# plt.ylabel('n(m)')
# plt.legend()
# plt.savefig('HaloMassFunction.png')

# print(sum(n_PS))
# print(sum(n_ST))

class HaloMassFunction_ST(rv_continuous):
    def _pdf(self,m):
        alph_m = -alpha(m)
        sigm_m = sigma_cdm(m)
        nu = delta_c**2/(sigm_m)**2
        dln_nu_dln_m = 4*np.log(delta_c)*alph_m
        nu_f_ST = A_p*(1+(q*nu)**(-p))*np.sqrt((q*nu)/(2*np.pi))*np.exp(-(q*nu)/2)
        nST = rho_bar/(m**2)*nu_f_ST*dln_nu_dln_m
        return nST/summ_nST


hmf_ST = HaloMassFunction_ST(a=1e8,b=2e15,name='hmf_ST')

# print(hmf_ST.rvs(size=10))
# print(hmf_ST.pdf(masses))
temp_ST = np.zeros(n)

for i in range(n):
    temp_ST[i] = hmf_ST.pdf(masses[i])

cdf_ST = cumulative_trapezoid(temp_ST,masses,initial=0)
cdf_ST /= cdf_ST[-1]

ppf_ST = interp1d(cdf_ST,masses,kind='cubic',bounds_error=False,fill_value=(1e8,2e15))

# u_ST = np.random.rand(10000000)
# samples = ppf_ST(u_ST)


# def pdf_ST(m):
#     alph_m = -alpha(m)
#     sigm_m = sigma_cdm(m)
#     nu = delta_c**2/(sigm_m)**2
#     dln_nu_dln_m = 4*np.log(delta_c)*alph_m
#     nu_f_ST = A_p*(1+(q*nu)**(-p))*np.sqrt((q*nu)/(2*np.pi))*np.exp(-(q*nu)/2)
#     nST = rho_bar/(m**2)*nu_f_ST*dln_nu_dln_m
#     return nST/summ_nST

# m_min, m_max = 1e8,2e15
# y_max = pdf_ST(m_min)

# samples = []
# n_sample = 100
# while len(samples) < n_sample:
#     m_cand = np.random.uniform(m_min,m_max)
#     y_cand = np.random.uniform(0,y_max)

#     if y_cand <= pdf_ST(m_cand):
#         samples.append(m_cand)

# plt.hist(samples,bins=masses,density=True)
# plt.plot(masses,n_ST)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('m')
# plt.ylabel('Dist')
# plt.savefig('DistCheck.png')
# plt.show()

class HaloMassFunction_PS(rv_continuous):
    def _pdf(self,m):
        alph_m = -alpha(m)
        sigm_m = sigma_cdm(m)
        nu = delta_c**2/(sigm_m)**2
        dln_nu_dln_m = 4*np.log(delta_c)*alph_m
        nu_f_PS = A_p*(1+(q*nu)**(-p))*np.sqrt((q*nu)/(2*np.pi))*np.exp(-(q*nu)/2)
        nPS = rho_bar/(m**2)*nu_f_PS*dln_nu_dln_m
        return nPS/summ_nPS


hmf_PS = HaloMassFunction_PS(a=1e8,b=2e15,name='hmf_PS')

# print(hmf_ST.rvs(size=10))
# print(hmf_ST.pdf(masses))
temp_PS = np.zeros(n)

for i in range(n):
    temp_PS[i] = hmf_PS.pdf(masses[i])

cdf_PS = cumulative_trapezoid(temp_PS,masses,initial=0)
cdf_PS /= cdf_PS[-1]

ppf_PS = interp1d(cdf_PS,masses,kind='cubic',bounds_error=False,fill_value=(1e8,2e15))

# u_PS = np.random.rand(10000000)
# samples = ppf_PS(u_PS)
# plt.hist(samples,bins=masses,density=True)
# plt.plot(masses,n_PS)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('m')
# plt.ylabel('Dist')
# plt.savefig('DistCheckPS.png')
# plt.show()

# dn_dm_PS_fct = UnivariateSpline(masses[::2],n_PS[::2]).derivative(n=1)
# dn_dm_ST_fct = UnivariateSpline(masses[::2],n_ST[::2]).derivative(n=1)

# dn_dm_PS = np.zeros(n)
# dn_dm_ST = np.zeros(n)

# for i in range(n):
#     dn_dm_PS[i] = dn_dm_PS_fct(masses[i])
#     dn_dm_ST[i] = dn_dm_ST_fct(masses[i])

# plt.plot(masses,dn_dm_PS,label='Press & Schechter')
# plt.plot(masses,dn_dm_ST,label='Sheth & Tormen')
# plt.grid()
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('m')
# plt.ylabel('dn/dm')
# plt.legend()
# plt.savefig('HaloMassFunctionDeriv.png')