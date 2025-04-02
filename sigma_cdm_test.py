import numpy as np

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

def sigma_cdm(m,a):
    scla = 0.9/(spl_intp(3.61766217E+12)[0])
    sclm = 0.1825**3/0.25
    return spl_intp(sclm*m)[0]*scla

def alpha(m,a):
    sclm = 0.1825**3/0.25
    return spl_intp(sclm*m)[1]