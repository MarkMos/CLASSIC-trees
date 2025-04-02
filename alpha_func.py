from sigma_cdm_func import *
import numpy as np
import scipy.interpolate as intp

m_res=1e8
m_max=1e15

lsig = lambda M: np.log(sigma(M))

m_array = np.linspace(m_res,m_max,100)

logSig = np.zeros_like(m_array)
for i in range(len(m_array)):
    logSig[i] = lsig(m_array[i])

logM = np.log(m_array)

interpolation = intp.UnivariateSpline(logM,logSig,k=4)

def alpha(m):
    # lsig = lambda M: np.log(sigma_cdm(M,a))

    # m_array = np.linspace(m_res,m_max)

    # logSig = []
    # for i in range(len(m_array)):
    #     logSig.append(lsig(m_array[i]))

    # logM = np.log(m_array)

    # interpolation = intp.UnivariateSpline(logM,logSig,k=4)
    alpha_ret = interpolation.derivative(n=1)
    return alpha_ret(np.log(m))