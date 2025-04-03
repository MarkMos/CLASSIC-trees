from sigma_cdm_func import *
import numpy as np
import scipy.interpolate as intp
from numba import njit, prange

m_res=1e8
m_max=1e15

@njit(parallel=True, fastmath=True)
def compute_log_sig(m_array):
    log_sig = np.zeros_like(m_array)
    for i in prange(len(m_array)):
        log_sig[i] = np.log(sigma_cdm(m_array[i]))
    return log_sig


# lsig = lambda M: np.log(sigma(M))

m_array = np.geomspace(1e8,1e15,200)

logSig =  compute_log_sig(m_array) #np.zeros_like(m_array)
# for i in range(len(m_array)):
#     logSig[i] = lsig(m_array[i])

logM = np.log(m_array)

interpolation = intp.UnivariateSpline(logM,logSig,k=4,s=.1)
alpha_ret = interpolation.derivative(n=1)

def alpha(m):
    return alpha_ret(np.log(m))