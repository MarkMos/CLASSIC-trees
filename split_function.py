from numpy import log,sqrt
import random
import numpy as np
import scipy.integrate as itg
# from sigma_cdm_func import sigma_cdm
# from alpha_func import alpha
from classic_trees import sigma_cdm
from classic_trees import SigmaInterpolator
from functions import *

SQRT2OPI=0.7978845608

def split(m_2, w,m_min,dw_max,eps_1,eps_2,m_min_last):
    #print('dw_max = ',dw_max)
    #print('in split function')
    m_prog =[0,0]
    eps_eta = 1e-6
    eps_q   = 6e-6
    gamma_1 = 0.38
    gamma_2 = -0.01
    G_0 = 0.57

    if m_min != m_min_last:
        sig_q_min = sigma_cdm(m_min)
        sigsq_q_min = sig_q_min**2
        m_min_last = m_min
        #print(1)
    else:
        sig_q_min = sigma_cdm(m_min)
        sigsq_q_min = sig_q_min**2
    #print('m_2 = ',m_2)
    sigma_m2 = sigma_cdm(m_2)
    #print('sigma_m2 = ',sigma_m2)
    # print('sigma_q_min = ',sig_q_min)
    sigsq_m2 = sigma_m2**2
    sig_hf = sigma_cdm(0.5*m_2) # sigma(m_2/2) and slope alpha_hf
    sigsq_hf = sig_hf**2
    alpha_hf = -SigmaInterpolator().alpha(0.5*m_2) # log(sig_hf)/log(0.5*m_2)
    #print('alpha_hf = ',alpha_hf)
    #print('m_min = ',m_min,'\n m_2 = ',m_2)
    q_min = m_min/m_2 # Minimum mass ratio
    #q_res = 0.125

    #sigma_res = sigma_cdm(q_res*m_2)
    #print(sigma_res**2 -sigma_m2**2)
    #u_res = sigma_m2/sqrt(abs(sigma_res**2 -sigma_m2**2))
    #print('u_res = ',u_res)
    g_fac0 = G_0*((w/sigma_m2)**gamma_2)

    if q_min < (0.5-eps_q):
        if gamma_1 <= 0:
            mu = -log(sig_q_min/sig_hf)/log(2*q_min)
        else:
            mu = alpha_hf
        g_fac1 = g_fac0*((sig_hf/sigma_m2)**gamma_1)/(2**(mu*gamma_1))

        diff_q_min = sigsq_q_min - sigsq_m2
        diff12_q_min = sqrt(diff_q_min)
        diff32_q_min = diff_q_min*diff12_q_min

        v_q_min = sigsq_q_min/diff32_q_min

        diff_hf = sigsq_hf - sigsq_m2
        diff12_hf = sqrt(diff_hf)
        diff32_hf = diff_hf*diff12_hf

        v_hf = sigsq_hf/diff32_hf
        s_fac = sqrt(2)*diff12_hf

        beta = log(v_q_min/v_hf)/log(2*q_min)
        if beta <= 0:
            raise ValueError('split(): beta <= 0')
        
        two_pow_beta = 2**beta
        b = v_hf*two_pow_beta
        eta = beta - 1 - gamma_1*mu

        half_pow_eta = 2**(-eta)
        if abs(eta) > eps_eta:
            eta_inv = 1/eta
            q_min_exp = q_min**eta
            f_fac = half_pow_eta-q_min_exp
            dn_dw = SQRT2OPI*alpha_hf*b*eta_inv*f_fac*g_fac1
        else:
            dn_dw = -SQRT2OPI*alpha_hf*b*log(2*q_min)*g_fac1

        if dn_dw > 0:
            dw_eps2 = eps_2/dn_dw
        else:
            dw_eps2 = dw_max
        dw = min(eps_1*s_fac,dw_eps2,dw_max)
        if dw<0:
            print('dw < 0-----------------------!')
            print('s_fac = ',s_fac)
            print('dw_eps2 = ',dw_eps2)
            print('dw_max = ',dw_max)
        n_av = dn_dw*dw

        #x = np.linspace(1.001*sigma_m2,100*sigma_m2)
        z = sigma_m2/diff12_q_min
        # print(z)
        f = SQRT2OPI*dw*g_fac0*J_unresolved(z)/sigma_m2
        #print('J(u_res) = ',J_unresolved(z))
        rand = random.random() #random1[r1_idx] #ran3(i_seed) #random.random()
        # r1_idx += 1
        # print('random1 = ',rand,'\n i_seed = ',i_seed)
        #print('n_av = ',n_av)
        if rand <= n_av:
            rand = random.random() #random2[r2_idx] #ran3(i_seed) #random.random()
            # r2_idx +=1
            #print('random2 = ',rand,'\n i_seed = ',i_seed)
            if abs(eta) > eps_eta:
                q_pow_eta = q_min_exp + rand*f_fac
                q = q_pow_eta**eta_inv
            else:
                q = q_min*(2*q_min)**(-rand)
            sig_q = sigma_cdm(q*m_2)
            sigsq_q = sig_q**2
            alpha_q = -SigmaInterpolator().alpha(q*m_2) #-log(sig_q)/log(q*m_2)
            diff12_q = sqrt(sigsq_q - sigsq_m2)
            v_q = sigsq_q/diff12_q**3

            R_q = (alpha_q/alpha_hf)*((sig_q*(2*q)**mu/sig_hf)**gamma_1)*v_q/(b*q**beta)

            if R_q > 1.00001:
                raise ValueError('split(): R_q > 1')
            rand = random.random() #random3[r3_idx] #ran3(i_seed) #random.random()
            # r3_idx +=1
            #print('random3 = ',rand,'\n i_seed = ',i_seed)
            if rand >= R_q:
                q = 0
        else:
            q = 0
        #print('q = ',q)
        m_prog[1] = q*m_2
        if m_prog[1] <= m_min:
            n_prog = 1
        else:
            n_prog = 2
        #print('f = ',f)
        m_prog[0] =(1-q-f)*m_2
        
        #print('here m = ',m_prog[0])
        if m_prog[0] <= m_min:
            n_prog -= 1
            m_prog[0] = m_prog[1]
            m_prog[1] = 0
        elif m_prog[0] < m_prog[1]:
            m_temp = m_prog[0]
            m_prog[0] = m_prog[1]
            m_prog[1] = m_temp
        
        #print('and herer m = ',m_prog[0])
    else:
        diff_hf = sigsq_hf - sigsq_m2
        diff12_hf = sqrt(diff_hf)
        s_fac = sqrt(2)*diff12_hf
        dw = min(eps_1*s_fac,dw_max)
        if dw<0:
            print('dw < 0-----------------------!')
            print('s_fac = ',s_fac)
            print('dw_eps2 = ',dw_eps2)
            print('dw_max = ',dw_max)
        diff_q_min = sigsq_q_min - sigsq_m2
        if diff_q_min < 0:
            diff_q_min = 0
        diff12_q_min = sqrt(diff_q_min)
        if diff12_q_min > SQRT2OPI*dw:
            z = sigma_m2/diff12_q_min
            # print(z)
            f = SQRT2OPI*dw*g_fac0*J_unresolved(z)/sigma_m2
            #print('J(u_res) = ',J_unresolved(z))
        else:
            f = 1
        m_prog[0] = (1-f)*m_2
        if m_prog[0] > m_min:
            n_prog = 1
        else:
            n_prog = 0
            m_prog[0] = 0
        m_prog[1] = 0
    #print('dw = ',dw)
    #print('n_prog = ',n_prog,'m_prog = ',m_prog)
    return dw, n_prog, m_prog,m_min_last