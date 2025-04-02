from numpy import pi,acosh,sinh,cosh,sqrt
from locate_function import *
import numpy as np
import bisect

# class Cosmo_parameters:
#     def __init__(self,l_0,omega_0):
#         self.l_0 = l_0
#         self.omega_0 = omega_0
filename = './Code_own/Data/flat.txt'
data = np.loadtxt(filename)
omega_flat= data[:,0]
delflat   = data[:,-1]

def delta_crit(a,l_0=0.75,omega_0=0.25):
    #print('in delta_crit function')
    n_table = 200
    n_v = 1000
    a_min = 0.1
    eps_om = 1e-5
    n_sum = 2000
    delta_flat = [0]*n_table
    # omega_flat = []
    # delflat = []
    a_flat     = []

    omega_0_save = 0
    l_0_save = 0

    if abs(1-omega_0) <= eps_om:
        del_c0 = 3*(12*pi)**(2/3)/20
        delta_c = del_c0/a
    elif (1-omega_0) > eps_om and l_0 < eps_om:
        eta_0 = acosh(2/omega_0-1)
        sh_0  = sinh(eta_0)
        ch_0  = cosh(eta_0)
        t_omega = 2*pi/(sh_0-eta_0)
        d_0   = 3*sh_0*(sh_0-eta_0)/(ch_0-1)**2-2

        ch = a*(ch_0-1)+1
        eta= acosh(ch)
        sh = sinh(eta)
        t  = (sh-eta)/(sh_0-eta_0)
        delta_c = 1.5*d_0*(1+(t_omega/t)**(2/3))

    else:
        if omega_0 != omega_0_save or l_0 != l_0_save:
            omega_0_save = omega_0
            l_0_save = l_0
            if abs(omega_0+l_0-1) > eps_om:
                raise ValueError('ERROR: omega_0+l_0+1 != 1')
            x_0 = (2*(1/omega_0-1))**0.333333
            #print('x_0 = ',x_0)
            summ= 0
            dxp = x_0/float(n_sum)
            for i_s in range(1,n_sum+1):
                x_p = x_0*(float(i_s)-0.5)/float(n_sum)
                if x_p < 0:
                    x_p = 0
                summ += ((x_p/(x_p**3+2))**1.5)*dxp
            dlin_0 = summ*sqrt(x_0**3+2)/sqrt(x_0**3)
            
            for i in range(1,n_table+1):
                aa = a_min+(1-a_min)*float(i-1)/float(n_table-1)
                a_flat.append(aa)
                l = l_0/(l_0+(1-omega_0-l_0)/aa**2+omega_0/aa**3)
                omega = omega_0*l/(l_0*aa**3)
                x = x_0*aa
                summ = 0
                dxp = x/float(n_sum)
                for i_s in range(1,n_sum+1):
                    x_p = x*(float(i_s)-0.5)/float(n_sum)
                    if x_p<0:
                        x_p = 0
                    summ += ((x_p/(x_p**3+2))**1.5)*dxp
                dlin = (summ*sqrt(x**3+2)/sqrt(x**3))/dlin_0
                i_o = bisect.bisect_left(omega_flat,omega) #locate(omega_flat,n_v-1,omega)
                if i_o < n_v:
                    h = (omega_flat[i_o]-omega)/(omega_flat[i_o]-omega_flat[i_o-1])
                    delta_flat[i-1]=(delflat[i_o-1]*h+delflat[i_o]*(1-h))/dlin
                else:
                    delta_flat[i-1]=delflat[n_v-1]/dlin
        if a > a_min and a <= 1.001:
            i = 1+int((a-a_min)*float(n_table-1)/(1-a_min))
            i = min(n_table-2,i)-1
            #print('i = ',i)
            h = (a_flat[i+1]-a)/(a_flat[i+1]-a_flat[i])
            delta_c = delta_flat[i]*h+delta_flat[i+1]*(1-h)
        elif a <= a_min:
            delta_c = delta_flat[0]*a_min/a
        else:
            raise ValueError('delta_crit(): FATAL - look up tale only for a<1 \n a=',a)
    #print('delta_flat = ',delta_flat)
    return delta_c