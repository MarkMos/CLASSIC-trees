import numpy as np

class trees:
    def __init__(self):
        self.pk_method = 'class'
        self.default = {'output':'mPk','P_k_max_h/Mpc':1000}
        self.h_0 = 0.6781
        self.omega_0 = 0.30988304304812053
        self.l_0 = 1-0.30988304304812053
        self.cosmo_params = {'h':self.h_0,
                             'Omega_m':self.omega_0,
                             'Omega_Lambda':self.l_0}
        self.G_0=0.57
        self.gamma_1=0.38
        self.gamma_2=-0.01
        self.eps_1=0.1
        self.eps_2=0.1
        self.file_name_pk = None
    def set(self,pk_method='class',cosmo_params=None,random_mass='constant',
            add_cosmo_params=None,file=None,P_values='uncorrected'):
        if pk_method=='class':
            pk_method = self.pk_method
            file_name_pk = None
        elif pk_method=='default':
            file_name_pk = './CLASSIC-trees/pk_CLASS_default.txt'
            self.file_name_pk = file_name_pk
        elif pk_method=='self':
            file_name_pk = file
            self.file_name_pk = file
        else:
            raise KeyError('Choose one of the three options for pk_method:' \
                        '\n - class' \
                        '\n - default' \
                        '\n - self')
        if cosmo_params==None:
            cosmo_params = self.cosmo_params


        if file_name_pk=='./CLASSIC-trees/pk_CLASS_default.txt':
            h_0 = cosmo_params['h']
            omega_0 = cosmo_params['Omega_m']
            l_0 = cosmo_params['Omega_Lambda']
        elif file_name_pk==None:
            from classy import Class

            z = np.array([0],dtype='float64')
            N_k = 1000
            k_array = np.zeros((N_k, 1, 1),dtype='float64')
            k_array[:,0,0] = np.logspace(-6,3,N_k)
            cosmo = Class()
            cosmo.set(self.default)
            cosmo.set(cosmo_params)
            if add_cosmo_params!=None:
                cosmo.set(add_cosmo_params)
            cosmo.compute()
            Pk_0_np = cosmo.get_pk(k_array*cosmo_params['h'],z,N_k,1,1)[:,0,0]
            k_0_np = k_array[:,0,0]
        elif file_name_pk==file:
            h_0 = cosmo_params['h']
            omega_0 = cosmo_params['Omega_m']
            l_0 = cosmo_params['Omega_Lambda']
            self.h_0 = h_0
            self.omega_0 = omega_0
            self.l_0 = l_0

    def compute_fast(self,
                     file_name,
                     random_mass = None,
                     BoxSize = 479.0,
                     n_tree = 30,
                     i_seed_0 = -8635,
                     a_halo = 1,
                     m_res = 1e8,
                     z_max  = 4,
                     n_lev = 10,
                     n_halo_max = 1000000,
                     n_halo = 1,
                     n_part = 40000):
        omega_0 = self.omega_0
        l_0 = self.l_0
        h_0 = self.h_0
        from GenerateTreeFast import compute_tree_fast
        compute_tree_fast(random_mass,file_name,omega_0,l_0,h_0,BoxSize,n_tree,
                          i_seed_0,a_halo,m_res,z_max,n_lev,n_halo_max,n_halo,n_part)
    def compute_slow(self,
                     file_name = None,
                     random_mass = None,
                     BoxSize = 479.0,
                     n_lev = 10,
                     m_res = 1e8,
                     n_tree = 1,
                     n_halo = 1,
                     i_seed_0 = -8635,
                     a_halo = 1,
                     z_max = 4):
        omega_0 = self.omega_0
        l_0 = self.l_0
        h_0 = self.h_0
        from Generate_Tree import compute_tree
        compute_tree(random_mass,file_name,omega_0,l_0,h_0,BoxSize,n_lev,m_res,
                     n_tree,n_halo,i_seed_0,a_halo,z_max)