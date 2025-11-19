import numpy as np

class trees:
    #
    def __init__(self):
        self.pk_method = 'class'
        self.default = {'output':'mPk','P_k_max_h/Mpc':10000}
        self.h_0 = 0.6781
        self.omega_0 = 0.30988304304812053
        self.l_0 = 1-0.30988304304812053
        self.cosmo_params = {'h':self.h_0,
                             'Omega_m':self.omega_0,
                             'Omega_Lambda':self.l_0}
        self.file_name_pk = None
        self.k_0_np = None
        self.Pk_0_np = None
        self.m_max = 1e16
        self.m_min = 1e8
        self.verbose = 0
    def set(self,pk_method='class',cosmo_params=None,
            add_cosmo_params=None,file=None,P_values='uncorrected',verbose_level=0):
        if pk_method=='class':
            pk_method = self.pk_method
            file_name_pk = None
        elif pk_method=='default':
            file_name_pk = './PowerSpectra/pk_CLASS_default.txt'
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
        self.verbose = verbose_level


        if file_name_pk=='./PowerSpectra/pk_CLASS_default.txt':
            h_0 = cosmo_params['h']
            omega_0 = cosmo_params['Omega_m']
            l_0 = cosmo_params['Omega_Lambda']
            pk_data = np.loadtxt(file_name_pk)
            self.k_0_np = pk_data[0]
            self.Pk_0_np = pk_data[1]*h_0**3
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
            h_0 = cosmo_params['h']
            omega_0 = cosmo_params['Omega_m']
            l_0 = cosmo_params['Omega_Lambda']
            self.h_0 = h_0
            self.omega_0 = omega_0
            self.l_0 = l_0
            self.Pk_0_np = cosmo.get_pk(k_array*h_0,z,N_k,1,1)[:,0,0]*h_0**3
            self.k_0_np = k_array[:,0,0]
        elif file_name_pk==file:
            h_0 = cosmo_params['h']
            omega_0 = cosmo_params['Omega_m']
            l_0 = cosmo_params['Omega_Lambda']
            self.h_0 = h_0
            self.omega_0 = omega_0
            self.l_0 = l_0
            pk_data = np.loadtxt(file_name_pk)
            self.k_0_np = pk_data[0]
            if P_values=='corrected':
                self.Pk_0_np = pk_data[1]
            elif P_values=='uncorrected':
                self.Pk_0_np = pk_data[1]*h_0**3
        from classic_trees import set_trees
        set_trees(self)

    def compute_parallel(self,
                     file_name,
                     random_mass = None,
                     m_max = 1e16,
                     m_min = 1e11,
                     mass = None,
                     BoxSize = 479.0,
                     n_tree = 30,
                     i_seed_0 = -8635,
                     a_halo = 1,
                     m_res = 1e8,
                     z_max  = 4,
                     n_lev = 10,
                     n_halo_max = 1000000,
                     n_halo = 1,
                     n_part = 40000,
                     times = 'equal a',
                     mode='FoF',
                     pos_base = np.array([0,0,0],dtype=np.float64),
                     vel_base = np.array([10,10,10],dtype=np.float64),
                     scaling = 0.3):
        '''
        Function to call the routines of classic_trees for huge numbers of trees to 
        compute. Ideally to produce large merger tree files.
        ----------------------
        Input:
            random_mass : Distribution to draw the mass of the base node(s) of merger tree(s)
            mass        : Mass of the base node of a merger tree (only if random_mass=None)
            file_name   : Name of hdf5-file
            omega_0     : Relative density of matter in the universe
            l_0         : Relative cosmological constant
            h_0         : Reduced Hubble-parameter
            BoxSize     : Size of the volume
            n_tree      : Number of trees that are computed in one Pool
            i_seed_0    : Used for seed to generate random numbers
            a_halo      : Value of scale factor today (default) or up to which time the tree is calculated
            m_res       : Mass resolution limit; minimum mass
            m_min       : Minimum mass for the random drawing of masses
            m_max       : Maximum mass for the random drawing of masses
            z_max       : Maximum redshift for lookback
            n_lev       : Number of time levels
            n_halo_max  : Maximum number of nodes per tree; used for preallocation 
            n_halo      : Start of counter of nodes inside the tree(s)
            n_part      : Number of runs of a Pool
            times       : Either equally spaced times in z or a, or a custom array of z or a
            mode    	: Defining the usage of the merger tree.
            pos_base    : Initial 3 position of base node
            velo_base   : Initial 3 velocity of base node
            scaling     : Factor of scattering of positional change over time
        ----------------------
        Output:
            hdf5-file with values of random or constant mass merger trees
        '''
        if random_mass is not None:
            self.m_min = m_min
            self.m_max = m_max
        omega_0 = self.omega_0
        l_0 = self.l_0
        h_0 = self.h_0
        verbose = self.verbose
        from .GenerateTreeFast import compute_tree_parallel
        compute_tree_parallel(random_mass,mass,file_name,omega_0,l_0,h_0,BoxSize,n_tree,i_seed_0,
                          a_halo,m_res,m_min,m_max,z_max,n_lev,n_halo_max,n_halo,n_part,times,mode,pos_base,vel_base,scaling,verbose)
    def compute_inline(self,
                     mass = None,
                     n_halo_max = 1000000,
                     file_name = None,
                     random_mass = None,
                     m_max = 1e16,
                     m_min = 1e11,
                     BoxSize = 479.0,
                     n_lev = 10,
                     m_res = 1e8,
                     n_tree = 1,
                     n_halo = 1,
                     i_seed_0 = -8635,
                     a_halo = 1,
                     z_max = 4,
                     times = 'equal a',
                     mode ='FoF',
                     pos_base = np.array([0,0,0],dtype=np.float64),
                     vel_base = np.array([10,10,10],dtype=np.float64),
                     scaling = 0.3):
        '''
        Function to call the routines of classic_trees for small numbers of trees to 
        compute. Ideally to see what different starting values yield.
        ----------------------
        Input:
            mass        : Mass of the base node of a merger tree (only if random_mass=None)
            random_mass : Distribution to draw the mass of the base node(s) of merger tree(s)
            file_name   : Name of hdf5-file (optional)
            omega_0     : Relative density of matter in the universe
            l_0         : Relative cosmological constant
            h_0         : Reduced Hubble-parameter
            BoxSize     : Size of the volume
            n_lev       : Number of time levels
            m_res       : Mass resolution limit; minimum mass
            n_tree      : Number of trees that are computed
            n_halo      : Start of counter of nodes inside the tree(s)
            i_seed_0    : Used for seed to generate random numbers
            a_halo      : Value of scale factor today (default) or up to which time the tree is calculated
            z_max       : Maximum redshift for lookback
            times       : Either equally spaced times in z or a, or a custom array of z or a
            mode    	: Defining the usage of the merger tree.
            pos_base    : Initial 3 position of base node
            velo_base   : Initial 3 velocity of base node
            scaling     : Factor of scattering of positional change over time
        ----------------------
        Output:
            hdf5-file if file_name is not None
        '''
        if random_mass is not None:
            self.m_min = m_min
            self.m_max = m_max
        omega_0 = self.omega_0
        l_0 = self.l_0
        h_0 = self.h_0
        verbose = self.verbose
        from .Generate_Tree import compute_tree
        compute_tree(mass,n_halo_max,random_mass,file_name,omega_0,l_0,h_0,BoxSize,n_lev,m_res,
                     m_min,n_tree,n_halo,i_seed_0,a_halo,z_max,times,mode,pos_base,vel_base,scaling,verbose)