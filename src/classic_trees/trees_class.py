import numpy as np

class forrest:
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
        self.masses = None
        self.jlevels = None
        self.redshifts = None
        self.factor = 1.0
    def set(self,pk_method='class',cosmo_params=None,spin_factor = 1,
            add_cosmo_params=None,file=None,P_values='uncorrected',verbose_level=0):
        if pk_method=='class':
            pk_method = self.pk_method
            file_name_pk = None
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
        self.factor = spin_factor

        if file_name_pk==None:
            from classy import Class

            z = np.array([0],dtype='float64')
            N_k = 1000
            k_array = np.zeros((N_k, 1, 1),dtype='float64')
            k_array[:,0,0] = np.logspace(-6,4,N_k)
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
            self.k_0_np = pk_data[:,0]
            if P_values=='corrected':
                self.Pk_0_np = pk_data[:,1]
            elif P_values=='uncorrected':
                self.Pk_0_np = pk_data[:,1]*h_0**3
        from .module import set_trees, set_sig_alph
        set_trees(self)
        set_sig_alph()

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
                     n_part = 40000,
                     times = 'equal a',
                     substructure='On'):
        '''
        Function to call the routines of classic_trees for huge numbers of trees to 
        compute. Ideally to produce large merger tree files.
        ----------------------
        Input:
            random_mass  : Distribution to draw the mass of the base node(s) of merger tree(s)
            mass         : Mass of the base node of a merger tree (only if random_mass=None)
            file_name    : Name of hdf5-file
            BoxSize      : Size of the volume
            n_tree       : Number of trees that are computed in one Pool
            i_seed_0     : Used for seed to generate random numbers
            a_halo       : Value of scale factor today (default) or up to which time the tree is calculated
            m_res        : Mass resolution limit; minimum mass
            m_max        : Maximum mass for the random drawing of masses
            m_min        : Minimum mass for the random drawing of masses
            z_max        : Maximum redshift for lookback
            n_lev        : Number of time levels
            n_halo_max   : Maximum number of nodes per tree; used for preallocation 
            n_part       : Number of runs of a Pool
            times        : Either equally spaced times in z or a, or a custom array of z or a
            substructure : Defining the usage of the merger tree with or without substructure.
        ----------------------
        Output:
            hdf5-file with values of random or constant mass merger trees
        '''
        n_halo = 1 # Start of counter of nodes inside the tree(s)
        scaling = 0.3
        if random_mass is not None:
            self.m_min = m_min
            self.m_max = m_max
        omega_0 = self.omega_0
        l_0 = self.l_0
        h_0 = self.h_0
        verbose = self.verbose
        if substructure=='On':
            mode = 'FoF'
        else:
            mode = 'Normal'
        from .parallel import compute_tree_parallel
        compute_tree_parallel(random_mass,mass,file_name,omega_0,l_0,h_0,BoxSize,n_tree,i_seed_0,
                          a_halo,m_res,m_min,m_max,z_max,n_lev,n_halo_max,n_halo,n_part,times,mode,scaling,verbose)
    def compute_serial(self,
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
                     substructure ='On'):
        '''
        Function to call the routines of classic_trees for small numbers of trees to 
        compute. Ideally to see what different starting values yield.
        ----------------------
        Input:
            mass         : Mass of the base node of a merger tree (only if random_mass=None)
            random_mass  : Distribution to draw the mass of the base node(s) of merger tree(s)
            file_name    : Name of hdf5-file (optional)
            m_max        : Maximum mass for the random drawing of masses
            m_min        : Minimum mass for the random drawing of masses
            BoxSize      : Size of the volume
            n_lev        : Number of time levels
            m_res        : Mass resolution limit; minimum mass
            n_tree       : Number of trees that are computed
            i_seed_0     : Used for seed to generate random numbers
            a_halo       : Value of scale factor today (default) or up to which time the tree is calculated
            z_max        : Maximum redshift for lookback
            times        : Either equally spaced times in z or a, or a custom array of z or a
            substructure : Defining the usage of the merger tree with or without substructure.
        ----------------------
        Output:
            hdf5-file if file_name is not None
        '''
        n_halo = 1 # Start of counter of nodes inside the tree(s)
        scaling = 0.3
        if random_mass is not None:
            self.m_min = m_min
            self.m_max = m_max
        omega_0 = self.omega_0
        l_0 = self.l_0
        h_0 = self.h_0
        verbose = self.verbose
        if substructure=='On':
            mode = 'FoF'
        else:
            mode = 'Normal'
        from .serial import compute_tree
        self.masses,self.jlevels,self.redshifts = compute_tree(mass,n_halo_max,random_mass,file_name,omega_0,l_0,h_0,BoxSize,n_lev,m_res,
                     m_min,m_max,n_tree,n_halo,i_seed_0,a_halo,z_max,times,mode,scaling,verbose)
        
    def hmf_at_z(self,z,n_bins=50,filename=None):
        '''
        Function to compute the halo mass function at certain z.
        IMPORTANT: Only use z that you gave as an input!
        ----------------------
        Input:
            z       : Redshift to extract information for the halo mass function
            n_bins  : Number of bins for the mass histogramm
            filename: Name of the tree-file
        ----------------------
        Output:
            hmf_bin     : Mass bins of the halos at redshift z
            hmf_bin_edge: Corresponding bin edges
        '''
        if filename==None:
            mass = self.masses
            time_level = self.jlevels
            z_arr = self.redshifts
            z_level = np.where(z_arr==z)[0][0]
            mass_z = mass[np.where(time_level==z_level)[0]]
            hmf_bin,hmf_bin_edge = np.histogram(mass_z,bins=np.geomspace(min(mass_z),max(mass_z),n_bins))
        else:
            import h5py
            f = h5py.File(filename)
            mass = f['TreeHalos/SubhaloMass'][:]
            time_level = f['TreeHalos/SnapNum'][:]
            z_arr = f['TreeTimes/Redshift'][:]
            z_level = np.where(z_arr==z)[0][0]
            mass_z = mass[np.where(time_level==z_level)[0]]
            hmf_bin,hmf_bin_edge = np.histogram(mass_z,bins=np.geomspace(min(mass_z),max(mass_z),n_bins))
        return hmf_bin,hmf_bin_edge