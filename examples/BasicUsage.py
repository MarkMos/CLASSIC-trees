'''
This script gives you a first overview on how to generate a merger-tree.
'''
import classic_trees as ct

# create an instance of the class 'forest'
merger_tree = ct.forest()

# set a cosmology and a Power-Spectrum
merger_tree.set(pk_method='file',                      # one of two methods; here for speed a pre-computed Power-Spectrum, other method is directly running class to get it
                file='./PowerSpectra/pk_class.txt',    # relative path to the Power-Spectrum file
                h_units = True,                        # multiplies the Power-Spectrum values by h^3 if needed
                cosmo_params = {'h':0.6781,            #
                             'Omega_m':0.309974,       # cosmological parameters as one would set in class
                             'Omega_Lambda':0.690026}, # also add_cosmo_params for more than the three listed here
                verbose_level=4,                       # level of verbosity, here set to 1 to see some basic output of the code
                spin_factor=1e-3)                      # Factor to adjust the placement of the spins absolute value         

# compute merger-tree with given parameters
merger_tree.compute_serial(mass = 1e13,                 # mass of the halo today in M_sun/h
                           n_halo_max = 1000000,        # number of maximum halos that can be within a merger tree
                           file_name = None,            # file_name; has to be set to .hdf5 which is the output-format in classic_trees
                           random_mass = None,          # random mass that can be drawn either from the Press-Schechter or Sheth-Torman approximation or neither for same mass trees; masses in M_sun/h
                           m_max = 1e16,                # maximum mass to draw masses from in M_sun/h
                           m_min = 1e11,                # minimum mass to draw masses from in M_sun/h
                           BoxSize = 479.0,             # Box-size of the simulation volume in (Mpc/h)^3
                           n_steps = 10,                # number of timesteps of the tree
                           m_res = 1e8,                 # mass resolution in M_sun/h
                           n_tree = 1,                  # number of trees to be computed
                           i_seed_0 = -8635,            # seed to generate always the same random trees
                           a_halo = 1,                  # scale factor today
                           z_max = 4,                   # redshift to which the tree is tracked
                           time_spacing = 'equal_a',    # spacing of the times between z = 0 and z = z_max; either equal in a or z, or a list of scale factors or redshifts that is custom
                           subhalos = True)             # mode of the algorithm; False means no substructure, True means substructure
