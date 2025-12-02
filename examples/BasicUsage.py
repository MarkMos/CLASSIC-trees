# The first thing to do is simply import classic_trees
import classic_trees as ct

# Then we use some basic lines to get our first merger tree

# First we need to get the class 'tree' in order to access the differenet asspects
merger_tree = ct.tree()

# Then we need to set our cosmology and get our Power-Spectrum
merger_tree.set(pk_method='self',                      # one of two methods; here for speed a pre-computed Power-Spectrum, other method is directly running class to get it
                file='./PowerSpectra/pk_class.txt',    # relative path two the Power-Spectrum file
                P_values = 'corrected',                # multiplies the Power-Spectrum values by h^3 if needed
                cosmo_params = {'h':0.6781,            #
                             'Omega_m':0.309974,       # cosmological parameters as one would set in class
                             'Omega_Lambda':0.690026}, # also add_cosmo_params for more then the three listed here
                verbose_level=1)                       # level of verbosity, here set to 1 to see some basic output of the code

# After setting the cosmology we can e.g. run the following line to get one merger tree
merger_tree.compute_serial(mass = 1e13,          # mass of the halo today
                           n_halo_max = 1000000, # number of maximum halos that can be within a merger tree
                           file_name = None,     # file_name; has to be set to .hdf5 which is the output-format in classic_trees
                           random_mass = None,   # random mass that can be drawn either from the Press-Schechter or Sheth-Torman approximation or neither for same mass trees
                           m_max = 1e16,         # maximum mass to draw masses from
                           m_min = 1e11,         # minimum mass to draw masses from
                           BoxSize = 479.0,      # Boxsize of the simulation volume
                           n_lev = 10,           # number of time-levels of the tree
                           m_res = 1e8,          # mass resolution
                           n_tree = 1,           # number of trees to be made
                           n_halo = 1,           # start of counter of nodes inside the tree
                           i_seed_0 = -8635,     # seed to generate always the same random trees
                           a_halo = 1,           # scale factor today
                           z_max = 4,            # redshift to which the tree is tracked
                           times = 'equal a',    # spacing of the times between z = 0 and z = z_max; either equal in a or z, or a list of scale factors or redshifts that is custom
                           mode ='Normal')       # mode of the algorithm; Normal means no substructure FoF means substructure