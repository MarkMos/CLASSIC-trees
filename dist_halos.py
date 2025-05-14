import h5py
import matplotlib.pyplot as plt
import numpy as np

filenames = ['groups_088.hdf5']#,'groups_090.hdf5','groups_038.hdf5','groups_032.hdf5','groups_058.hdf5','groups_062.hdf5','groups_042.hdf5','groups_028.hdf5','groups_048.hdf5','groups_084.hdf5']
# sub_masses = []
# nsubs = []
for str in filenames:
    filename = str
    sub_masses = []
    nsubs = []

    f = h5py.File(filename,'r')

    lengths = f['Group/GroupFirstSub'][:]
    sub_bin1 = []
    sub_bin2 = []
    sub_bin3 = []
    sub_bin4 = []
    sub_bin5 = []
    sub_bin6 = []
    sub_bin7 = []
    sub_bin8 = []
    sub_bin9 = []
    sub_bin10 = []
    for j,i in enumerate(lengths):
        sub_masses.append(f['Subhalo/SubhaloMass'][i]*1e10)
    c = 0
    count = 0
    for j in range(len(f['Group/GroupNsubs'][:])):
        m = f['Subhalo/SubhaloMass'][lengths[j]]*1e10
        nsubs.append(f['Group/GroupNsubs'][j])
        if 1e9<m<5e9:
            sub_bin1 += np.ndarray.tolist(f['Subhalo/SubhaloMass'][c+1:c+nsubs[j]]*1e10)
        elif 5e9<m<1e10:
            sub_bin2 += np.ndarray.tolist(f['Subhalo/SubhaloMass'][c+1:c+nsubs[j]]*1e10)
        elif 1e10<m<5e10:
            sub_bin3 += np.ndarray.tolist(f['Subhalo/SubhaloMass'][c+1:c+nsubs[j]]*1e10)
        elif 5e10<m<1e11:
            sub_bin4 += np.ndarray.tolist(f['Subhalo/SubhaloMass'][c+1:c+nsubs[j]]*1e10)
        elif 1e11<m<5e11:
            sub_bin5 += np.ndarray.tolist(f['Subhalo/SubhaloMass'][c+1:c+nsubs[j]]*1e10)
        elif 5e11<m<1e12:
            sub_bin6 += np.ndarray.tolist(f['Subhalo/SubhaloMass'][c+1:c+nsubs[j]]*1e10)
        elif 1e12<m<5e12:
            sub_bin7 += np.ndarray.tolist(f['Subhalo/SubhaloMass'][c+1:c+nsubs[j]]*1e10)
        elif 5e12<m<1e13:
            sub_bin8 += np.ndarray.tolist(f['Subhalo/SubhaloMass'][c+1:c+nsubs[j]]*1e10)
        elif 1e13<m<5e13:
            sub_bin9 += np.ndarray.tolist(f['Subhalo/SubhaloMass'][c+1:c+nsubs[j]]*1e10)
        else:
            sub_bin10 += np.ndarray.tolist(f['Subhalo/SubhaloMass'][c+1:c+nsubs[j]]*1e10)
        c += nsubs[j]
    # sub_bin1 = np.log(np.array(sub_bin1))
    # sub_bin2 = np.log(np.array(sub_bin2))
    # sub_bin3 = np.log(np.array(sub_bin3))
    # sub_bin4 = np.log(np.array(sub_bin4))
    # sub_bin5 = np.log(np.array(sub_bin5))
    # sub_bin6 = np.log(np.array(sub_bin6))
    # sub_bin7 = np.log(np.array(sub_bin7))
    # sub_bin8 = np.log(np.array(sub_bin8))
    # sub_bin9 = np.log(np.array(sub_bin9))
    # sub_bin10 = np.log(np.array(sub_bin10))
    f.close()
    bin1_m = []
    bin1_n = []
    bin2_m = []
    bin2_n = []
    bin3_m = []
    bin3_n = []
    bin4_m = []
    bin4_n = []
    bin5_m = []
    bin5_n = []
    bin6_m = []
    bin6_n = []
    bin7_m = []
    bin7_n = []
    bin8_m = []
    bin8_n = []
    bin9_m = []
    bin9_n = []
    bin10_m = []
    bin10_n = []
    for i,m in enumerate(sub_masses):
        if 1e9<m<5e9:
            bin1_m.append(m)
            bin1_n.append(nsubs[i])
        elif 5e9<m<1e10:
            bin2_m.append(m)
            bin2_n.append(nsubs[i])
        elif 1e10<m<5e10:
            bin3_m.append(m)
            bin3_n.append(nsubs[i])
        elif 5e10<m<1e11:
            bin4_m.append(m)
            bin4_n.append(nsubs[i])
        elif 1e11<m<5e11:
            bin5_m.append(m)
            bin5_n.append(nsubs[i])
        elif 5e11<m<1e12:
            bin6_m.append(m)
            bin6_n.append(nsubs[i])
        elif 1e12<m<5e12:
            bin7_m.append(m)
            bin7_n.append(nsubs[i])
        elif 5e12<m<1e13:
            bin8_m.append(m)
            bin8_n.append(nsubs[i])
        elif 1e13<m<5e13:
            bin9_m.append(m)
            bin9_n.append(nsubs[i])
        else:
            bin10_m.append(m)
            bin10_n.append(nsubs[i])

    # avg_mass = np.array([np.mean(bin1_m),np.mean(bin2_m),np.mean(bin3_m),np.mean(bin4_m),np.mean(bin5_m),np.mean(bin6_m),np.mean(bin7_m),np.mean(bin8_m),np.mean(bin9_m),np.mean(bin10_m)])
    # avg_nsub = np.array([np.mean(bin1_n),np.mean(bin2_n),np.mean(bin3_n),np.mean(bin4_n),np.mean(bin5_n),np.mean(bin6_n),np.mean(bin7_n),np.mean(bin8_n),np.mean(bin9_n),np.mean(bin10_n)])
    # std_mass = np.array([np.std(bin1_m),np.std(bin2_m),np.std(bin3_m),np.std(bin4_m),np.std(bin5_m),np.std(bin6_m),np.std(bin7_m),np.std(bin8_m),np.std(bin9_m),np.std(bin10_m)])
    # std_nsub = np.array([np.std(bin1_n),np.std(bin2_n),np.std(bin3_n),np.std(bin4_n),np.std(bin5_n),np.std(bin6_n),np.std(bin7_n),np.std(bin8_n),np.std(bin9_n),np.std(bin10_n)])
    # plt.hist(sub_bin1,density=True,histtype='step')
    # plt.hist(sub_bin2,density=True,histtype='step')
    # plt.hist(sub_bin3,density=True,histtype='step')
    # plt.hist(sub_bin4,density=True,histtype='step')
    # plt.hist(sub_bin5,density=True,histtype='step')
    # plt.hist(sub_bin6,density=True,histtype='step')
    # plt.hist(sub_bin7,density=True,histtype='step')
    # plt.hist(sub_bin8,density=True,histtype='step')
    # plt.hist(sub_bin9,density=True,histtype='step')
    # plt.hist(sub_bin10,density=True,histtype='step')
    # plt.errorbar(avg_mass,avg_nsub,std_nsub,std_mass,fmt='.')
    # plt.scatter(sub_masses,nsubs,color='gray',marker='.')
sub_bin_arr = [sub_bin1,sub_bin2,sub_bin3,sub_bin4,sub_bin5,sub_bin6,sub_bin7,sub_bin8,sub_bin9,sub_bin10]
for i in range(10):
    # name = 'SubhaloMasses'+str(i+1)+'.png'
    plt.hist(sub_bin_arr[i],density=True,histtype='step')    
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('m')
    plt.ylabel('Count')
    # plt.savefig(name)
    plt.show()