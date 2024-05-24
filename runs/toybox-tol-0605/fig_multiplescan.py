import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('lateky')

# Load the data
flist = ['perturbed-6-1/amplitude_scan_2/results.pkl',
         'perturbed-12-2/amplitude_scan/results.pkl', 
         'perturbed-18-3/amplitude_scan/results.pkl']

for file in flist:
    result = pickle.load(open(file, "rb"))

    a = []
    r = []
    err_by_diff = []
    err_by_estim = []
    isok = []
    for res in result:
        if len(res) == 0:
            continue
        for ra in res:
            a.append(ra[0])

            rtmp = ra[1][:,0]
            err_by_diff_tmp = ra[1][:,1]
            err_by_estim_tmp = ra[1][:,2]
            r.append(rtmp[rtmp > 0].sum())
            err_by_diff.append(err_by_diff_tmp[rtmp > 0].sum())
            err_by_estim.append(err_by_estim_tmp[rtmp > 0].sum())
            isok.append(np.isclose(rtmp.sum(), 0, atol=1e-2*np.abs(rtmp).max()))
    
    isok = np.array(isok, dtype=bool)
    a = np.array(a)
    r = np.array(r)
    err_by_diff = np.array(err_by_diff)
    # Plot the data
    plt.errorbar(a[isok], r[isok], err_by_diff[isok], fmt='s', markersize=3, label=file.split('/')[0])

plt.legend(loc='best')
plt.xlabel('Amplitude')
plt.ylabel('Total turnstile area')

plt.show()
