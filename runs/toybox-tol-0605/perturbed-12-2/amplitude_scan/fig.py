import pickle
import matplotlib.pyplot as plt
plt.style.use('lateky')

# Load the data
result = pickle.load(open("results.pkl", "rb"))

a = []
r = []
err_by_diff = []
err_by_estim = []
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

# Plot the data
plt.errorbar(a, r, err_by_diff, fmt='s', markersize=1)
plt.show()