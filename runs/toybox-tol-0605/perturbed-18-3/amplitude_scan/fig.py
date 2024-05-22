import pickle
import matplotlib.pyplot as plt
plt.style.use('lateky')

# Load the data
result = pickle.load(open("results.pkl", "rb"))

a = []
r = []
for res in result:
    if len(res) == 0:
        continue
    for ra in res:
        a.append(ra[0])
        r.append(ra[1][ra[1] > 0].sum())

# Plot the data
plt.scatter(a, r)
plt.show()
