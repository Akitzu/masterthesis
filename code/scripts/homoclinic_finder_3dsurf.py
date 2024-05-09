

n_s, n_u = 5, 6
def evolution(self, eps):
    eps_s, eps_u = eps
    r_s = self.rfp_s + eps_s * self.vector_s
    r_u = self.rfp_u + eps_u * self.vector_u

    try:
        r_s_evolved = self.integrate_single(r_s, n_s, -1, ret_jacobian = False)
    except Exception as e:
        print(f"Error in stable manifold integration : {e}")

    try:
        r_u_evolved = self.integrate_single(r_u, n_u, 1, ret_jacobian = False)
    except Exception as e:
        print(f"Error in unstable manifold integration : {e}")

    return np.linalg.norm(r_s_evolved - r_u_evolved)

eps_s = np.logspace(np.log(bounds_1[0][0]) / np.log(manifold.lambda_s), 
            np.log(bounds_1[0][1]) / np.log(manifold.lambda_s),
            10, base=manifold.lambda_s)

eps_u = np.logspace(np.log(bounds_1[1][0]) / np.log(manifold.lambda_u), 
            np.log(bounds_1[1][1]) / np.log(manifold.lambda_u),
            10, base=manifold.lambda_u)

es, eu = np.meshgrid(eps_s, eps_u)

E = np.array([evolution(manifold, [S, U]) for S, U in zip(es.flatten(), eu.flatten())]).reshape(es.shape)

from mpl_toolkits.mplot3d import Axes3D

fig_n, ax_n = plt.subplots(subplot_kw={"projection": "3d"})

ax_n.plot_surface(es, eu, E)
ax_n.plot_surface(es, eu, np.zeros_like(E))

plt.show()