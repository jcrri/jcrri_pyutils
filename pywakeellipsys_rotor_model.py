import numpy as np


def polar_grid_weigthed(n_r, n_theta):
    '''The weignts are assigned according to the area
    respective to each of the nodes. The area of a node is given
    by the product of the delta_r (radio segment) times the
    delta_theta (arc lenght). Thus, more weight is given to
    outter points of the rotor.'''
    delta_r = 1 / (n_r + 1)
    delta_theta = 2*np.pi / n_theta
    r, theta = np.meshgrid(delta_r*np.arange(1, n_r + 1),
                           np.arange(0, 2*np.pi, delta_theta))
    Aij = r * delta_r * delta_theta
    x = (r * np.cos(theta)).flatten()
    y = (r * np.sin(theta)).flatten()
    w = Aij.flatten() / sum(Aij.flatten())
    return x, y, w


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        from py_wake.rotor_avg_models import GridRotorAvg
        from py_wake import BastankhahGaussian
        from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site

        windTurbines = V80()
        site = Hornsrev1Site()
        plt.figure()
        x, y, w = polar_grid_weigthed(12, 12)
        m = GridRotorAvg(x, y, w)
        wfm = BastankhahGaussian(site, windTurbines, rotorAvgModel=m)
        ws_eff = wfm([0], [0], wd=270, ws=10).WS_eff_ilk[0, 0, 0]
        c = plt.scatter(m.nodes_x, m.nodes_y, c=m.nodes_weight,
                        label="$WS_{eff}$= %.2fm/s" % (ws_eff))
        plt.colorbar(c, label='weight')
        plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False))
        plt.axis('equal')
        plt.xlabel("y/R [m]")
        plt.ylabel('z/R [m]')
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.title('Polar grid rotor model')
        plt.legend()


main()
