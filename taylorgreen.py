from lbm2d import LBM2D
import numpy as np
import matplotlib.pyplot as plt

Nx = 64
Ny = 64
Nsteps = 800
tau = 1
nu = (2 * tau - 1) / 6
rho0 = 1
umax = 0.02

def main():
    lbm = LBM2D(64, 64, 1, False)
    [tg_rho, tg_u, tg_P] = lbm.taylorgreen(0, nu, rho0, umax)
    lbm.f = LBM2D.equilibrium(tg_rho, tg_u)
    lbm.feq = np.zeros(lbm.f.shape)
    lbm.analytic = lambda t: lbm.taylorgreen(t, nu, rho0, umax)
    E, errrho, erru = lbm.simulate(Nsteps, 50, 'taylorgreen/', 10)
    t = range(0, Nsteps)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(t, E)
    axes[0].set_title('Energija s vremenom')
    axes[1].plot(t, errrho)
    axes[1].set_title('Greska u Rho')
    axes[2].plot(t, erru)
    axes[2].set_title('Greska u U')
    plt.savefig('taylorgreen/final.png')
    plt.show()
    

if __name__=='__main__':
    main()