from lbm2d import LBM2D, Circle
import numpy as np
import matplotlib.pyplot as plt

Nx = 600
Ny = 200
L = 0.02
r = 20
Nsteps = 3000
nu = 1
tau = (6*nu + 1) / 2
umax = 0.01
Re = umax * L / nu 


def main():
    lbm = LBM2D(Nx, Ny, tau, True, True)
    lbm.add_shape(Circle(Nx // 4, Ny // 2, r))
    # hocu da tecem desno
    lbm.f *= umax
    E, errrho, erru = lbm.simulate(Nsteps, 50, 'openbound/', 10)
    t = range(Nsteps)
    plt.plot(t, E)
    plt.title("Kineticka energija")
    plt.savefig('openbound/final.png')
    plt.show()

if __name__=='__main__':
    main()
