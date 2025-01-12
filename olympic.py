from lbm2d import LBM2D, Circle
import numpy as np
import matplotlib.pyplot as plt

Nx = 600
Ny = 200
L = 0.02
r = 10
Nsteps = 3000
nu = 1
tau = (6*nu + 1) / 2
umax = 0.01
Re = umax * L / nu 


def main():
    lbm = LBM2D(Nx, Ny, tau, True, True)
    lbm.add_shape(Circle(200, 70, r))
    lbm.add_shape(Circle(230, 70, r))
    lbm.add_shape(Circle(260, 70, r))
    lbm.add_shape(Circle(215, 100, r))
    lbm.add_shape(Circle(245, 100, r))
    # hocu da tecem desno
    lbm.f *= umax
    E, errrho, erru = lbm.simulate(Nsteps, 50, 'olympic/', 10)
    t = range(Nsteps)
    plt.plot(t, E)
    plt.title("Kineticka energija")
    plt.savefig('olympic/final.png')
    plt.show()

if __name__=='__main__':
    main()
