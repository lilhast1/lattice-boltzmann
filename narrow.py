from lbm2d import LBM2D, Circle, Rectangle
import numpy as np
import matplotlib.pyplot as plt

Nx = 400
Ny = 100
L = 0.02
r = 15
Nsteps = 3000
nu = 1/50
tau = (6*nu + 1) / 2
umax = 0.01
Re = umax * L / nu 


def main():
    lbm = LBM2D(Nx, Ny, tau, True)
    lbm.add_shape(Rectangle(100, 0, 10, 30))
    lbm.add_shape(Rectangle(100, 70, 10, 30))
    # hocu da tecem desno
    lbm.f *= umax
    E, errrho, erru = lbm.simulate(Nsteps, 50, 'narrow/', 10)
    t = range(Nsteps)
    plt.plot(t, E)
    plt.title("Kineticka energija")
    plt.savefig('narrow/final.png')
    plt.show()

if __name__=='__main__':
    main()
