from lbm2d import LBM2D, Circle
import numpy as np
import matplotlib.pyplot as plt

Nx = 600
Ny = 300
L = 0.02
r = 20
Nsteps = 200
nu = 0.01
tau = (6*nu + 1) / 2
umax = 0.1
Re = umax * L / nu 
g = 1.28e-6 # 1.1e-3

def main():
    lbm = LBM2D(Nx, Ny, tau, True, True)
    # hocu da tecem desno
    
    lbm.f = lbm.feq + 1
    lbm.f *= umax
    c = np.arange(Nx)
    v = (Nx - c) * L / Nx * g
    lbm.F[:, :] = v 
    E, errrho, erru = lbm.simulate(Nsteps, 20, 'gravity/', 10)
    t = range(Nsteps)
    plt.plot(t, E)
    plt.title("Kineticka energija")
    plt.savefig('gravity/final.png')
    plt.show()

if __name__=='__main__':
    main()
