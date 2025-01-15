from lbm2d import LBM2D, Circle, Rectangle
import numpy as np
import matplotlib.pyplot as plt

Nx = 600
Ny = 200
L = 0.02
r = 20
Nsteps = 1000
nu = 1
tau = (6*nu + 1) / 2
umax = 0.01
Re = umax * L / nu 


def main():
    lbm = LBM2D(Nx, Ny, tau, True, True)
    # E
    lbm.geometry[50:150, 30:60] = True
    lbm.geometry[50:70, 30:90] = True
    lbm.geometry[90:110, 30:90] = True
    lbm.geometry[130:150, 30:90] = True
    # T
    lbm.geometry[50:70, 100:200] = True
    lbm.geometry[50:150, 135:165] = True
    
	# F
    lbm.geometry[50:150, 230:260] = True
    lbm.geometry[50:70, 230:290] = True
    lbm.geometry[90:110, 230:290] = True
    
    #lbm.add_shape(Rectangle(50, 30, 100, 20))
    # hocu da tecem desno
    lbm.f *= umax
    E, errrho, erru = lbm.simulate(Nsteps, 50, 'etf/', 10)
    t = range(Nsteps)
    plt.plot(t, E)
    plt.title("Kineticka energija")
    plt.savefig('etf/final.png')
    plt.show()

if __name__=='__main__':
    main()
