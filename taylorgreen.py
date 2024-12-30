from lbm2d import LBM2D


Nx = 64
Ny = 64
Nsteps = 800
tau = 1
nu = (2 * tau - 1) / 6
rho0 = 1
umax = 0.02

def main():
    
    lbm = LBM2D(64, 64, 1)

if __name__=='__main__':
    main()