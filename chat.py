import numpy as np
from matplotlib import pyplot

class LBM2D:
    cx = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cy = np.array([0, 1, 1, 0, -1, -1, -1, -1, 0])
    w = np.array([4. / 9, 1. / 9, 1. / 36, 1. / 9, 1. / 36, 1. / 9, 1. / 36, 1. / 9, 1. / 36], dtype=np.float64)
    Nl = 9

    def __init__(self, Nx, Ny, tau, Nt):
        self.Nx = Nx
        self.Ny = Ny
        self.tau = tau
        self.Nt = Nt
        self.f = np.zeros((Ny, Nx, self.Nl))

        # Initialize the velocity and density fields
        self.rho = np.ones((Ny, Nx))  # Starting with constant density
        self.ux = np.zeros((Ny, Nx))  # Starting with zero velocity
        self.uy = np.zeros((Ny, Nx))  # Starting with zero velocity

        # Initialize the distribution function to equilibrium values
        for i, v, x, y in zip(range(self.Nl), self.w, self.cx, self.cy):
            cu = self.ux * x + self.uy * y
            usqr = self.ux**2 + self.uy**2
            self.f[:, :, i] = self.rho * v * (1 + 3 * cu + 9 * cu**2 / 2 - 3 * usqr / 2)

        self.geometry = np.full((Ny, Nx), False)
        self.feq = np.zeros(self.f.shape)

    def add_geometry(self, shape_map):
        self.geometry = self.geometry + shape_map

    def add_shape(self, shape):
        for y in range(self.Ny):
            for x in range(self.Nx):
                if shape.containsPoint(x, y):
                    self.geometry[y][x] = True

    def stream(self):
        for i, x, y in zip(range(self.Nl), self.cx, self.cy):
            self.f[:, :, i] = np.roll(self.f[:, :, i], x, axis=1)
            self.f[:, :, i] = np.roll(self.f[:, :, i], y, axis=0)

    def equilibrium(self):
        for i, x, y, v in zip(range(self.Nl), self.cx, self.cy, self.w):
            cu = self.ux * x + self.uy * y
            usqr = self.ux**2 + self.uy**2
            self.feq[:, :, i] = self.rho * v * (1 + 3 * cu + 9 * cu**2 / 2 - 3 * usqr / 2)

    def collision(self):
        self.equilibrium()
        self.f = self.f - (self.f - self.feq) / self.tau

    def boundary_collide(self):
        # Example of bounce-back boundary conditions (simple)
        for i, x, y in zip(range(self.Nl), self.cx, self.cy):
            if self.geometry[self.Ny-1, self.Nx-1]:  # Check if boundary
                self.f[self.geometry, i] = self.f[self.geometry, 7-i]  # Bounce-back rule

    def simulate_step(self):
        self.f[:, -1, [6, 7, 8]] = self.f[:, -2, [6, 7, 8]]
        self.f[:, 0, [2, 3, 4]] = self.f[:, 1, [2, 3, 4]]
        self.stream()
        self.rho = np.sum(self.f, axis=2)
        self.ux = np.sum(self.f * self.cx, axis=2) / self.rho
        self.uy = np.sum(self.f * self.cy, axis=2) / self.rho
        self.boundary_collide()    
        self.collision()

    def simulate(self, Nsteps, plot_every):
        for i in range(Nsteps):
            self.simulate_step()
            if i % plot_every == 0:
                pyplot.imshow(np.sqrt(self.ux**2 + self.uy**2))
                pyplot.pause(.01)
                pyplot.cla()
        pyplot.imshow(np.sqrt(self.ux**2 + self.uy**2))
        pyplot.pause(.01)
        pyplot.cla()

class Circle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y 
        self.r = r 

    def containsPoint(self, x, y):
        return np.sqrt((self.x - x)**2 + (self.y - y)**2) < self.r

if __name__=='__main__':
    lbm = LBM2D(400, 100, 0.53, 3000)  # Using reasonable tau
    lbm.add_shape(Circle(100, 50, 13))
    lbm.simulate(3000, 100)
