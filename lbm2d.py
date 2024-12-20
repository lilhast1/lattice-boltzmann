import numpy as np  
from matplotlib import pyplot
import pickle

class LBM2D:
	cx = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
	cy = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
	w = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])
	Nl = 9


	def __init__(self, Nx, Ny, tau, Nt):
		self.Nx = Nx
		self.Ny = Ny
		self.tau = tau
		self.Nt = Nt
		self.f = np.load('taylorgreen.np.npy') #+ 0.01 * np.random.randn(Ny, Nx, LBM2D.Nl)

		# hocu da tecem desno
		#self.f[:, :, 3] = 2.3 # sta god

		self.geometry = np.full((Ny, Nx), False)
		
		# self.rho = np.sum(self.f, 2)
		# self.ux = np.sum(self.f * self.cx, 2) / self.rho
		# self.uy = np.sum(self.f * self.cy, 2) / self.rho

		self.simres = []
		self.rho = np.ones((Ny, Nx))
		self.ux = np.zeros((Ny, Nx))  # Set initial velocity to 0
		self.uy = np.zeros((Ny, Nx))

		self.feq = np.zeros(self.f.shape)


	def add_geometry(self, shape_map):
		self.geometry = self.geometry + shape_map

	def add_shape(self, shape):
		for y in range(self.Ny):
			for x in range(self.Nx):
				if shape.containsPoint(x, y):
					self.geometry[y, x] = True


	def stream(self):
		for i, x, y in zip(range(LBM2D.Nl), LBM2D.cx, LBM2D.cy):
			self.f[:, :, i] = np.roll(self.f[:, :, i], x, axis=1)
			self.f[:, :, i] = np.roll(self.f[:, :, i], y, axis=0)

	def equilibrium(self):
		self.feq = np.zeros(self.f.shape)
		for i, x, y, v in zip(range(LBM2D.Nl), LBM2D.cx, LBM2D.cy, LBM2D.w):
			self.feq[:, :, i] = self.rho * v * (1 + 3 * (x*self.ux + y*self.uy) + 9 * (x*self.ux + y*self.uy)**2 / 2 - 3 * (self.ux**2 + self.uy**2) / 2)

	def collision(self):
		self.f = self.f + -(1 / self.tau) * (self.f - self.feq) 

	def boundary_collide(self):
		bound = self.f[self.geometry, :]
		bound = bound[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

		self.f[self.geometry, :] = bound

		self.ux[self.geometry] = 0 
		self.uy[self.geometry] = 0 

	def simulate_step(self):
		#Zhou-Hei boundary
		# self.f[:, -1, [6, 7, 8]] = self.f[:, -2, [6, 7, 8]]
		# self.f[:, 0, [2, 3, 4]] = self.f[:, 1, [2, 3, 4]]

		self.stream()

		self.rho = np.sum(self.f, 2)
		self.ux = np.sum(self.f * LBM2D.cx, 2) / self.rho
		self.uy = np.sum(self.f * LBM2D.cy, 2) / self.rho

		self.boundary_collide()	
		self.simres.append(np.sqrt(self.ux**2 + self.uy**2))
		self.equilibrium()
		self.collision()

	def simulate(self, Nsteps, plot_every):
		for i in range(Nsteps):
			self.simulate_step()
			if i % plot_every == 0:
				pyplot.imshow(np.sqrt(self.ux**2 + self.uy**2))
				pyplot.pause(.01)
				pyplot.cla()
			print(i)
		# with open('my.pkl', 'wb') as outfile:
		# 	pickle.dump(self.simres, outfile, pickle.HIGHEST_PROTOCOL)

class Circle:
	def __init__(self, x, y, r):
		self.x = x
		self.y = y 
		self.r = r 

	def containsPoint(self, x, y):
		return np.sqrt((self.x - x)**2 + (self.y - y)**2) < self.r

if __name__=='__main__':
	lbm = LBM2D(64, 64, 1, 3000)
	lbm.simulate(4000, 10)