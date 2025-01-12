import numpy as np  
from matplotlib import pyplot as plt
import pickle
from jax import jit

class LBM:
	cx = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1], dtype=np.float32)
	cy = np.array([0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 1, -1, 1, -1, 0, 0, 0, 0], dtype=np.float32)
	cz = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1], dtype=np.float32)
	
	
	w = np.array([1/3, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/36,  1/36,  1/36,  1/36,  1/36,  1/36,  1/36,  1/36,  1/36,  1/36,  1/36,  1/36], dtype=np.float32)
	Nl = 19
	c = np.array([cx, cy, cz], dtype=np.float32)
	dims = 3

	u_max  = 0.04/2      # maximum velocity
	nu     = (2-1)/6     # kinematic shear viscosity
	rho0   = 1               # rest density


	def __init__(self, Nx, Ny, Nz, tau):
		self.Nx = Nx
		self.Ny = Ny
		self.Nz = Nz
		self.tau = tau
		self.f = np.ones((Nz, Ny, Nx, LBM.Nl), dtype=np.float32) + np.float32(0.01) * np.random.randn(Nz, Ny, Nx, LBM.Nl)
		#self.f = np.load('taylorgreen.np.npy') #+ 0.01 * np.random.randn(Ny, Nx, LBM.Nl)

		# hocu da tecem desno
		self.f[:, :, :, 1] = 2.3 # sta god

		self.geometry = np.full((Nz, Ny, Nx), False)
		
		# self.rho = np.sum(self.f, 2)
		# self.ux = np.sum(self.f * self.cx, 2) / self.rho
		# self.uy = np.sum(self.f * self.cy, 2) / self.rho

		self.simres = []
		self.rho = np.ones((Nz, Ny, Nx), dtype=np.float32)
		self.ux = np.zeros((Nz, Ny, Nx), dtype=np.float32)  # Set initial velocity to 0
		self.uy = np.zeros((Nz, Ny, Nx), dtype=np.float32)
		self.uz = np.zeros((Nz, Ny, Nx), dtype=np.float32)
		self.u = np.array([self.ux, self.uy, self.uz], dtype=np.float32)

		self.feq = np.zeros(self.f.shape, dtype=np.float32)

		#[self.tg_rho, self.tg_u, self.tg_P] = self.taylorgreen(0, self.nu, self.rho0, self.u_max)
		#self.f = LBM.equilibrium(self.f.shape, self.tg_rho, self.tg_u[0, :], self.tg_u[1, :]) 
		#self.feq = self.f


	def add_geometry(self, shape_map):
		self.geometry = self.geometry + shape_map

	
	def add_shape(self, shape):
		Z, Y, X = np.meshgrid(np.arange(self.Nz), np.arange(self.Ny), np.arange(self.Nx))
		containsPoint = np.vectorize(lambda z, y, x: shape.containsPoint(x, y, z))
		mask = containsPoint(Z, Y, X)
		self.geometry |= mask	
	def stream(self):
		for i, x, y, z in zip(range(LBM.Nl), LBM.cx, LBM.cy, LBM.cz):
			self.f[:, :, :, i] = np.roll(self.f[:, :, :, i], x, axis=2)
			self.f[:, :, :, i] = np.roll(self.f[:, :, :, i], y, axis=1)
			self.f[:, :, :, i] = np.roll(self.f[:, :, :, i], z, axis=0)

	@staticmethod
	def equilibrium(insu, rho, u):
		# c . u 
		# feq(r) = rho(r) * w * [1 + 3 u(r) . c + 9/2 (c . u(r))^2 - 3/2 u(r) . u(r)]
		
		c = LBM.c.reshape(LBM.dims, LBM.Nl, 1, 1, 1)  # -> (2, 9, 1, 1)
		w = LBM.w.reshape(LBM.Nl, 1, 1, 1)     # -> (9, 1, 1)
		#rho = rho[np.newaxis]      # -> (1, 100, 400)
		
		# Compute dot products
		cu = np.sum(c * u[:, np.newaxis], axis=0)  # (9, 100, 400)
		uu = u[0, :, :, :]**2 + u[1, :, :, :]**2 +  u[2, :, :, :]**2 #np.sum(u * u, axis=0)                 # (100, 400)
		
		# Compute feq using the formula
		feq = rho * w * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * uu)
		feq = np.transpose(feq, (1, 2, 3, 0))
		return feq
	def collision(self):
		#self.f = self.f + -(1 / self.tau) * (self.f - self.feq) 
		self.f = self.f * (1 - 1 / self.tau) + self.feq / self.tau
	def boundary_collide(self):
		bound = self.f[self.geometry, :]
		bound = bound[:, [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]]

		self.f[self.geometry, :] = bound

		for i in range(LBM.dims):
			self.u[i, self.geometry] = 0
	
	def simulate_step(self):
		#Zhou-Hei boundary
		# self.f[-1, :, :, [5, 11, 14, 15, 18]] = self.f[-2, :, :, [6, 12, 13, 16, 17]]
		# self.f[0, :, :, [6, 12, 13, 16, 17]] = self.f[1, :, :, [5, 11, 14, 15, 18]]

		# right stream
		#self.f[:, 0, 3] = 2.3

		self.stream()

		self.rho = np.sum(self.f, 3)

		for i in range(LBM.dims):
			self.u[i] = np.sum(self.f * LBM.c[i], 3) / self.rho

		self.boundary_collide()	

		#self.simres.append(np.sqrt(self.ux**2 + self.uy**2))
		self.feq = self.equilibrium(self.f.shape, self.rho, self.u)
		self.collision()

		# compare with analytical solution
		

	def simulate(self, Nsteps, plot_every, path='sphere'):
		Es = []
		for i in range(Nsteps):
			self.simulate_step()
			if i % plot_every == 0:
				data1 = self.u[0, :, :, :]**2 + self.u[1, :, :, :]**2 +  self.u[2, :, :, :]**2
				data2 = data1[50,:,:]
				fig, axes = plt.subplots(1, 1, figsize=(10, 10))
				axes.imshow(data2.transpose())
				axes.set_title('|u|')
				plt.savefig(f'{path}/lbm3d_{i}.png')
			E = np.sum(self.rho * self.u**2 / 2)
			print(f'itr={i} E={E}')
			Es.append(E)
		# with open('my.pkl', 'wb') as outfile:
		# 	pickle.dump(self.simres, outfile, pickle.HIGHEST_PROTOCOL)
		return Es


class Sphere:
	def __init__(self, x, y, z, r):
		self.x = x
		self.y = y 
		self.z = z
		self.r = r 
	def containsPoint(self, x, y, z):
		return np.sqrt((self.x - x)**2 + (self.y - y)**2 + (self.z - z)**2) < self.r

class Square:
	def __init__(self, x, y, a):
		self.x = x
		self.y = y 
		self.a = a 

	def containsPoint(self, x, y):
		return x >= self.x and x <= self.x + self.a and  y >= self.y and y <= self.y + self.a

class Rectangle:
	def __init__(self, x, y, a, b):
		self.x = x
		self.y = y 
		self.a = a 
		self.b = b

	def containsPoint(self, x, y):
		return x >= self.x and x <= self.x + self.a and  y >= self.y and y <= self.y + self.b


if __name__=='__main__':
	lbm = LBM(400, 100, 100, 0.53)
	lbm.add_shape(Sphere(100, 50, 50, 10))
	
	lbm.simulate(200, 10)