import numpy as np  
from matplotlib import pyplot
import pickle

class LBM2D:
	cx = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
	cy = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
	w = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])
	Nl = 9
	c = np.array([cx, cy])
	dims = 2

	u_max  = 0.04/2      # maximum velocity
	nu     = (2-1)/6     # kinematic shear viscosity
	rho0   = 1               # rest density


	def __init__(self, Nx, Ny, tau):
		self.Nx = Nx
		self.Ny = Ny
		self.tau = tau
		self.f = np.ones((Ny, Nx, LBM2D.Nl)) + 0.01 * np.random.randn(Ny, Nx, LBM2D.Nl)
		#self.f = np.load('taylorgreen.np.npy') #+ 0.01 * np.random.randn(Ny, Nx, LBM2D.Nl)

		# hocu da tecem desno
		self.f[:, :, 3] = 2.3 # sta god

		self.geometry = np.full((Ny, Nx), False)
		
		# self.rho = np.sum(self.f, 2)
		# self.ux = np.sum(self.f * self.cx, 2) / self.rho
		# self.uy = np.sum(self.f * self.cy, 2) / self.rho

		self.simres = []
		self.rho = np.ones((Ny, Nx))
		self.ux = np.zeros((Ny, Nx))  # Set initial velocity to 0
		self.uy = np.zeros((Ny, Nx))
		self.u = np.array([self.ux, self.uy])

		self.feq = np.zeros(self.f.shape)

		[self.tg_rho, self.tg_u, self.tg_P] = self.taylorgreen(0, self.nu, self.rho0, self.u_max)
		#self.f = LBM2D.equilibrium(self.f.shape, self.tg_rho, self.tg_u[0, :], self.tg_u[1, :]) 
		#self.feq = self.f


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

	@staticmethod
	def equilibrium(shape, rho, ux, uy, u):
		feq = np.zeros(shape)
		for i, x, y, v in zip(range(LBM2D.Nl), LBM2D.cx, LBM2D.cy, LBM2D.w):
			feq[:, :, i] = rho * v * (1 + 3 * (x*ux + y*uy) + 9 * (x*ux + y*uy)**2 / 2 - 3 * (ux**2 + uy**2) / 2)
		# c . u 
		# feq(r) = rho(r) * w * [1 + 3 u(r) . c + 9/2 (c . u(r))^2 - 3/2 u(r) . u(r)]
		
		c = LBM2D.c.reshape(2, 9, 1, 1)  # -> (2, 9, 1, 1)
		w = LBM2D.w.reshape(9, 1, 1)     # -> (9, 1, 1)
		rho = rho[np.newaxis]      # -> (1, 100, 400)
		
		# Compute dot products
		cu = np.sum(c * u[:, np.newaxis], axis=0)  # (9, 100, 400)
		uu = np.sum(u * u, axis=0)                 # (100, 400)
		
		# Compute feq using the formula
		feq = rho * w * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * uu)
		feq = np.transpose(feq, (1, 2, 0))
		return feq

	def collision(self):
		#self.f = self.f + -(1 / self.tau) * (self.f - self.feq) 
		self.f = self.f * (1 - 1 / self.tau) + self.feq / self.tau

	def boundary_collide(self):
		bound = self.f[self.geometry, :]
		bound = bound[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

		self.f[self.geometry, :] = bound

		self.ux[self.geometry] = 0 
		self.uy[self.geometry] = 0 

		for i in range(LBM2D.dims):
			self.u[i][self.geometry] = 0

	def simulate_step(self):
		#Zhou-Hei boundary
		self.f[:, -1, [6, 7, 8]] = self.f[:, -2, [6, 7, 8]]
		self.f[:, 0, [2, 3, 4]] = self.f[:, 1, [2, 3, 4]]

		# right stream
		#self.f[:, 0, 3] = 2.3

		self.stream()

		self.rho = np.sum(self.f, 2)
		self.ux = np.sum(self.f * LBM2D.cx, 2) / self.rho
		self.uy = np.sum(self.f * LBM2D.cy, 2) / self.rho

		for i in range(LBM2D.dims):
			self.u[i] = np.sum(self.f * LBM2D.c[i], 2) / self.rho

		self.boundary_collide()	

		#self.simres.append(np.sqrt(self.ux**2 + self.uy**2))
		self.feq = self.equilibrium(self.f.shape, self.rho, self.ux, self.uy, self.u)
		self.collision()

		# compare with analytical solution
		

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
	def taylorgreen(self, t, nu, rho0, u_max):
		kx = 2*np.pi/self.Nx
		ky = 2*np.pi/self.Ny
		td = 1/(nu*(kx*kx+ky*ky))
		
		x = np.arange(self.Nx)+0.5
		y = np.arange(self.Ny)+0.5
		[X, Y] = np.meshgrid(x,y)

		u = np.array([-u_max*np.sqrt(ky/kx)*np.cos(kx*X)*np.sin(ky*Y)*np.exp(-t/td),
					u_max*np.sqrt(kx/ky)*np.sin(kx*X)*np.cos(ky*Y)*np.exp(-t/td)])
		P = -0.25*rho0*u_max*u_max*((ky/kx)*np.cos(2*kx*X)+(kx/ky)*np.cos(2*ky*Y))*np.exp(-2*t/td)
		rho = rho0+3*P
		return [rho, u, P]


class Circle:
	def __init__(self, x, y, r):
		self.x = x
		self.y = y 
		self.r = r 

	def containsPoint(self, x, y):
		return np.sqrt((self.x - x)**2 + (self.y - y)**2) < self.r

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
	lbm = LBM2D(400, 100, 0.53)
	lbm.add_shape(Circle(100, 50, 10))
	lbm.add_shape(Circle(120, 50, 10))
	lbm.add_shape(Circle(140, 50, 10))
	lbm.add_shape(Circle(110, 70, 10))
	lbm.add_shape(Circle(130, 70, 10))
	lbm.add_shape(Square(200, 40, 20))
	lbm.add_shape(Rectangle(300, 40, 20, 10))
	lbm.simulate(4000, 10)