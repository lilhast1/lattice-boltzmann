import numpy as np
from matplotlib import pyplot
import pickle

def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def main():
    Nx = 400
    Ny = 100
    tau = 0.53
    Nt = 3000

    Nl = 9

    plot_every = 3000

    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    simres = []

    #inital conditions
    f = np.ones((Ny, Nx, Nl)) + 0.01 * np.random.randn(Ny, Nx, Nl)
    f[:, :, 3] = 2.3

    cylinder = np.full((Ny, Nx), False)
    for y in range(Ny):
        for x in range(Nx):
            if distance(Nx // 4, Ny // 2, x, y) < 13:
                cylinder[y, x] = True 
    
    for it in range(Nt):
        print(it)

        #stream
        for i, cx, cy in zip(range(Nl), cxs, cys):
            f[:, :, i] = np.roll(f[:, :, i], cx, axis=1)
            f[:, :, i] = np.roll(f[:, :, i], cy, axis=0)
        
        rho = np.sum(f, 2)
        ux = np.sum(f * cxs, 2) / rho
        uy = np.sum(f * cys, 2) / rho

        

        # boundary collisions

        bndry = f[cylinder, :]
        bndry = bndry[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        f[cylinder, :] = bndry
        
        ux[cylinder] = 0
        uy[cylinder] = 0
        simres.append(np.sqrt(ux**2 + uy**2))
        # collide
        feq = np.zeros(f.shape)
        for i, cx, cy, w in zip(range(Nl), cxs, cys, weights):
            feq[:, :, i] = rho * w * (1 + 3 * (cx*ux + cy*uy) + 9 * (cx*ux + cy*uy)**2 / 2 - 3 * (ux**2 + uy**2) / 2)
        f = f + -(1 / tau) * (f - feq)

        #plot
        if it % plot_every == 0:
            pyplot.imshow(np.sqrt(ux**2 + uy**2))
            pyplot.pause(0.01)
            pyplot.cla()
    
    with open('yt.pkl', 'wb') as outfile:
        pickle.dump(simres, outfile, pickle.HIGHEST_PROTOCOL)


if __name__=='__main__':
    main()

    with open('yt.pkl', 'rb') as infile:
        result = pickle.load(infile)
    with open('my.pkl', 'rb') as infile:
        myres = pickle.load(infile)
    maxdiff = 0
    for i in range(len(result)):
        diff = np.subtract(result[i], myres[i])
        diff = np.absolute(diff)
        if diff.max() > maxdiff:
            maxdiff = diff.max()
    print('MAXIMUM DIFFRENCE', maxdiff)

