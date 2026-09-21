import numpy as np
from matplotlib import pyplot as plt

def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def main():
    Nx = 400
    Ny = 100
    tau = 0.53
    Nt = 3000

    # lattices and weights
    NL = 9
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0 ,1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # initial conditions
    F = np.ones((Ny, Nx, NL)) + 0.01 * np.random.randn(Ny, Nx, NL)
    F[:, :, 3] = 2.3

    cylinder = np.full((Ny, Nx), False)

    for y in range(0, Ny):
        for x in range(0, Nx):
            if distance(x, y, Nx // 4, Ny // 2) < 13:
                cylinder[y, x] = True

    # show the initial state of the cylinder
    # plt.imshow(cylinder, cmap='gray')
    # plt.colorbar()
    # plt.show()

    log_every = 100

    # main loop
    for it in range(Nt):
        if it % log_every == 0:
            print(f"Iteration {it}/{Nt}")

        for i, cx, cy in zip(range(NL), cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)
        bndryF = F[cylinder, :]
        bndryF = bndryF[:,[0, 5, 6, 7, 8, 1, 2, 3, 4]]

        # fluid variables
        rho = np.sum(F, axis=2)
        ux = np.sum(F * cxs, axis=2) / rho
        uy = np.sum(F * cys, axis=2) / rho

        F[cylinder, :] = bndryF
        ux[cylinder] = 0
        uy[cylinder] = 0

        # collision
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
            Feq[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy) + 9/2 * (cx * ux + cy * uy) ** 2 - 3/2 * (ux ** 2 + uy ** 2))
        F += -(1 / tau) * (F - Feq)

        if it % log_every == 0:
            plt.imshow(np.sqrt(ux ** 2 + uy ** 2))
            plt.pause(0.01)
            plt.cla()

if __name__ == "__main__":
    print("Running LBM demo...")
    print("#"*100)
    main()