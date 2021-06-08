import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def report(iteration, x_curr, x_prev, f_curr, f_prev):
    print(
        f"Iteration {iteration}:\n\t"
        f"Current location: {x_curr}\n\t"
        f"Current objective: {f_curr}\n\t"
        f"Step size: {np.linalg.norm(x_curr - x_prev)}\n\t"
        f"Objective change: {np.abs(f_curr - f_prev)}\n\t"
    )


def visualize_path(f, path, f_hist, title, levels=10, margin=0.1):
    path = np.array(path)
    p0 = path[0]
    p1 = path[-1]

    xmin, xmax = min(p0[0], p1[0]), max(p0[0], p1[0])
    ymin, ymax = min(p0[1], p1[1]), max(p0[1], p1[1])
    xdiff = xmax - xmin
    ydiff = ymax - ymin
    x = np.linspace(xmin - margin * xdiff, xmax + margin * xdiff, 200)
    y = np.linspace(ymin - margin * ydiff, ymax + margin * ydiff, 200)
    xx, yy = np.meshgrid(x, y)

    def f2(x, y):
        return f(np.array([x, y]))[0]

    z = np.vectorize(f2)(xx, yy)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    CS = ax.contour(x, y, z, levels=levels)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.plot(path[:, 0], path[:, 1], marker="*", color="r")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Path visualization")
    ax.grid()

    ax = axes[1]
    ax.plot(np.arange(1, len(f_hist) + 1), f_hist)
    ax.set_title("Objective function history")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objcetive function")
    ax.grid()

    fig.suptitle(title)

    plt.show()


def visualize_qp_path(x_hist):
    x_hist = np.array(x_hist)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ax.scatter(x_hist[:, 0], x_hist[:, 1], x_hist[:, 2])
    ax.scatter(x_hist[-1:, 0], x_hist[-1:, 1], x_hist[-1:, 2], marker="*", c="r", s=200)
    ax.plot(x_hist[:, 0], x_hist[:, 1], x_hist[:, 2])

    ax.set_title("QP path visualization")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)

    xx, yy = np.meshgrid(x, y)
    zz = np.maximum(0, 1 - xx - yy)
    c = np.zeros((xx.shape[0], yy.shape[0], 4))
    c[np.abs(xx + yy + zz - 1) <= 1e-2] = np.array([1, 1, 0, 0.5])

    ax.plot_surface(xx, yy, zz, zorder=-1, facecolors=c, vmin=0, vmax=1, linewidth=0)

    plt.show()


def visualize_lp_path(x_hist):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    x_hist = np.array(x_hist)
    ax.scatter([x_hist[-1, 0]], [x_hist[-1, 1]], c="r", marker="*", zorder=2, s=100)
    ax.plot(x_hist[:, 0], x_hist[:, 1], marker="o", markersize=3, zorder=1)
    ax.set_xlim([0.4, 2.1])
    ax.set_ylim([-0.05, 1.1])

    x = np.linspace(*ax.get_xlim(), 200)
    y = np.linspace(*ax.get_ylim(), 200)
    xx, yy = np.meshgrid(x, y)

    m1 = yy >= -xx + 1
    m2 = yy <= 1
    m3 = xx <= 2
    m4 = yy >= 0

    M = m1 & m2 & m3 & m4
    ax.pcolormesh(xx, yy, np.where(M, 1, np.nan), cmap="viridis_r", alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.grid()
    ax.set_title("LP path visualization")

    plt.show()
