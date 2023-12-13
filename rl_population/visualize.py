from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np


def plot_matrix(matrix: np.ndarray):
    """Plots a matrix of two dimensions"""
    assert len(matrix.shape) == 2

    # initialize plot
    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap="Wistia")

    # write agent ids on major axis
    plt.xticks(range(matrix.shape[0]), range(matrix.shape[0]))
    ax.xaxis.tick_top()
    plt.yticks(range(matrix.shape[1]), range(matrix.shape[1]))

    # draw grid on minor axis
    ax.grid(which="minor", linestyle="-", color="k", linewidth=2)
    ax.set_xticks([x - 0.5 for x in range(1, matrix.shape[0])], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, matrix.shape[1])], minor=True)

    # write score values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j - 0.2,
                i + 0.1,
                f"{matrix[i][j]:.1f}",
                color="k",
                fontsize=15,
                fontweight="bold",
            )

    # # write legends
    # plt.title(f"Cooperation scores between agents on '{matrix}'", fontsize=20, y=-0.1)
    # plt.xlabel("Agents ids as second player", fontsize=15)
    # plt.ylabel("Agents ids as first player", fontsize=15)

    plt.show()
