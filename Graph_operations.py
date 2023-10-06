import numpy as np
import networkx as nx
import itertools, os
import matplotlib.pyplot as plt


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def AMreduce(AM_old):
    """
    Checks if any nodes of degree 1 are in the given graph and reduces the matrix accordingly
    :param AM_old: Adjacency Matrix which will be checked if it can be reduced
    :return: Reduced Adjacency Matrix
    """
    def ReduceAM(AM):
        AM_new = AM
        d = 0
        for i in range(AM.shape[1]):
            Zero = 0
            for j in range(AM.shape[1]):

                if AM.item((i, j)) != 0:
                    Zero += 1

            if Zero <= 1:
                AM_new = np.delete(AM_new, obj=i - d, axis=0)
                AM_new = np.delete(AM_new, obj=i - d, axis=1)
                d += 1

        return AM_new

    while True:
        AM_new = ReduceAM(AM_old)
        if AM_new.size == AM_old.size:
            break
        else:
            AM_old = AM_new
    return AM_new


def check_coloring(A: np.array, x, directory: str, show: bool, draw_graph: bool = False, print_outs: bool = True):
    colorlist = []
    Test = None
    for i in range(0, len(x), 2):

        if (x[i] == 0) & (x[i + 1] == 0):
            colorlist.append("red")
        elif (x[i] == 0) & (x[i + 1] == 1):
            colorlist.append("green")
        elif (x[i] == 1) & (x[i + 1] == 0):
            colorlist.append("blue")
        elif (x[i] == 1) & (x[i + 1] == 1):
            colorlist.append("yellow")

    for permutation in range(A.shape[0]):
        Test = True
        colorlist2 = []
        for i in range(A.shape[0]):
            colorlist2.append(colorlist[i - permutation])
        G = nx.from_numpy_matrix(A)
        for u in range(nx.number_of_nodes(G)):
            for v in nx.neighbors(G, u):
                if colorlist2[u] == colorlist2[v]:
                    Test = False
                    break
            if not Test:
                break

        if Test:
            if draw_graph:
                plt.figure()
                nx.draw(G, node_color=colorlist2, with_labels=True)
                plt.title(str(permutation))
                plt.savefig(os.path.join(directory, "colored_graph"))
                if show:
                    plt.show()
            if print_outs:
                print(bcolors.OKGREEN + "Coloring successful with solution bitstring", colorlist2, bcolors.ENDC)
            break

    if not Test:
        if print_outs:
            print(bcolors.FAIL + "Coloring Failed" + bcolors.ENDC)

    return Test


def brute_force_coloring(A, num_colors):

    length = A.shape[0]
    gr = nx.from_numpy_matrix(A)
    counter = 0
    possibilities = list(itertools.product([0, 1], repeat=int(length*np.floor(np.log2(num_colors)))))
    for x in possibilities:
        Test = True
        colorlist = []
        for i in range(0, len(x), 2):
            if (x[i] == 0) & (x[i + 1] == 0):
                colorlist.append("red")
            elif (x[i] == 0) & (x[i + 1] == 1):
                colorlist.append("green")
            elif (x[i] == 1) & (x[i + 1] == 0):
                colorlist.append("blue")
            elif (x[i] == 1) & (x[i + 1] == 1):
                colorlist.append("yellow")
        for u in range(nx.number_of_nodes(gr)):
            for v in nx.neighbors(gr, u):
                if colorlist[u] == colorlist[v]:
                    Test = False
                    break
            if not Test:
                break
        if Test:
            counter += 1
    ratio = counter/len(possibilities)
    print(ratio, counter)
    return ratio, counter


def calculate_c_value(adj_m: np.array):
    """
    :param adj_m: Adjacency matrix of the given graph
    :return: complexity value of the graph based on how many colorable solution the graph posses
    """
    V = adj_m.shape[0]
    E = 0
    for i in range(V):
        for j in range(i):
            if adj_m.item(i, j) == 1:
                E += 1

    c = (2*E)/(V - 1)
    print(c)

    return c
