import numpy as np

from qiskit.opflow import I, Z

from random import shuffle


def create_SHA_layers(colors: int, layers: int, type: str, Ham_terms, adj_m):
    """
    partitions the Hamiltonian into a set amount of layers after the predefined partitioning strategies
    :param colors: maximum number of colors used in the graph coloring problem
    :param layers: number of times the Ham will be partitioned
    :param type: partitiong strategy
    :param Ham_terms: Individual Terms of the Hamiltonina
    :param adj_m: Adjacency Matrix of the given graph
    :return:
    """
    def convert_to_operator(input, colors, adj_m, penalty_term=False):

        operator = []
        bi = np.ceil(np.log2(colors)) * adj_m.shape[0]

        for terms in input:
            for v in range(adj_m.shape[0]):
                for w in range(adj_m.shape[0]):
                    aux = ""
                    for i in range(int(bi)):
                        aux += "I"

                    if adj_m.item(v, w) != 0:
                        v1 = terms.find("v,1")
                        v2 = terms.find("v,2")
                        if v1 != -1:
                            # 01 23 45  # 0123456789
                            # 0 = 01 1 = 23 2=45 3=67 4=89

                            aux = aux[:2 * v] + "Z" + aux[2 * v + 1:]

                            if penalty_term is False:
                                aux = aux[:2 * w] + "Z" + aux[2 * w + 1:]

                        if v2 != -1:
                            aux = aux[:2 * v + 1] + "Z" + aux[2 * v + 2:]

                            if penalty_term is False:
                                aux = aux[:2 * w + 1] + "Z" + aux[2 * w + 2:]

                        plus = terms.find("+")
                        star = terms.find("*")

                        operator_aux = 0

                        if penalty_term is False:
                            operator_aux += float(terms[plus:star]) * int(adj_m.item(v, w))
                        else:
                            if terms.find("+") != -1:
                                operator_aux += 1
                            elif terms.find("-") != -1:
                                operator_aux -= 1

                        for z in range(int(bi)):
                            if aux[z] == "Z":
                                try:
                                    operator_aux *= Z
                                except:
                                    operator_aux ^= Z
                            elif aux[z] == "I":
                                try:
                                    operator_aux *= I
                                except:
                                    operator_aux ^= I
                        operator.append(operator_aux)

        return operator

    def convert_to_operator_for_nodewise(input, colors, adj_m, strategy, penalty_term=False):
        # bugged for penalty term
        operator = []
        bi = np.ceil(np.log2(colors)) * adj_m.shape[0]

        for v in strategy:
            operator2 = []
            for w in range(adj_m.shape[0]):
                for terms in input:

                    aux = ""
                    for i in range(int(bi)):
                        aux += "I"

                    if adj_m.item(v, w) != 0:
                        v1 = terms.find("v,1")
                        v2 = terms.find("v,2")
                        if v1 != -1:
                            # 01 23 45  # 0123456789
                            # 0 = 01 1 = 23 2=45 3=67 4=89

                            aux = aux[:2 * v] + "Z" + aux[2 * v + 1:]

                            if penalty_term is False:
                                aux = aux[:2 * w] + "Z" + aux[2 * w + 1:]

                        if v2 != -1:
                            aux = aux[:2 * v + 1] + "Z" + aux[2 * v + 2:]

                            if penalty_term is False:
                                aux = aux[:2 * w + 1] + "Z" + aux[2 * w + 2:]

                        plus = terms.find("+")
                        star = terms.find("*")

                        operator_aux = 0

                        if penalty_term is False:
                            operator_aux += float(terms[plus:star]) * int(adj_m.item(v, w))
                        else:
                            if terms.find("+") != -1:
                                operator_aux += 1
                            elif terms.find("-") != -1:
                                operator_aux -= 1

                        for z in range(int(bi)):
                            if aux[z] == "Z":
                                try:
                                    operator_aux *= Z
                                except:
                                    operator_aux ^= Z
                            elif aux[z] == "I":
                                try:
                                    operator_aux *= I
                                except:
                                    operator_aux ^= I
                        operator2.append(operator_aux)
            operator.append(operator2)

        return operator

    assert type == "random" or type == "sequential" or type == "node_wise", \
        "Unknown layering type. Please choose either 'random','sequential' or 'node_wise'"
    # für nicht passende Farben penalty berücksichtgien

    length = 0
    for i in range(adj_m.shape[0]):
        for j in range(adj_m.shape[0]):
            if adj_m.item(i, j):
                length += 1
    length = length * len(Ham_terms)

    if type == "random":
        assert length > layers, f"Too many layers for the terms. Number of terms {length}, Number of layers {layers}"
        not_random_list = convert_to_operator(Ham_terms, colors, adj_m)
        shuffle(not_random_list)
        random_list = np.array_split(not_random_list, layers)
        operator = []
        for v in range(len(random_list)):
            operator_aux = 0
            for w in range(len(random_list[v])):
                operator_aux += random_list[v][w]
            if v == 0:
                operator.append(operator_aux)
            else:
                operator.append(operator[v - 1] + operator_aux)

    if type == "sequential":
        assert length > layers, f"Too many layers for the terms. Number of terms {length}, Number of layers {layers}"
        not_random_list = convert_to_operator(Ham_terms, colors, adj_m)

        random_list = np.array_split(not_random_list, layers)
        operator = []
        for v in range(len(random_list)):
            operator_aux = 0
            for w in range(len(random_list[v])):
                operator_aux += random_list[v][w]
            if v == 0:
                operator.append(operator_aux)
            else:
                operator.append(operator[v - 1] + operator_aux)

    if type == "node_wise":
        assert layers <= adj_m.shape[0], f"Unable to partition the graph into more parts than there are nodes. Please " \
                                         f"choose a value <={adj_m.shape[0]}"
        node_strategy = [0]  # starting point top of adj matrix
        node_x = 0
        while len(node_strategy) < adj_m.shape[0]:

            for node_y in range(adj_m.shape[0]):
                if adj_m.item(node_x, node_y) != 0 and node_y not in node_strategy:
                    node_x = node_y
                    jump = False
                    break
                jump = True
            counter = 0
            while jump: #procedure to find neighbouring node not in node_strategy
                for node_y in range(adj_m.shape[0]):
                    if adj_m.item(node_x, node_y) != 0 and node_y not in node_strategy:
                        node_x = node_y
                        jump = False
                        break
                if jump:
                    try:
                        node_x = node_strategy[counter]
                    except IndexError:
                        print("Invalid Input graph. Input consists of one or more independet graphs,"
                              " which can be looked at seperatly. ")
                    counter +=1
            node_strategy.append(node_x)

        inter_terms = convert_to_operator_for_nodewise(Ham_terms, colors, adj_m, node_strategy)

        inter_operator = []

        for v in range(len(inter_terms)):

            operator_aux = 0
            for w in range(len(inter_terms[v])):
                operator_aux += inter_terms[v][w]
            inter_operator.append(operator_aux)

        split_operator = np.array_split(np.array(inter_operator,dtype=object), layers)


        operator = []
        for i in range(len(split_operator)):
            operator_aux = 0
            for j in range(len(split_operator[i])):
                operator_aux += split_operator[i][j]
            if i == 0:
                operator.append(operator_aux)
            else:
                operator.append(operator[i - 1] + operator_aux)

    return operator

