from typing import Tuple, List, Any

import numpy as np
import itertools

import qiskit.opflow.primitive_ops.pauli_sum_op
from qiskit.opflow import Z, I


def build_Hamiltonian(colors: int, adj_m: np.matrix, penalty: list = []) \
        -> Tuple[int, List[Any]]:
    """
    builds the Hamiltonian out of the adjacency matrix and the number of colors
    :param colors: number of intended colors used
    :param adj_m: adjacency matrix of the given Graph
    :param penalty: penalty term to discourage set colors being used in the coloring process
    :return: Hamiltonian for the given graph structure to be used in the cost function
    """
    assert type(colors) is int, "Number of used colors needs to be given in form of an integer"
    assert type(adj_m) is np.matrix, "Adjacency matrix needs to be in form of a np.matrix"
    assert adj_m.shape[0] == adj_m.shape[1], "Input is not a valid adjacency matrix. Matrix is not a square matrix"
    for i in range(adj_m.shape[0]):
        for j in range(adj_m.shape[0]):
            assert adj_m.item(i, i) == 0, "Input is not a valid adjacency matrix. There should be no elements on the " \
                                          "diagonal "
            assert adj_m.item(i, j) == adj_m.item(j,
                                                  i), "Input is not a valid adjacency matrix. Matrix is not symmetric"
    if penalty != []:
        assert type(penalty) is list, "penalised bits are not given in form of a list"

    m = np.ceil(np.log2(colors))
    m = int(m)
    if (np.log2(colors) != m) & (penalty == []):
        raise ValueError("No penalised bits are given for non allocated colors")
    if penalty != []:
        raise NotImplementedError("penalty term is not properly implemented")
        binarystuff = 0
        for stuff in penalty[0]:
            binarystuff += 1
        assert binarystuff == m, "Somethings wrong I can feel it "

    def add(input):

        result = []
        while True:
            first_pos = input.find(" +")
            if first_pos == -1:
                break
            else:
                second_pos = input.find(" ", first_pos + 1)
                search = input[first_pos:second_pos]

                amounts = 0
                c = 0
                while True:
                    if amounts == -1:
                        break
                    amounts = input.find(search + " ")
                    input = input.replace(search + " ", " ", 1)

                    c += 1

                result.append(search[:2] + str(c - 1) + "*(" + search[2:] + ")")

        return result

    def multiplicate(H, m, penalty_term=False):

        s1 = 0
        H_p = []

        while True:
            s = H.find("+(", s1)
            if s != -1:
                H_p.append(H[s1:s])
            else:
                H_p.append(H[s1:])
                break
            s1 = s + 1

        sep = 9
        result = ""
        var = 0
        lenn = 0
        for q in range(len(H_p)):  # multiplication
            multi = H_p[q]
            for kl in range(int(m) - 1):

                if kl == 0:
                    y = multi.find("I")
                    result += "+" + multi[y] + multi[y + sep]

                    y = multi.find("Z")

                    result += multi[y - 1:y + 5]
                    result += multi[y + sep - 1:y + sep + 5]
                    result += multi[y - 1] + multi[y - 1 + sep]
                    result += multi[y:y + 5] + multi[y + sep:y + sep + 5]

                    for d in range(multi.find(")(") + sep + 1):
                        multi = multi.replace(multi[0], "", 1)

                    terms = 4

                else:

                    multi_point = 0
                    res_point = 0
                    multi_point = multi.find("Z", multi_point)
                    result_old = result[lenn:]

                    for lauf in range(0, terms + 1):

                        plus_pos = result.find("+", res_point)
                        minus_pos = result.find("-", res_point)

                        if ((plus_pos < minus_pos) or minus_pos == -1) and (plus_pos != -1):
                            res_point = plus_pos

                        elif ((minus_pos < plus_pos) or plus_pos == -1) and (minus_pos != -1):
                            res_point = minus_pos
                        elif (plus_pos == -1) and (minus_pos == -1):
                            res_point = len(result_old)

                        if lauf == 0:

                            pass

                        else:
                            result += result_old[var] + multi[multi_point - 1:multi_point + 5] + result_old[
                                                                                                 var + 1:res_point]
                            # add the length of the finished result

                        var = res_point

                        res_point += 1

                    terms *= 2

                    if multi.find(")(") != -1:

                        for hihi in range(multi.find(")(") + 1):
                            multi = multi.replace(multi[0], "", 1)

                result = result.replace("+-", "-")
                result = result.replace("-+", "-")
                result = result.replace("++", "+")
                result = result.replace("--", "+")

            lenn = len(result)

        result = result.replace("+-", "-")
        result = result.replace("-+", "-")
        result = result.replace("++", "+")
        result = result.replace("--", "+")
        result = result.replace("-", " -")
        result = result.replace("+", " +")

        pos = 0
        while True:
            pos = result.find(" ", pos)
            if pos != -1:
                pos2 = result.find(" ", pos + 1)
                if (result[pos + 1] == "+") and (result.find(" -" + result[pos + 2:pos2] + " ") != -1):

                    delete1 = " -" + result[pos + 2:pos2] + " "
                    delete2 = result[pos:pos2] + " "

                    result = result.replace(delete1, " ", 1)
                    result = result.replace(delete2, " ", 1)
                    pos = 0
                elif (result[pos + 1] == "-") and (result.find(" +" + result[pos + 2:pos2] + " ") != -1):
                    delete1 = " +" + result[pos + 2:pos2] + " "
                    delete2 = result[pos:pos2] + " "

                    result = result.replace(delete1, " ", 1)
                    result = result.replace(delete2, " ", 1)
                    pos = 0
                else:
                    pos += 1
            elif pos == -1:
                break

        result = result.replace("+II", "")
        result = result.replace("II", "").replace("  ", " ")
        result += " "
        if penalty_term == False:
            result = add(result)

        return result

    def convert_to_operator(input, colors, adj_m, penalty_term=False):

        operator = 0
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
                        operator += operator_aux

        return operator

    def add_penalty(penalty):
        H_pen = ""
        for penalty_length in range(len(penalty)):
            for bits in range(m):
                if penalty[penalty_length][bits] == "1":
                    H_pen += "(I-Z_v," + str(bits + 1) + ")"
                if penalty[penalty_length][bits] == "0":
                    H_pen += "(I+Z_v," + str(bits + 1) + ")"

        H_pen = multiplicate(H_pen, m, penalty_term=True)

        start = 0
        H_penlist = []
        while True:
            end = start + 1
            blank1 = H_pen.find(" ", start)
            blank2 = H_pen.find(" ", end)
            if blank2 == -1:
                break
            H_penlist.append(H_pen[blank1:blank2])
            start = blank2

        H_pen = convert_to_operator(H_penlist, colors, adj_m, penalty_term=True)

        return H_pen

    H_1 = ""

    binary = list(map(list, itertools.product([0, 1], repeat=int(m))))
    for k in range(len(binary)):

        for l in range(int(m)):
            H_1 += "(I+" + str(np.power(-1, binary[k][l])) + "*Z_v," + str(l + 1) + ")" + \
                   "(I+" + str(np.power(-1, binary[k][l])) + "*Z_w," + str(l + 1) + ")"  # v1,w1,v2,w2
        H_1 += "+"
    x = str(H_1)
    y = x[:-1]
    y = y.replace("+-", "-")
    y = y.replace("1*", "")
    result = multiplicate(y, 2 * m)
    Ham_Terms = result
    Ham = convert_to_operator(result, colors, adj_m)
    if penalty != []:
        raise print("still under construction")

        penalty_term = add_penalty(penalty)
        Ham += penalty_term

    return Ham, Ham_Terms
