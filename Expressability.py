import numpy as np
import os, time, progressbar
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from random import random

from qiskit import (QuantumCircuit, execute, Aer)
from qiskit import QuantumRegister, ClassicalRegister

from Circuits import circuit


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


def expressability(instructions: list, entanglement_block: str, entangle_map: list, Num_qubits: int, name: str,
                   directory: str, num_cricparams: int, repetitions: int, num_shots: int = 2000, num_params: int = 200,
                   final_layer: bool = False, alternating_layer: bool = False, detailed: bool = False,
                   FLAG_show_saved: bool = False):
    """
    Calculate the expressibility of a given circiut. Can be done via histogram ansatz

    :param directory: directory to save the results in
    :param name: name of the Circuit
    :param alternating_layer: describes wether or not the entanglement strategy changes
    :param final_layer: describes wether or not a final rotation layer exists in Two local function
    :param instructions: type of gates used in the rotation layer
    :param entanglement_block: gates used in the entanglement layer
    :param entangle_map: instruction on the entanglement structure of the entanglement layer
    :param Num_qubits: Number of qubits used in the circuit
    :param num_cricparams: Number of parameters used in the circuit
    :param repetitions: number of repetions of the different layers
    :param num_shots: number of shots for the simulator
    :param num_params: number of samples done for the expressibility
    :return: Plot of the expressibility compered to Haar random states
    """

    def P_harr(l, u, N):
        return (1 - l) ** (N - 1) - (1 - u) ** (N - 1)

    def build_pqc(repations: int, draw_circuit: bool = False):

        theta = []
        for y in range(num_cricparams):
            theta.append(2 * np.pi * random())
        final = 0

        if final_layer:
            final = 1
        if alternating_layer:
            repations = (repations + 1) * len(entangle_map)
        else:
            repations = repations + 1

        cr = ClassicalRegister(Num_qubits)
        qr = QuantumRegister(Num_qubits)
        qc = QuantumCircuit(qr, cr)
        counter = 0
        if not alternating_layer:
            for reps in range(repations + final):
                barr = 0
                for instruction in instructions:  # building rotation gates
                    if instruction == "rz":
                        for bits in range(Num_qubits):
                            qc.rz(theta[counter], qr[bits])
                            counter += 1

                    elif instruction == "h":
                        for bits in range(Num_qubits):
                            qc.h(qr[bits])
                    elif instruction == "ry":
                        for bits in range(Num_qubits):
                            qc.ry(theta[counter], qr[bits])
                            counter += 1
                    elif instruction == "rx":
                        for bits in range(Num_qubits):
                            qc.rx(theta[counter], qr[bits])
                            counter += 1
                    elif instruction == "crz":
                        for bits in range(0, Num_qubits, 2):
                            qc.crz(theta[counter], control_qubit=qr[bits], target_qubit=qr[bits + 1])
                            counter += 1
                    else:
                        raise ValueError("Unkown Rotation Gate")

                    barr += 1
                    if barr == len(instructions):
                        qc.barrier()

                if reps != repations:
                    if entanglement_block == "crz":

                        for ent in range(len(entangle_map)):
                            qc.crz(theta[counter], qr[entangle_map[ent][0]], qr[entangle_map[ent][1]])
                            counter += 1
                        qc.barrier()
                    elif entanglement_block == "crx":

                        for ent in range(len(entangle_map)):
                            qc.crx(theta[counter], qr[entangle_map[ent][0]], qr[entangle_map[ent][1]])
                            counter += 1
                        qc.barrier()
                    elif entanglement_block == "cry":

                        for ent in range(len(entangle_map)):
                            qc.cry(theta[counter], qr[entangle_map[ent][0]], qr[entangle_map[ent][1]])
                            counter += 1
                        qc.barrier()
                    elif entanglement_block == "cz":

                        for ent in range(len(entangle_map)):
                            qc.cz(qr[entangle_map[ent][0]], qr[entangle_map[ent][1]])

                        qc.barrier()
                    elif entanglement_block == "cx":

                        for ent in range(len(entangle_map)):
                            qc.cx(qr[entangle_map[ent][0]], qr[entangle_map[ent][1]])

                        qc.barrier()
                    elif entanglement_block == "cy":

                        for ent in range(len(entangle_map)):
                            qc.cy(qr[entangle_map[ent][0]], qr[entangle_map[ent][1]])

                        qc.barrier()
                    elif entanglement_block == 0:
                        pass
                    else:
                        raise ValueError("Unkown entanglement Gate")

        elif alternating_layer:
            alternation = 0
            for reps in range(repations + final):
                barr = 0
                for instruction in instructions:  # building rotation gates
                    if instruction == "rz":
                        for bits in range(Num_qubits):
                            qc.rz(theta[counter], qr[bits])
                            counter += 1

                    elif instruction == "h":
                        for bits in range(Num_qubits):
                            qc.h(qr[bits])
                    elif instruction == "ry":
                        for bits in range(Num_qubits):
                            qc.ry(theta[counter], qr[bits])
                            counter += 1
                    elif instruction == "rx":
                        for bits in range(Num_qubits):
                            qc.rx(theta[counter], qr[bits])
                            counter += 1
                    elif instruction == "crz":
                        for bits in range(0, Num_qubits, 2):
                            qc.crz(theta[counter], control_qubit=qr[bits], target_qubit=qr[bits + 1])
                            counter += 1
                    else:
                        raise ValueError("Unkown Rotation Gate")

                    barr += 1
                    if barr == len(instructions):
                        qc.barrier()
                if reps != repations:  # building entanglement gates
                    if entanglement_block == "crz":
                        for ent in range(len(entangle_map[alternation])):
                            qc.crz(theta[counter], qr[entangle_map[alternation][ent][0]],
                                   qr[entangle_map[alternation][ent][1]])
                            counter += 1
                        qc.barrier()
                    elif entanglement_block == "crx":

                        for ent in range(len(entangle_map[alternation])):
                            qc.crx(theta[counter], qr[entangle_map[alternation][ent][0]],
                                   qr[entangle_map[alternation][ent][1]])
                            counter += 1
                        qc.barrier()
                    elif entanglement_block == "cry":

                        for ent in range(len(entangle_map[alternation])):
                            qc.cry(theta[counter], qr[entangle_map[alternation][ent][0]],
                                   qr[entangle_map[alternation][ent][1]])
                            counter += 1
                        qc.barrier()
                    elif entanglement_block == "cz":

                        for ent in range(len(entangle_map[alternation])):
                            qc.cz(qr[entangle_map[alternation][ent][0]], qr[entangle_map[alternation][ent][1]])

                        qc.barrier()
                    elif entanglement_block == "cx":

                        for ent in range(len(entangle_map[alternation])):
                            qc.cx(qr[entangle_map[alternation][ent][0]], qr[entangle_map[alternation][ent][1]])

                        qc.barrier()
                    elif entanglement_block == "cy":

                        for ent in range(len(entangle_map[alternation])):
                            qc.cy(qr[entangle_map[alternation][ent][0]], qr[entangle_map[alternation][ent][1]])

                        qc.barrier()
                    elif entanglement_block == 0:
                        pass
                    else:
                        raise ValueError("Unkown entanglement Gate")
                    alternation += 1
                    if alternation == len(entangle_map):
                        alternation = 0

        """for reps in range(repations + final):
            #for alternation in len((entangle_map)):# build circuit
            barr = 0
            for instruction in instructions: # building rotation gates
                if instruction == "rz":
                    for bits in range(Num_qubits):
                        qc.rz(theta[counter], qr[bits])
                        counter += 1

                elif instruction == "h":
                    for bits in range(Num_qubits):
                        qc.h(qr[bits])
                elif instruction == "ry":
                    for bits in range(Num_qubits):
                        qc.ry(theta[counter], qr[bits])
                        counter += 1
                elif instruction == "rx":
                    for bits in range(Num_qubits):
                        qc.rx(theta[counter], qr[bits])
                        counter += 1
                elif instruction == "crz":
                    for bits in range(0, Num_qubits, 2):
                        qc.crz(theta[counter], control_qubit=qr[bits], target_qubit=qr[bits + 1])
                        counter += 1
                else:
                    raise ValueError("Unkown Rotation Gate")

                barr += 1
                if barr == len(instructions):
                    qc.barrier()
            if reps != repations: # building entanglement gates
                if alternating_layer == False:
                    if entanglement_block == "crz":

                        for ent in range(len(entangle_map)):
                            qc.crz(theta[counter], qr[entangle_map[ent][0]], qr[entangle_map[ent][1]])
                            counter += 1
                        qc.barrier()
                    elif entanglement_block == "crx":

                        for ent in range(len(entangle_map)):
                            qc.crx(theta[counter], qr[entangle_map[ent][0]], qr[entangle_map[ent][1]])
                            counter += 1
                        qc.barrier()
                    elif entanglement_block == "cry":

                        for ent in range(len(entangle_map)):
                            qc.cry(theta[counter], qr[entangle_map[ent][0]], qr[entangle_map[ent][1]])
                            counter += 1
                        qc.barrier()
                    elif entanglement_block == "cz":

                        for ent in range(len(entangle_map)):
                            qc.cz( qr[entangle_map[ent][0]], qr[entangle_map[ent][1]])

                        qc.barrier()
                    elif entanglement_block == "cx":

                        for ent in range(len(entangle_map)):
                            qc.cx(qr[entangle_map[ent][0]], qr[entangle_map[ent][1]])

                        qc.barrier()
                    elif entanglement_block == "cy":

                        for ent in range(len(entangle_map)):
                            qc.cy( qr[entangle_map[ent][0]], qr[entangle_map[ent][1]])

                        qc.barrier()
                    elif entanglement_block == 0:
                        pass
                    else:
                        raise ValueError("Unkown entanglement Gate")
                elif alternating_layer == True:

                        if entanglement_block == "crz":
                            print(reps)
                            print(len(entangle_map[reps]))
                            for ent in range(len(entangle_map[reps])):
                                qc.crz(theta[counter], qr[entangle_map[reps][ent][0]], qr[entangle_map[reps][ent][1]])
                                counter += 1
                            qc.barrier()
                        elif entanglement_block == "crx":

                            for ent in range(len(entangle_map[reps])):
                                qc.crx(theta[counter], qr[entangle_map[reps][ent][0]], qr[entangle_map[reps][ent][1]])
                                counter += 1
                            qc.barrier()
                        elif entanglement_block == "cry":

                            for ent in range(len(entangle_map[reps])):
                                qc.cry(theta[counter], qr[entangle_map[reps][ent][0]], qr[entangle_map[reps][ent][1]])
                                counter += 1
                            qc.barrier()
                        elif entanglement_block == "cz":

                            for ent in range(len(entangle_map[reps])):
                                qc.cz(qr[entangle_map[reps][ent][0]], qr[entangle_map[reps][ent][1]])

                            qc.barrier()
                        elif entanglement_block == "cx":

                            for ent in range(len(entangle_map[reps])):
                                qc.cx(qr[entangle_map[reps][ent][0]], qr[entangle_map[reps][ent][1]])

                            qc.barrier()
                        elif entanglement_block == "cy":

                            for ent in range(len(entangle_map[reps])):
                                qc.cy(qr[entangle_map[reps][ent][0]], qr[entangle_map[reps][ent][1]])

                            qc.barrier()
                        elif entanglement_block == 0:
                            pass
                        else:
                            raise ValueError("Unkown entanglement Gate")
"""
        if draw_circuit == True:
            print(qc.decompose().draw())

        return qc, theta

    def build_other(repetitions: int, draw_circuit: bool = False):
        if name == "Circuit_9":
            theta = []
            instructions = []
            for j in reversed(range(1, Num_qubits)):
                instructions.append((j, j - 1))

            for y in range(num_cricparams):
                theta.append(2 * np.pi * random())

            qr = QuantumRegister(Num_qubits)
            cr = ClassicalRegister(Num_qubits)
            qc = QuantumCircuit(qr, cr)
            counter = 0
            for layer in range(repetitions + 1):
                for a in range(Num_qubits):
                    qc.h(a)
                for instruct in instructions:
                    qc.cz(instruct[0], instruct[1])
                for b in range(Num_qubits):
                    qc.rx(theta[counter], b)
                    counter += 1

        if name == "Circuit_11":

            instructions = []
            theta = []

            i = 0
            j = Num_qubits - 1

            # construct instructions for entanglement
            for lays in range(int(np.floor(Num_qubits / 2))):

                single_instructions = []
                if i + 1 != j:
                    single_instructions.append((i + 1, i))
                    single_instructions.append((j, j - 1))
                else:
                    single_instructions.append((i, i + 1))
                instructions.append(single_instructions)
                i += 1
                j -= 1

            for t in range(num_cricparams):
                theta.append(2 * np.pi * random())

            qr = QuantumRegister(Num_qubits)
            cr = ClassicalRegister(Num_qubits)
            qc = QuantumCircuit(qr, cr)
            counter = 0
            for layer in range(repetitions + 1):
                for w in range(len(instructions)):

                    for a in range(w, Num_qubits - w):
                        qc.ry(theta[counter], a)
                        counter += 1

                    for b in range(w, Num_qubits - w):
                        qc.rz(theta[counter], b)
                        counter += 1

                    for instruct in instructions[w]:
                        qc.cx(instruct[0], instruct[1])

        if name == "Circuit_12":

            instructions = []
            theta = []

            i = 0
            j = Num_qubits - 1

            # construct instructions for entanglement
            for lays in range(int(np.floor(Num_qubits / 2))):

                single_instructions = []
                if i + 1 != j:
                    single_instructions.append((i + 1, i))
                    single_instructions.append((j, j - 1))
                else:
                    single_instructions.append((i, i + 1))
                instructions.append(single_instructions)
                i += 1
                j -= 1

            for t in range(num_cricparams):
                theta.append(2 * np.pi * random())

            qr = QuantumRegister(Num_qubits)
            cr = ClassicalRegister(Num_qubits)
            qc = QuantumCircuit(qr, cr)
            counter = 0
            for layer in range(repetitions + 1):
                for w in range(len(instructions)):

                    for a in range(w, Num_qubits - w):
                        qc.ry(theta[counter], a)
                        counter += 1

                    for b in range(w, Num_qubits - w):
                        qc.rz(theta[counter], b)
                        counter += 1

                    for instruct in instructions[w]:
                        qc.cz(instruct[0], instruct[1])

        if draw_circuit == True:
            print(qc.decompose().draw())

        return qc, theta

    # Possible Bin

    Num_qubits = 4
    bins_list = []
    for bin_index in range(76):
        bins_list.append(bin_index / 75)

    # Center of the bin
    bins_x = []
    for bin_pos in range(75):
        bins_x.append(bins_list[1] + bins_list[bin_pos])

    # Harr histogram
    P_harr_hist = []

    for i in range(75):
        P_harr_hist.append(P_harr(bins_list[i], bins_list[i + 1], 2 ** Num_qubits))

    backend = Aer.get_backend('qasm_simulator')

    layer_express = []
    for layers in range(repetitions):

        fidelity = []
        for params in progressbar.progressbar(range(num_params)):

            if name != "Circuit_9" and name != "Circuit_11" and name != "Circuit_12":
                qc_expres, theta = build_pqc(layers, draw_circuit=False)
            else:
                qc_expres, theta = build_other(layers, draw_circuit=False)

            measure_map = []

            for i in range(Num_qubits):
                measure_map.append(i)

            qc_expres.measure(measure_map, measure_map)

            job = execute(qc_expres, backend, shots=num_shots)
            result = job.result()
            count = result.get_counts()

            zero = ""
            for zeros in range(Num_qubits):
                zero += "0"

            if '0000' in count and '1' in count:
                ratio = count['0000'] / num_shots
            elif '0000' in count and '1' not in count:
                ratio = count['0000'] / num_shots
            else:
                ratio = 0
            fidelity.append(ratio)

        weights = np.ones_like(fidelity) / float(len(fidelity))
        if detailed:
            plt.figure()
            plt.hist(fidelity, bins=bins_list, weights=weights, range=[0, 1], label='Ansatz')
            plt.plot(bins_x, P_harr_hist, label='Harr')
            plt.legend(loc='upper right')
            plt.title(name)
            plt.show()

        P_1_hist = np.histogram(fidelity, bins=bins_list, weights=weights, range=[0, 1])[0]
        kl_pq = rel_entr(P_1_hist, P_harr_hist)
        print(bcolors.OKGREEN + "KL(P || Q): %.3f nats" % sum(kl_pq) + bcolors.ENDC)
        layer_express.append(sum(kl_pq))

    print(bcolors.OKBLUE + "Expressibilities for layers of "+ str(name) + ": " + str(layer_express) + bcolors.ENDC)


    plt.figure()
    plt.plot(layer_express, marker="o")
    plt.title(name)
    plt.savefig(os.path.join(directory, "layer_expressebility_circuit_" + str(name) + ".pdf"))
    if FLAG_show_saved:
        plt.show()

    return layer_express


def compare_layer_express(circ_list: list, num_qubits: int, repetition: int, num_shots: int, num_params: int,
                          directory: str, detailed, ordered, FLAG_show_saved):
    res_list = []
    directory = os.path.join(directory,"expressibility_results")
    if not os.path.exists(directory):
        print(bcolors.WARNING + directory + " not found. Creating corresponding directory" + bcolors.ENDC)
        os.mkdir(directory)
    for circ_e in circ_list:
        print(bcolors.OKGREEN + "Start of Circuit " + str(circ_e) + bcolors.ENDC)
        name_e = "Circuit_" + str(circ_e)
        ansatz_e, instruction_e, entanglement_block_e, final_layer_e, alternating_layer_e = circuit(circ_e, num_qubits,
                                                                                                    repetition,
                                                                                                    layering=False,
                                                                                                    print_circuit=False)

        try:
            entangler_map_e = ansatz_e[-1].entanglement
        except AttributeError:
            entangler_map_e = ""

        aux = expressability(instructions=instruction_e, entanglement_block=entanglement_block_e,
                             entangle_map=entangler_map_e, name=name_e, directory=directory, repetitions=repetition,
                             Num_qubits=num_qubits, num_shots=num_shots, num_params=num_params,
                             num_cricparams=ansatz_e[-1].num_parameters, final_layer=final_layer_e,
                             alternating_layer=alternating_layer_e, detailed=detailed, FLAG_show_saved=FLAG_show_saved)

        res_list.append(aux)

    symbols = ["o", "v", "^", "8", "<", ">", "p", "P", "x", "+", "*"]
    colors = ["red", "blue", "black", "green", "purple", "yellow", "teal", "indigo", "olive", "forestgreen",
              "chocolate"]

    plt.figure()
    for j in range(repetition):
        x = [i for i in range(len(circ_list))]
        y = [i[j] for i in res_list]
        plt.plot(x, y, symbols[j], color=colors[j], label='L='+str(j+1))

    plt.legend(loc="upper right")
    plt.title("Comparison of different circuit Expressibilities")
    plt.yscale('log', base=10)
    plt.xlabel('Circuit ID')
    plt.ylabel('Expressibility')
    plt.xticks([i for i in range(len(circ_list))], circ_list)
    plt.savefig(os.path.join(directory, "layer_expressebility_comparison.pdf"))
    if FLAG_show_saved:
        plt.show()
    else:
        plt.close()

    if ordered:

        x = [str(i + 1) for i in range(19)]
        x_ticks_labels_gen = ['9', '1', '2', '16', '3', '18', '10', '12', '15', '17', '4', '11', '7', '8', '19', '5',
                              '13', '14', '6']
        x_ticks_labels = [j for j in x_ticks_labels_gen if int(j) in circ_list]
        xarr = np.array(x)
        ind = np.where(xarr.reshape(xarr.size, 1) == np.array(x_ticks_labels))[1]

        fig, ax = plt.subplots(1, 1)
        for j in range(repetition):
            y = [i[j] for i in res_list]
            ax.scatter(ind, y, marker=symbols[j], color=colors[j], label='L=' + str(j + 1))

        ax.set_yscale('log', base=10)
        ax.set_xlabel('Circuit ID')
        ax.set_ylabel('Expressibility')
        ax.legend(loc='upper right')
        plt.title("Comparison of different circuit Expressibilities ordered")
        ax.set_xticks(range(len(x_ticks_labels)))
        ax.set_xticklabels(x_ticks_labels)
        plt.savefig(os.path.join(directory, "layer_expressebility_comparison_ordered.pdf"))
        if FLAG_show_saved:
            plt.show()
        else:
            plt.close()

    return None
