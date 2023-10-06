import numpy as np
import os, time
import matplotlib.pyplot as plt
from random import random

from qiskit.circuit import ParameterVector
from qiskit.opflow.state_fns.circuit_state_fn import CircuitStateFn
from qiskit import (QuantumCircuit, execute, Aer)
from qiskit import QuantumRegister
from qiskit.compiler import transpile

from orqviz.loss_function import LossFunctionWrapper
from orqviz.scans import perform_1D_interpolation, plot_1D_interpolation_result, perform_2D_interpolation, \
    plot_2D_scan_result, perform_2D_scan
from orqviz.pca import get_pca, perform_2D_pca_scan, plot_pca_landscape, plot_optimization_trajectory_on_pca
from orqviz.geometric import get_random_orthonormal_vector, get_random_normal_vector


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


def trainability(instructions: list, entanglement_block: str, entangle_map: list, Num_qubits: int, num_cricparams: int,
                 repetition: int, name: str, directory, Ham, final_parameters, parameter_trajectory,
                 final_layer: bool = False, alternating_layer: bool = False, FLAG_show_saved: bool = False,
                 Flag_verbose: bool = False):
    """

    :param directory:
    :param Flag_verbose:
    :param FLAG_show_saved:
    :param instructions:
    :param entanglement_block:
    :param entangle_map:
    :param Num_qubits:
    :param num_cricparams:
    :param repetition:
    :param name:
    :param Ham:
    :param final_parameters:
    :param parameter_trajectory:
    :param final_layer:
    :param alternating_layer:
    :return:
    """

    def build_pqc(param_vector, repetition: int = repetition, draw_circuit: bool = False):

        theta = []
        final = 0
        for i in range(num_cricparams):
            theta.append(2 * np.pi * random())
        num_ent_layer = 0

        if final_layer:
            final = 1
        if alternating_layer:
            repetition = (repetition) * len(entangle_map)

        qr = QuantumRegister(Num_qubits)
        qc = QuantumCircuit(qr)
        counter = 0

        if not alternating_layer:
            for reps in range(repetition + final):
                # build circuit
                barr = 0
                for instruction in instructions:
                    if instruction == "rz":
                        for bits in range(Num_qubits):
                            qc.rz(param_vector[counter], qr[bits])
                            counter += 1
                    elif instruction == "h":
                        for bits in range(Num_qubits):
                            qc.h(qr[bits])
                    elif instruction == "ry":
                        for bits in range(Num_qubits):
                            qc.ry(param_vector[counter], qr[bits])
                            counter += 1
                    elif instruction == "rx":
                        for bits in range(Num_qubits):
                            qc.rx(param_vector[counter], qr[bits])
                            counter += 1

                    elif instruction == "crz":
                        for bits in range(0, Num_qubits, 2):
                            qc.crz(param_vector[counter], control_qubit=qr[bits], target_qubit=qr[bits + 1])
                            counter += 1
                    else:

                        raise ValueError("Unkown Rotation Gate")

                    barr += 1
                    if barr == len(instructions):
                        qc.barrier()

                if reps != repetition:
                    if entanglement_block == "crz":

                        for ent in range(len(entangle_map)):
                            qc.crz(param_vector[counter], qr[entangle_map[ent][0]], qr[entangle_map[ent][1]])
                            counter += 1
                        qc.barrier()

                    elif entanglement_block == "crx":

                        for ent in range(len(entangle_map)):
                            qc.crx(param_vector[counter], qr[entangle_map[ent][0]], qr[entangle_map[ent][1]])
                            counter += 1
                        qc.barrier()

                    elif entanglement_block == "cry":

                        for ent in range(len(entangle_map)):
                            qc.cry(param_vector[counter], qr[entangle_map[ent][0]], qr[entangle_map[ent][1]])
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
                        print(entanglement_block)
                        raise ValueError("Unkown entanglement gate")
        elif alternating_layer:

            alternation = 0
            for reps in range(repetition + final):
                barr = 0
                for instruction in instructions:  # building rotation gates
                    if instruction == "rz":
                        for bits in range(Num_qubits):
                            qc.rz(param_vector[counter], qr[bits])
                            counter += 1

                    elif instruction == "h":
                        for bits in range(Num_qubits):
                            qc.h(qr[bits])
                    elif instruction == "ry":
                        for bits in range(Num_qubits):
                            qc.ry(param_vector[counter], qr[bits])
                            counter += 1
                    elif instruction == "rx":
                        for bits in range(Num_qubits):
                            qc.rx(param_vector[counter], qr[bits])
                            counter += 1
                    elif instruction == "crz":
                        for bits in range(0, Num_qubits, 2):
                            qc.crz(param_vector[counter], control_qubit=qr[bits], target_qubit=qr[bits + 1])
                            counter += 1
                    else:
                        raise ValueError("Unkown Rotation Gate")

                    barr += 1
                    if barr == len(instructions):
                        qc.barrier()
                if reps != repetition:  # building entanglement gates
                    if entanglement_block == "crz":
                        for ent in range(len(entangle_map[alternation])):
                            qc.crz(param_vector[counter], qr[entangle_map[alternation][ent][0]],
                                   qr[entangle_map[alternation][ent][1]])
                            counter += 1
                        qc.barrier()
                    elif entanglement_block == "crx":

                        for ent in range(len(entangle_map[alternation])):
                            qc.crx(param_vector[counter], qr[entangle_map[alternation][ent][0]],
                                   qr[entangle_map[alternation][ent][1]])
                            counter += 1
                        qc.barrier()
                    elif entanglement_block == "cry":

                        for ent in range(len(entangle_map[alternation])):
                            qc.cry(param_vector[counter], qr[entangle_map[alternation][ent][0]],
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

        if draw_circuit:
            print(qc.decompose().draw())

        return transpile(qc), theta

    def other_pqc(param_vector, repetition: int = repetition, draw_circuit: bool = False):
        if name == "Circuit_9":
            theta = []
            instructions = []
            for j in reversed(range(1, Num_qubits)):
                instructions.append((j, j - 1))

            for y in range(num_cricparams):
                theta.append(2 * np.pi * random())

            qr = QuantumRegister(Num_qubits)
            qc = QuantumCircuit(qr)
            counter = 0
            for layer in range(repetition):
                for a in range(Num_qubits):
                    qc.h(a)
                for instruct in instructions:
                    qc.cz(instruct[0], instruct[1])
                for b in range(Num_qubits):
                    qc.rx(param_vector[counter], b)
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
            qc = QuantumCircuit(qr)
            counter = 0
            for layer in range(repetition):
                for w in range(len(instructions)):

                    for a in range(w, Num_qubits - w):
                        qc.ry(param_vector[counter], a)
                        counter += 1

                    for b in range(w, Num_qubits - w):
                        qc.rz(param_vector[counter], b)
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
            qc = QuantumCircuit(qr)
            counter = 0
            for layer in range(repetition):
                for w in range(len(instructions)):

                    for a in range(w, Num_qubits - w):
                        qc.ry(param_vector[counter], a)
                        counter += 1

                    for b in range(w, Num_qubits - w):
                        qc.rz(param_vector[counter], b)
                        counter += 1

                    for instruct in instructions[w]:
                        qc.cz(instruct[0], instruct[1])

        if draw_circuit:
            print(qc.decompose().draw())

        return transpile(qc), theta

    def calculate_energy(parameters, transpiled_circuit, param_vector):
        """
        Function that receives parameters for a quantum circuit, as well as specifications for the circuit depth,
        and returns the energy over a previously defined Hamiltonian H.
        """
        # bind the parameter values for this calculation to the transpiled circuit
        final_circuit = transpiled_circuit.bind_parameters({param_vector: parameters})

        # convert to a state (do this only for small systems)
        psi = CircuitStateFn(final_circuit)
        # print(psi.to_matrix())
        # print("asdfghjklö")
        # print((Ham.to_matrix()))
        return np.dot(psi.adjoint().to_matrix(), np.dot(Ham.to_matrix(), psi.to_matrix())).real

    param_vector = ParameterVector('param_vector', num_cricparams)

    if name == "Circuit_9" or name == "Circuit_11" or name == "Circuit_12":
        qc, initial_parameters = other_pqc(param_vector)
    else:
        qc, initial_parameters = build_pqc(param_vector)

    loss_function = LossFunctionWrapper(calculate_energy, transpiled_circuit=qc,
                                        param_vector=param_vector)

    if Flag_verbose:
        print(bcolors.OKBLUE + "Start of 1D loss landscape scan" + bcolors.ENDC)
    interpolation_result = perform_1D_interpolation(initial_parameters, final_parameters,
                                                    loss_function, end_points=(-1, 1), verbose=Flag_verbose)

    if Flag_verbose:
        print(bcolors.OKBLUE + "End of 1D loss landscape scan" + bcolors.ENDC)

    plt.figure()
    plot_1D_interpolation_result(interpolation_result, label="linear interpolation", color="gray")
    plt.title(name)
    plt.legend()
    time.sleep(1)
    plt.savefig(os.path.join(directory, "1D_scan_circuit_" + str(name) + ".pdf"))
    if FLAG_show_saved:
        plt.show()

    dir1 = get_random_normal_vector(num_cricparams)
    dir2 = get_random_orthonormal_vector(dir1)
    if Flag_verbose:
        print(bcolors.OKBLUE + "Start of 2D loss landscape scan" + bcolors.ENDC)

    scan_2D_result = perform_2D_scan(final_parameters, loss_function,
                                     direction_x=dir1, direction_y=dir2, n_steps_x=40, end_points_x=(-25, 25),
                                     end_points_y=(-25, 25), verbose=Flag_verbose)

    if Flag_verbose:
        print(bcolors.OKBLUE + "End of 2D loss landscape scan" + bcolors.ENDC)

    plt.figure()
    plot_2D_scan_result(scan_2D_result)
    plt.title(name)
    time.sleep(1)
    plt.savefig(os.path.join(directory, "2D_scan_circuit_" + str(name) + ".pdf"))
    if FLAG_show_saved:
        plt.show()

    pca = get_pca(parameter_trajectory)
    if Flag_verbose:
        print(bcolors.OKBLUE + "Start of optimization trajectory exploration" + bcolors.ENDC)
    scan_pca_result = perform_2D_pca_scan(pca, loss_function, n_steps_x=40, offset=15, verbose=Flag_verbose)
    if Flag_verbose:
        print(bcolors.OKBLUE + "End of optimization trajectory exploration" + bcolors.ENDC)
    fig, ax = plt.subplots()
    plot_pca_landscape(scan_pca_result, pca, fig=fig, ax=ax)
    plot_optimization_trajectory_on_pca(parameter_trajectory, pca, ax=ax,
                                        label="Optimization Trajectory", color="lightsteelblue")
    plt.legend()
    plt.title(name)
    time.sleep(1)
    plt.savefig(os.path.join(directory, "trajectory_circuit_" + str(name) + ".pdf"))
    if FLAG_show_saved:
        plt.show()
