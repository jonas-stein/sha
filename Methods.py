import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import Union, Dict, List

from Circuits import circuit
from build_Hamiltonian import build_Hamiltonian
from Graph_operations import check_coloring
from trainability import trainability
from SHA import create_SHA_layers

from qiskit import (Aer)
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.algorithms.optimizers import SLSQP, COBYLA
from qiskit.algorithms import VQE, QAOA
from qiskit.circuit import ParameterVector
from qiskit.opflow import StateFn, CircuitSampler
from qiskit.quantum_info import Statevector
from qiskit.result import QuasiDistribution
from qiskit.utils import QuantumInstance
from qiskit.utils.backend_utils import is_aer_provider
from qiskit.opflow import PauliSumOp

global number_solutions
global number_correct_solutions


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


class Methods:
    def __init__(self, A, circuit_type, repetitions, colors, directory, graph_counter, num_qubits, optimizer, max_iter,
                 verbose, FLAG_show_saved, FLAG_trainability, pretraining_precision, final_precision, SHA_layers,
                 num_shots, stat):
        """

        :param A: Adjacency matrix of graph coloring problem
        :param circuit_type: Circuit ID
        :param repetitions: number of repetitions of the base circuit layer
        :param colors: number of max allowed colors in graph coloring problem
        :param directory: directory where to results are going to be saved
        :param graph_counter: index to track the number of graphs
        :param num_qubits: number of qubits for the PQC
        :param optimizer: optimizer used for the variational algorithm
        :param max_iter: maximum number iteration allowed in a single optimization process
        :param verbose: show additional print_outs
        :param FLAG_show_saved: shows printed out results that are otherwise saved
        :param FLAG_trainability: turns on the trainability exploration of the cost landscape
        :param pretraining_precision: precision of the pre-training layers
        :param final_precision: precision of the final optimization layer
        :param SHA_layers: number of times the Hamiltonian gets partitioned
        :param num_shots: number of shots used for the shot based Quantum simulation
        """
        self.A = A
        self.circuit_type = circuit_type
        self.repetitions = repetitions
        self.SHA_layers = SHA_layers
        self.colors = colors
        self.directory = directory
        self.graph_counter = graph_counter
        self.num_qubits = num_qubits
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.num_shots = num_shots
        self.pretraining_precision = pretraining_precision
        self.final_precision = final_precision
        self.verbose = verbose
        self.FLAG_show_saved = FLAG_show_saved
        self.FLAG_trainability = FLAG_trainability
        self.stat = stat

    @staticmethod
    def sample_most_likely(state_vector):
        """Compute the most likely binary string from state vector.
        Args:
            state_vector (numpy.ndarray or dict): state vector or counts.
        Returns:
            numpy.ndarray: binary string as numpy.ndarray of ints.
        """
        if isinstance(state_vector, QuasiDistribution):
            probabilities = state_vector.binary_probabilities()
            binary_string = max(probabilities.items(), key=lambda kv: kv[1])[0]
            x = np.asarray([int(y) for y in reversed(list(binary_string))])
            return x
        elif isinstance(state_vector, Statevector):
            probabilities = state_vector.probabilities()
            n = state_vector.num_qubits
            k = np.argmax(np.abs(probabilities))
            x = np.zeros(n)
            for i in range(n):
                x[i] = k % 2
                k >>= 1
            return x
        elif isinstance(state_vector, (OrderedDict, dict)):
            # get the binary string with the largest count
            binary_string = max(state_vector.items(), key=lambda kv: kv[1])[0]
            x = np.asarray([int(y) for y in (list(binary_string))])
            return x
        elif isinstance(state_vector, StateFn):
            binary_string = list(state_vector.sample().keys())[0]
            x = np.asarray([int(y) for y in reversed(list(binary_string))])
            return x
        elif isinstance(state_vector, np.ndarray):
            n = int(np.log2(state_vector.shape[0]))
            k = np.argmax(np.abs(state_vector))
            x = np.zeros(n)
            for i in range(n):
                x[i] = k % 2
                k >>= 1
            return x
        else:
            raise ValueError(
                "state vector should be QuasiDistribution, Statevector, ndarray, or dict. "
                f"But it is {type(state_vector)}."
            )

    @staticmethod
    def accuracy_per_shot(A, state_vector, directory, verbose, FLAG_show_saved, num_shots):
        acc_counter = 0
        shots = sorted(state_vector.items(), key=lambda kv: kv[1])
        for shot in shots:
            x = np.asarray([int(y) for y in list(shot[0])])
            check = check_coloring(A, x, print_outs=verbose, directory=directory, show=FLAG_show_saved)
            if check:
                acc_counter += shot[1] * np.sqrt(num_shots)
        if verbose:
            print(bcolors.OKGREEN + f"The accuracy over all {num_shots} number of shots is {acc_counter / num_shots}")

        return acc_counter / num_shots

    @staticmethod
    def result_plots(directory, name, layers, SHA_layers, ratio_solutions, values, counts, FLAG_show_saved):

        plt.figure()
        plt.plot(counts, values)
        plt.xlabel("Evaluation counts")
        plt.ylabel("Loss")
        plt.title(name)
        plt.savefig(os.path.join(directory, "loss_of_circuit_" + str(name) + "of_layer" + str(layers) + "of_SHA" +
                                 str(SHA_layers) + ".pdf"))
        if FLAG_show_saved:
            plt.show()
        else:
            plt.close()

        plt.figure()
        plt.plot(ratio_solutions)
        plt.xlabel("Iterations of Optimizer")
        plt.ylabel("Ratio of correct solutions")
        plt.title(name)
        plt.savefig(os.path.join(directory, "ratio_of_correct_solutions_of" + str(name) + "of_layer" + str(layers) +
                                 "_of_SHA" + str(SHA_layers) + ".pdf"))
        if FLAG_show_saved:
            plt.show()
        else:
            plt.close()

        return None

    @staticmethod
    def extract_parameters(result, ansatz, layers):
        finalparams = np.zeros(len(result.optimal_parameters))
        params = (str(result.optimal_parameters))

        params = params.replace("ParameterVectorElement(θ[0]):", "")
        params = params.replace("ParameterVectorElement(β[0]):", "")
        params = params.replace("ParameterVectorElement(γ[0]):", "")
        params = params.replace("{", "").replace("}", "")

        for values in range(len(result.optimal_parameters)):

            params = params.replace(" ParameterVectorElement(θ[" + str(values) + "]): ", "")
            params = params.replace(" ParameterVectorElement(β[" + str(values) + "]): ", "")
            params = params.replace(" ParameterVectorElement(γ[" + str(values) + "]): ", "")
            secondpos = params.find(",")

            if secondpos is not -1:

                finalparams[values] = (float(params[:secondpos]))
                params = params.replace(params[:secondpos + 1], "", 1)
            else:
                finalparams[values] = (float(params[:]))

        point_initial = list(finalparams)
        adds = 0
        try:
            adds = ansatz[layers + 1].num_parameters - ansatz[layers].num_parameters
        except IndexError:
            pass

        for additions in range(adds):
            point_initial.extend([1e-06])

        return point_initial, finalparams


    def execution(self, ansatz, layers, child_directory, name, Ham, SHA_layers, instruction, final_layer,
                  entanglement_block, entangler_map, alternating_layer, precision, QAOA_exe=False, initial_point=None,
                  LL_layer=None):
        counts = []
        values = []
        intermediate_params = []
        est_error = []
        if LL_layer is None:
            LL_layer = layers

        def build_optimal_qaoa_circuit(Hamiltonian: PauliSumOp, reps, optimal_parameters):

            num_qubits = Hamiltonian.num_qubits
            qr = QuantumRegister(num_qubits)
            circuit = QuantumCircuit(qr, )

            gamma = [optimal_parameters[2 * i] for i in range(reps)]
            beta = [optimal_parameters[2 * i + 1] for i in range(reps)]

            circuit.h(qr)

            for i in range(reps):

                for term in Hamiltonian:

                    pauli_ops = term.primitive.to_list()
                    positions = [i for i in range(len(pauli_ops[0][0])) if pauli_ops[0][0].startswith("Z", i)]
                    strategy = []
                    for j in range(len(positions) - 1):
                        strategy.append([positions[j], positions[j + 1]])

                    for connection in strategy:
                        circuit.cz(control_qubit=connection[0], target_qubit=connection[1])

                    circuit.rz(gamma[i], strategy[-1][1])

                    for connection in reversed(strategy):
                        circuit.cz(control_qubit=connection[1], target_qubit=connection[0])

                for mixer_q in range(num_qubits):
                    circuit.rz(beta[i], mixer_q)

            return circuit

        def store_intermediate_result(eval_count, parameters, mean, std):

            r = open(os.path.join(self.directory, "recovery_file.csv"), "a")
            svqe_writer = csv.writer(r)
            counts.append(eval_count)
            values.append(mean)
            intermediate_params.append(parameters)
            est_error.append(std)

            svqe_writer.writerow([counts[-1]])
            svqe_writer.writerow([values[-1]])
            svqe_writer.writerow([intermediate_params[-1]])
            r.close()

        def cb(xk):
            global number_solutions
            global number_correct_solutions
            global ratio_solutions
            number_solutions += 1

            def _get_eigenstate(optimal_parameters) -> Union[List[float], Dict[str, int]]:
                """Get the simulation outcome of the ansatz, provided with parameters."""
                _circuit_sampler = CircuitSampler(qi, param_qobj=is_aer_provider(qi.backend))
                if QAOA_exe:
                    optimal_circuit = build_optimal_qaoa_circuit(Ham, self.repetitions, optimal_parameters)
                else:
                    optimal_circuit = ansatz[layers].bind_parameters(optimal_parameters)
                state_fn = _circuit_sampler.convert(StateFn(optimal_circuit)).eval()

                if qi.is_statevector:
                    state = state_fn.primitive.data  # VectorStateFn -> State vector -> np.array
                else:
                    state = state_fn.to_dict_fn().primitive  # SparseVectorStateFn -> DictStateFn -> dict

                return state

            check = check_coloring(A=self.A, x=self.sample_most_likely(_get_eigenstate(xk)), directory=self.directory,
                                   show=self.FLAG_show_saved, print_outs=self.verbose)

            if check:
                number_correct_solutions += 1
            ratio_solutions.append(number_correct_solutions / number_solutions)

            return print(bcolors.OKCYAN + "The ratio of correct solutions is {}".
                         format(number_correct_solutions / number_solutions) + bcolors.ENDC)

        qi = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=self.num_shots)

        global number_solutions
        global number_correct_solutions
        global ratio_solutions
        number_solutions = 0
        number_correct_solutions = 0
        ratio_solutions = []

        optimizer = None
        if self.optimizer == "COBYLA":
            optimizer = COBYLA(maxiter=4000, disp=self.verbose, tol=precision, callback=cb)
            print(bcolors.WARNING + "Callback is for python 3.7 not supported. Some functionalities may be lost"
                  + bcolors.ENDC)
        elif self.optimizer == "SLSQP":
            optimizer = SLSQP(maxiter=1000, disp=self.verbose, ftol=precision, callback=cb)  # stuff to change !!!!

        if QAOA_exe:
            exe = QAOA(optimizer=optimizer, quantum_instance=qi, callback=store_intermediate_result, reps=self.repetitions)
        else:
            exe = VQE(ansatz[layers], optimizer=optimizer,
                      quantum_instance=qi, callback=store_intermediate_result,
                      initial_point=initial_point)

        result = exe.compute_minimum_eigenvalue(Ham)

        x = self.sample_most_likely(result.eigenstate)

        if self.verbose:
            print(result.eigenstate)
            print(x)

        self.result_plots(directory=child_directory, name=name, layers=layers, SHA_layers=SHA_layers,
                          ratio_solutions=ratio_solutions, values=values, counts=counts,
                          FLAG_show_saved=self.FLAG_show_saved)

        acc_per_shot = self.accuracy_per_shot(A=self.A, state_vector=result.eigenstate, directory=child_directory,
                                              verbose=self.verbose, FLAG_show_saved=self.FLAG_show_saved,
                                              num_shots=self.num_shots)

        accc = open(os.path.join(child_directory, "acc_pershot.csv"), "w")
        writer = csv.writer(accc)
        writer.writerow([acc_per_shot])

        sr = open(os.path.join(child_directory, f"saved_params_{LL_layer}_{SHA_layers}.csv"), "w")
        writer = csv.writer(sr)
        writer.writerow(ratio_solutions)

        initial_point, final_parameters = self.extract_parameters(result, ansatz=ansatz, layers=layers)

        if self.FLAG_trainability:
            trainability(instructions=instruction, entanglement_block=entanglement_block, entangle_map=entangler_map,
                         repetition=self.repetitions, Num_qubits=self.num_qubits,
                         num_cricparams=ansatz[-1].num_parameters, Ham=Ham, final_parameters=final_parameters,
                         parameter_trajectory=intermediate_params, name=name, directory=child_directory,
                         final_layer=final_layer, alternating_layer=alternating_layer,
                         FLAG_show_saved=self.FLAG_show_saved, Flag_verbose=self.verbose)

        return result, final_parameters

    def SVQE(self, layers=0, layering=False, SHA_layers=0):

        Ham, terms = build_Hamiltonian(colors=self.colors, adj_m=self.A, penalty=[])

        ansatz, instruction, entanglement_block, final_layer, alternating_layer = circuit(self.circuit_type,
                                                                                          self.num_qubits,
                                                                                          self.repetitions,
                                                                                          layering=layering,
                                                                                          print_circuit=False)
        try:
            entangler_map = ansatz[-1].entanglement
        except AttributeError:
            entangler_map = ""

        name = "Circuit_" + str(self.circuit_type)

        self.directory = os.path.join(self.directory, f"{self.A.shape[0]}_nodes_{self.colors}_colors_{self.repetitions}"
                                                      f"_reps_SVQE_graph{self.graph_counter}")

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        child_directory = os.path.join(self.directory, name)
        if not os.path.exists(child_directory):
            os.mkdir(child_directory)

        result = self.execution(ansatz=ansatz, layers=layers, child_directory=child_directory, name=name, Ham=Ham,
                                SHA_layers=SHA_layers, instruction=instruction, final_layer=final_layer,
                                entanglement_block=entanglement_block, entangler_map=entangler_map,
                                alternating_layer=alternating_layer, precision=self.final_precision)

        return result

    def LVQE(self):

        SHA_layers = 0

        Ham, terms = build_Hamiltonian(colors=self.colors, adj_m=self.A, penalty=[])

        ansatz, instruction, entanglement_block, final_layer, alternating_layer = circuit(self.circuit_type,
                                                                                          self.num_qubits,
                                                                                          self.repetitions,
                                                                                          layering=True,
                                                                                          print_circuit=False)
        try:
            entangler_map = ansatz[-1].entanglement
        except AttributeError:
            entangler_map = ""

        name = "Circuit_" + str(self.circuit_type)

        self.directory = os.path.join(self.directory, f"{self.A.shape[0]}_nodes_{self.colors}_colors_{self.repetitions}"
                                                      f"_reps_LVQE_graph{self.graph_counter}")

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        child_directory = os.path.join(self.directory, name)
        if not os.path.exists(child_directory):
            os.mkdir(child_directory)

        base_number_of_parameters = ansatz[0].num_parameters
        initial_point, result = None, None

        for layer in range(self.repetitions):

            if layer == self.repetitions - 1:
                vqe_precision = self.final_precision
                print(bcolors.HEADER + "Start of final layer " + str(layer + 1) + bcolors.ENDC)
            else:
                vqe_precision = self.pretraining_precision
                print(bcolors.HEADER + "Start of layer " + str(layer + 1) + bcolors.ENDC)

            result, final_params = self.execution(ansatz=ansatz, layers=layer, child_directory=child_directory,
                                                  name=name, Ham=Ham, SHA_layers=SHA_layers, instruction=instruction,
                                                  final_layer=final_layer, entanglement_block=entanglement_block,
                                                  entangler_map=entangler_map, alternating_layer=alternating_layer,
                                                  precision=vqe_precision, initial_point=initial_point)

            initial_point = list(final_params)
            for adds in range(base_number_of_parameters):
                initial_point.extend([0])

        return result

    def SHA(self, SHA_method):

        _, terms = build_Hamiltonian(colors=self.colors, adj_m=self.A, penalty=[])

        ansatz, instruction, entanglement_block, final_layer, alternating_layer = circuit(self.circuit_type,
                                                                                          self.num_qubits,
                                                                                          self.repetitions,
                                                                                          layering=False,
                                                                                          print_circuit=False)

        Ham_layers = create_SHA_layers(self.colors, self.SHA_layers, SHA_method, terms, self.A)

        try:
            entangler_map = ansatz[-1].entanglement
        except AttributeError:
            entangler_map = ""

        name = "Circuit_" + str(self.circuit_type)

        self.directory = os.path.join(self.directory, f"{self.A.shape[0]}_nodes_{self.colors}_colors_{self.repetitions}"
                                                      f"_reps_SHA_{self.SHA_layers}_graph{self.graph_counter}")

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        child_directory = os.path.join(self.directory, name)
        if not os.path.exists(child_directory):
            os.mkdir(child_directory)

        initial_point, result = None, None

        for SHA_layer in range(self.SHA_layers):
            if SHA_layer == self.SHA_layers - 1:
                sha_precision = self.final_precision
                if self.verbose:
                    print(bcolors.HEADER + "Start of final SHA layer " + str(SHA_layer + 1) + bcolors.ENDC)
            else:
                sha_precision = self.pretraining_precision
                if self.verbose:
                    print(bcolors.HEADER + "Start of SHA layer " + str(SHA_layer + 1) + bcolors.ENDC)
            result, final_params = self.execution(ansatz=ansatz, layers=0, child_directory=child_directory, name=name,
                                                  Ham=Ham_layers[SHA_layer], SHA_layers=SHA_layer,
                                                  instruction=instruction, final_layer=final_layer,
                                                  entanglement_block=entanglement_block, entangler_map=entangler_map,
                                                  alternating_layer=alternating_layer, precision=sha_precision,
                                                  initial_point=initial_point)

            initial_point = list(final_params)

        return result

    def LL(self, p, q, r):

        assert p > 0 and q > 0 and r > 0 and type(p) == int and type(q) == int, "Hyperparamters must be greater than " \
                                                                                "0 and integers"
        assert p < self.repetitions, "Can't add more layers than there are in the circuit"
        assert r < 100 and type(r) == int, "r is a percentage and must be defined as an integer in an interval of 0 " \
                                           "to 100"

        SHA_layers = 0

        Ham, terms = build_Hamiltonian(colors=self.colors, adj_m=self.A, penalty=[])

        ansatz, instruction, entanglement_block, final_layer, alternating_layer = circuit(self.circuit_type,
                                                                                          self.num_qubits,
                                                                                          self.repetitions,
                                                                                          layering=True,
                                                                                          print_circuit=False)

        try:
            entangler_map = ansatz[-1].entanglement
        except AttributeError:
            entangler_map = ""

        name = "Circuit_" + str(self.circuit_type)

        self.directory = os.path.join(self.directory, f"{self.A.shape[0]}_nodes_{self.colors}_colors_{self.repetitions}"
                                                      f"_reps_LL_{p}_{q}_{r}_graph{self.graph_counter}")

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        child_directory = os.path.join(self.directory, name)
        if not os.path.exists(child_directory):
            os.mkdir(child_directory)

        aux_counter = self.repetitions
        p_list = []
        while aux_counter != 0:
            if aux_counter % p == 0:
                aux_counter -= p
                p_list.append(p)
            else:
                p_list.append(aux_counter)
                aux_counter = 0

        # pretraining
        layer_counter = 0
        old_frozen_parameters, frozen_parameters = [], []
        old_optimized_parameters, point_initial = None, None

        for layer in p_list:

            if layer_counter == self.repetitions:
                ll_precision = self.final_precision
                print(bcolors.HEADER + "Adding the final" + str(layer) + " new layers to the existing " + str(
                    layer_counter) +
                      " number of layers" + bcolors.ENDC)
            else:
                ll_precision = self.pretraining_precision
                print(bcolors.HEADER + "Adding " + str(layer) + " new layers to the existing " + str(layer_counter) +
                      " number of layers" + bcolors.ENDC)

            ansatz_ll = ansatz[layer_counter]

            base_num_of_parameters = len(ansatz[0].parameters)

            if q * base_num_of_parameters > (base_num_of_parameters * layer_counter):
                new_free_parameters = ParameterVector('θ', base_num_of_parameters * layer_counter)
            else:
                new_free_parameters = ParameterVector('θ', layer * base_num_of_parameters + q * base_num_of_parameters)

            try:
                num_frozen_parameters = ansatz_ll.num_parameters - len(new_free_parameters)
                frozen_parameters = old_frozen_parameters + list(old_optimized_parameters[:num_frozen_parameters])
                parameter_list = frozen_parameters + list(new_free_parameters)
                point_initial = list(old_optimized_parameters[num_frozen_parameters:])
                if point_initial == []:
                    point_initial = None
                if len(point_initial) != len(new_free_parameters):
                    for adds in range(len(new_free_parameters) - len(point_initial)):
                        point_initial.extend([0])


            except TypeError:
                parameter_list = list(new_free_parameters)

            ansatz_ll.assign_parameters(parameter_list, inplace=True)

            result, final_params = self.execution(ansatz=[ansatz_ll], layers=0, child_directory=child_directory,
                                                  name=name, Ham=Ham,
                                                  SHA_layers=SHA_layers, instruction=instruction,
                                                  final_layer=final_layer,
                                                  entanglement_block=entanglement_block, entangler_map=entangler_map,
                                                  alternating_layer=alternating_layer, precision=ll_precision,
                                                  LL_layer=layer_counter, initial_point=point_initial)
            layer_counter += layer
            old_optimized_parameters = final_params
            old_frozen_parameters = frozen_parameters
        window_start_parameter = list(old_frozen_parameters) + list(old_optimized_parameters)
        number_of_windows = np.ceil((len(window_start_parameter)) * (r / 100))
        number_of_windows = len(window_start_parameter) / number_of_windows
        if number_of_windows > 10:
            print(bcolors.WARNING + "The circuit has been partitioned into more than 10 parts. The optimization process"
                                    " in this case may take a significant amount of time. If this was not intended "
                                    "restart the programm with a larger r value" + bcolors.ENDC)

        print(bcolors.OKBLUE + "Start of moving window optimization process" + bcolors.ENDC)
        window_counter = 0

        while True:

            windows = np.array_split(window_start_parameter, number_of_windows)

            new_ansatz, instruction, entanglement_block, final_layer, alternating_layer = circuit(self.circuit_type,
                                                                                                  self.num_qubits,
                                                                                                  self.repetitions,
                                                                                                  layering=True,
                                                                                                  print_circuit=False)
            ansatz_ll = new_ansatz[self.repetitions - 1]

            window_free_parameters = ParameterVector('θ', len(windows[window_counter]))

            right_list = []
            for i, _ in enumerate(windows[window_counter + 1:]):
                right_list += list(windows[window_counter + 1:][i])
            left_list = []
            for j, _ in enumerate(windows[:window_counter]):
                left_list += list(windows[:window_counter][j])

            window_parameters_list = left_list + list(window_free_parameters) + right_list
            ansatz_ll.assign_parameters(window_parameters_list, inplace=True)
            print(bcolors.OKBLUE + f"Start of window {window_counter}" + bcolors.ENDC)
            if r != 0:
                self.max_iter = 500
            w_result, w_final_params = self.execution(ansatz=[ansatz_ll], layers=0, child_directory=child_directory,
                                                      name=name, Ham=Ham, SHA_layers=SHA_layers,
                                                      instruction=instruction, final_layer=final_layer,
                                                      entanglement_block=entanglement_block,
                                                      entangler_map=entangler_map, alternating_layer=alternating_layer,
                                                      precision=self.final_precision,
                                                      initial_point=windows[window_counter], LL_layer=layer_counter)
            if self.optimizer == "COBYLA":
                if w_result.cost_function_evals < self.max_iter:
                    break
            if self.optimizer == "SLSQP":
                if number_solutions < self.max_iter:
                    break

            window_start_parameter = left_list + list(w_final_params) + right_list
            if window_counter == np.shape(windows)[0] - 1:
                window_counter = 0
            else:
                window_counter += 1
            layer_counter += 1

        print(bcolors.OKGREEN + "End of moving window optimization process" + bcolors.ENDC)
        return w_result

    def LVSHA(self, SHA_method):

        Ham, terms = build_Hamiltonian(colors=self.colors, adj_m=self.A, penalty=[])

        ansatz, instruction, entanglement_block, final_layer, alternating_layer = circuit(self.circuit_type,
                                                                                          self.num_qubits,
                                                                                          self.repetitions,
                                                                                          layering=True,
                                                                                          print_circuit=False)

        Ham_layers = create_SHA_layers(self.colors, self.SHA_layers, SHA_method, terms, self.A)

        try:
            entangler_map = ansatz[-1].entanglement
        except AttributeError:
            entangler_map = ""

        name = "Circuit_" + str(self.circuit_type)

        self.directory = os.path.join(self.directory, f"{self.A.shape[0]}_nodes_{self.colors}_colors_{self.repetitions}"
                                                      f"_reps_LVSHA_graph{self.graph_counter}")

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        child_directory = os.path.join(self.directory, name)
        if not os.path.exists(child_directory):
            os.mkdir(child_directory)

        base_number_of_parameters = ansatz[0].num_parameters
        initial_point, result, final_params = None, None, None

        for c_layer in range(self.repetitions):
            if c_layer == self.repetitions - 1:
                if self.verbose:
                    print(bcolors.HEADER + "Start of final layer " + str(c_layer + 1) + bcolors.ENDC)
            else:
                if self.verbose:
                    print(bcolors.HEADER + "Start of layer " + str(c_layer + 1) + bcolors.ENDC)
            for sha_layer in range(self.SHA_layers):
                if sha_layer == self.SHA_layers - 1:
                    vqe_precision = self.final_precision
                    if self.verbose:
                        print(bcolors.HEADER + "Start of final layer " + str(sha_layer + 1) + bcolors.ENDC)
                else:
                    vqe_precision = self.pretraining_precision
                    if self.verbose:
                        print(bcolors.HEADER + "Start of layer " + str(sha_layer + 1) + bcolors.ENDC)

                result, final_params = self.execution(ansatz=ansatz, layers=c_layer, child_directory=child_directory,
                                                      name=name, Ham=Ham_layers[sha_layer], SHA_layers=sha_layer, instruction=instruction,
                                                      final_layer=final_layer, entanglement_block=entanglement_block,
                                                      entangler_map=entangler_map, alternating_layer=alternating_layer,
                                                      precision=vqe_precision, initial_point=initial_point)

            initial_point = list(final_params)
            for adds in range(base_number_of_parameters):
                initial_point.extend([0])

        return result

    def LLSHA(self, SHA_method, p, q, r):

        assert p > 0 and q > 0 and r > 0 and type(p) == int and type(q) == int, "Hyperparamters must be greater than " \
                                                                                "0 and integers"
        assert p < self.repetitions, "Can't add more layers than there are in the circuit"
        assert r <= 100 and type(r) == int, "r is a percentage and must be defined as an integer in an interval of 0 " \
                                           "to 100"

        Ham, terms = build_Hamiltonian(colors=self.colors, adj_m=self.A, penalty=[])

        ansatz, instruction, entanglement_block, final_layer, alternating_layer = circuit(self.circuit_type,
                                                                                          self.num_qubits,
                                                                                          self.repetitions,
                                                                                          layering=True,
                                                                                          print_circuit=False)

        Ham_layers = create_SHA_layers(self.colors, self.SHA_layers, SHA_method, terms, self.A)

        try:
            entangler_map = ansatz[-1].entanglement
        except AttributeError:
            entangler_map = ""

        name = "Circuit_" + str(self.circuit_type)

        self.directory = os.path.join(self.directory, f"{self.A.shape[0]}_nodes_{self.colors}_colors_{self.repetitions}"
                                                      f"_reps_LLSHA_{p}_{q}_{r}_graph{self.graph_counter}")

        print(self.directory)

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

        self.directory = os.path.join(self.directory, f"stat{self.stat}")
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

        child_directory = os.path.join(self.directory, name)
        if not os.path.exists(child_directory):
            os.mkdir(child_directory)

        aux_counter = self.repetitions
        p_list = []
        while aux_counter != 0:
            if aux_counter % p == 0:
                aux_counter -= p
                p_list.append(p)
            else:
                p_list.append(aux_counter)
                aux_counter = 0

        # pretraining
        layer_counter = 0
        old_frozen_parameters, frozen_parameters = [], []
        old_optimized_parameters = None
        point_initial = None

        for layer in p_list:
            if layer_counter == self.repetitions:
                if self.verbose:
                    print(bcolors.HEADER + "Adding the final" + str(layer) + " new layers to the existing " + str(
                        layer_counter) +
                          " number of layers" + bcolors.ENDC)
            else:
                if self.verbose:
                    print(bcolors.HEADER + "Adding " + str(layer) + " new layers to the existing " + str(layer_counter) +
                          " number of layers" + bcolors.ENDC)

            ansatz_ll = ansatz[layer_counter]

            base_num_of_parameters = len(ansatz[0].parameters)

            if q * base_num_of_parameters > (base_num_of_parameters * layer_counter):

                new_free_parameters = ParameterVector('θ', base_num_of_parameters * layer_counter)
            else:
                new_free_parameters = ParameterVector('θ', layer * base_num_of_parameters + q * base_num_of_parameters)

            try:
                num_frozen_parameters = ansatz_ll.num_parameters - len(new_free_parameters)
                frozen_parameters = old_frozen_parameters + list(old_optimized_parameters[:num_frozen_parameters])
                parameter_list = frozen_parameters + list(new_free_parameters)
                point_initial = list(old_optimized_parameters[num_frozen_parameters:])

                if point_initial == []:
                    point_initial = None
                if len(point_initial) != len(new_free_parameters):
                    for adds in range(len(new_free_parameters)-len(point_initial)):
                        point_initial.extend([0])


            except TypeError:
                parameter_list = list(new_free_parameters)

            ansatz_ll.assign_parameters(parameter_list, inplace=True)

            for sha_layer in range(self.SHA_layers):
                if sha_layer == self.SHA_layers + 1:
                    ll_precision = self.final_precision
                    if self.verbose:
                        print(bcolors.HEADER + "Adding the final SHA layer to the existing " + str(sha_layer) +
                              " SHA layers " + bcolors.ENDC)
                else:
                    ll_precision = self.pretraining_precision
                    if self.verbose:
                        print(
                            bcolors.HEADER + "Adding a new SHA layer existing " + str(sha_layer)
                            + " SHA layers" + bcolors.ENDC)

                result, final_params = self.execution(ansatz=[ansatz_ll], layers=0, child_directory=child_directory,
                                                      name=name, Ham=Ham_layers[sha_layer],
                                                      SHA_layers=sha_layer, instruction=instruction,
                                                      final_layer=final_layer,
                                                      entanglement_block=entanglement_block, entangler_map=entangler_map,
                                                      alternating_layer=alternating_layer, precision=ll_precision,
                                                      LL_layer=layer_counter, initial_point=point_initial)
                point_initial = list(final_params)
            layer_counter += layer
            old_optimized_parameters = final_params
            old_frozen_parameters = frozen_parameters
        window_start_parameter = list(old_frozen_parameters) + list(old_optimized_parameters)
        number_of_windows = np.ceil((len(window_start_parameter)) * (r / 100))
        number_of_windows = len(window_start_parameter) / number_of_windows
        if number_of_windows > 10:
            print(bcolors.WARNING + "The circuit has been partitioned into more than 10 parts. The optimization process"
                                    " in this case may take a significant amount of time. If this was not intended "
                                    "restart the program with a larger r value" + bcolors.ENDC)
        if self.verbose:
            print(bcolors.OKBLUE + "Start of moving window optimization process" + bcolors.ENDC)
        window_counter = 0
        iteration_counter = 0
        while True:

            windows = np.array_split(window_start_parameter, number_of_windows)

            new_ansatz, instruction, entanglement_block, final_layer, alternating_layer = circuit(self.circuit_type,
                                                                                                  self.num_qubits,
                                                                                                  self.repetitions,
                                                                                                  layering=True,
                                                                                                  print_circuit=False)
            ansatz_ll = new_ansatz[self.repetitions - 1]

            window_free_parameters = ParameterVector('θ', len(windows[window_counter]))

            right_list = []
            for i, _ in enumerate(windows[window_counter + 1:]):
                right_list += list(windows[window_counter + 1:][i])
            left_list = []
            for j, _ in enumerate(windows[:window_counter]):
                left_list += list(windows[:window_counter][j])

            window_parameters_list = left_list + list(window_free_parameters) + right_list
            ansatz_ll.assign_parameters(window_parameters_list, inplace=True)
            print(bcolors.OKBLUE + f"Start of window {window_counter}" + bcolors.ENDC)
            if r != 100:
                max_iter_orginial = self.max_iter
                self.max_iter = 500
                w_result, w_final_params = self.execution(ansatz=[ansatz_ll], layers=0, child_directory=child_directory,
                                                          name=name, Ham=Ham, SHA_layers=0,
                                                          instruction=instruction, final_layer=final_layer,
                                                          entanglement_block=entanglement_block,
                                                          entangler_map=entangler_map, alternating_layer=alternating_layer,
                                                          precision=self.final_precision,
                                                          initial_point=windows[window_counter], LL_layer=layer_counter)
                if self.optimizer == "COBYLA":
                    iteration_counter += w_result.cost_function_evals
                    if w_result.cost_function_evals < self.max_iter:
                        break
                if self.optimizer == "SLSQP":
                    iteration_counter += number_solutions
                    if number_solutions < self.max_iter:
                        break

                if iteration_counter <= max_iter_orginial:
                    break
                window_start_parameter = left_list + list(w_final_params) + right_list
                if window_counter == np.shape(windows)[0] - 1:
                    window_counter = 0
                else:
                    window_counter += 1
                layer_counter += 1
            else:
                
                w_result, w_final_params = self.execution(ansatz=[ansatz_ll], layers=0, child_directory=child_directory,
                                                          name=name, Ham=Ham, SHA_layers=0,
                                                          instruction=instruction, final_layer=final_layer,
                                                          entanglement_block=entanglement_block,
                                                          entangler_map=entangler_map,
                                                          alternating_layer=alternating_layer,
                                                          precision=self.final_precision,
                                                          initial_point=windows[window_counter], LL_layer=layer_counter)

                break

        if self.verbose:
            print(bcolors.OKGREEN + "End of moving window optimization process" + bcolors.ENDC)

        if os.path.exists(os.path.join(self.directory, "recovery_file.csv")):
            os.remove(os.path.join(self.directory, "recovery_file.csv"))

        return w_result

    def QAOA(self, layers=0, layering=False, SHA_layers=0):

        Ham, terms = build_Hamiltonian(colors=self.colors, adj_m=self.A, penalty=[])

        ansatz, instruction, entanglement_block, final_layer, alternating_layer = circuit(self.circuit_type,
                                                                                          self.num_qubits,
                                                                                          self.repetitions,
                                                                                          layering=layering,
                                                                                          print_circuit=False)
        try:
            entangler_map = ansatz[-1].entanglement
        except AttributeError:
            entangler_map = ""

        name = "Circuit_" + str(self.circuit_type)

        self.directory = os.path.join(self.directory, f"{self.A.shape[0]}_nodes_{self.colors}_colors_{self.repetitions}"
                                                      f"_reps_QAOA_graph{self.graph_counter}")

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        child_directory = os.path.join(self.directory, name)
        if not os.path.exists(child_directory):
            os.mkdir(child_directory)

        result = self.execution(ansatz=ansatz, layers=layers, child_directory=child_directory, name=name, Ham=Ham,
                                SHA_layers=SHA_layers, instruction=instruction, final_layer=final_layer,
                                entanglement_block=entanglement_block, entangler_map=entangler_map,
                                alternating_layer=alternating_layer, precision=self.final_precision, QAOA_exe=True)

        return result

    def QAOASHA(self, SHA_method):

        _, terms = build_Hamiltonian(colors=self.colors, adj_m=self.A, penalty=[])

        ansatz, instruction, entanglement_block, final_layer, alternating_layer = circuit(self.circuit_type,
                                                                                          self.num_qubits,
                                                                                          self.repetitions,
                                                                                          layering=False,
                                                                                          print_circuit=False)

        Ham_layers = create_SHA_layers(self.colors, self.SHA_layers, SHA_method, terms, self.A)

        try:
            entangler_map = ansatz[-1].entanglement
        except AttributeError:
            entangler_map = ""

        name = "Circuit_" + str(self.circuit_type)

        self.directory = os.path.join(self.directory, f"{self.A.shape[0]}_nodes_{self.colors}_colors_{self.repetitions}"
                                                      f"_reps_QAOA_SHA_{self.SHA_layers}_graph{self.graph_counter}")

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        child_directory = os.path.join(self.directory, name)
        if not os.path.exists(child_directory):
            os.mkdir(child_directory)

        initial_point, result = None, None

        for SHA_layer in range(self.SHA_layers):
            if SHA_layer == self.SHA_layers - 1:
                sha_precision = self.final_precision
                if self.verbose:
                    print(bcolors.HEADER + "Start of final SHA layer " + str(SHA_layer + 1) + bcolors.ENDC)
            else:
                sha_precision = self.pretraining_precision
                if self.verbose:
                    print(bcolors.HEADER + "Start of SHA layer " + str(SHA_layer + 1) + bcolors.ENDC)
            result, final_params = self.execution(ansatz=ansatz, layers=0, child_directory=child_directory, name=name,
                                                  Ham=Ham_layers[SHA_layer], SHA_layers=SHA_layer,
                                                  instruction=instruction, final_layer=final_layer,
                                                  entanglement_block=entanglement_block, entangler_map=entangler_map,
                                                  alternating_layer=alternating_layer, precision=sha_precision,
                                                  initial_point=initial_point, QAOA_exe=True)

            initial_point = list(final_params)

        return result


        return result





