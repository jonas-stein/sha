import sys
from multiprocessing import Process

from Circuits import circuit
from Expressability import compare_layer_express

from Methods import Methods
from Graph_operations import *

import numpy as np
import networkx as nx

import argparse


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


def accuracy_per_shot(state_vector):
    """

    :param state_vector: quantum state vector of the optimization process
    :return: accuracy over the last iteration's individual shots, measured as the ability of each shot to solve
    the graph coloring problem
    """
    acc_counter = 0
    shots = sorted(state_vector.items(), key=lambda kv: kv[1])
    for shot in shots:
        x = np.asarray([int(y) for y in list(shot[0])])
        check = check_coloring(x, print_outs=False)
        if check:
            print(shot[1] * np.sqrt(200), shot[1])
            acc_counter += shot[1] * np.sqrt(200)

    print(acc_counter / 200)

    return acc_counter / 200


def power_of_two(x: str):

    if int(x) % 2 == 0:
        pass
    else:
        raise TypeError("number of colors is not power of two" + bcolors.ENDC)


if __name__ == '__main__':
    print(bcolors.ENDC + "Start")
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph", help="Custom input graph/graphs, which will be investigated. Inputs are number"
                                              " of nodes n and connectivity p . Enter as floats in the form of: "
                                              "n_0 p_0 n_1 p_1 ... . If left empty the 10 predefined graphs will be "
                                              "used.", type=float, nargs="+")

    parser.add_argument("--show_plots", type=str, default=False, help="controls detail level of shown plots. Options "
                                                                      "are: all: shows all plots, all_saved: shows "
                                                                      "all saved plots, graphs: shows plots of to be "
                                                                      "investigated graphs, graphs_reduced: shows "
                                                                      "plots of reduced graphs. Combination are "
                                                                      "possible Default is False ", nargs="+",
                        choices=["all", "all_saved", "graphs", "graphs_reduced"])

    parser.add_argument("-c", "--colors", help="Number of colors used to color the Graph. Please use powers of 2. "
                                               "Default number of colors is 4", type=power_of_two, default=4)

    parser.add_argument("-pqc", "--circuit", type=int, nargs="+", default=[1, 3, 8, 12, 13, 16, 18],
                        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                        help="Circuit templates used to evaluate graphs. For information regarding the specific "
                             "circuits consult my Master thesis")

    parser.add_argument("-reps", "--repetition", type=int, nargs=1, default=3, help="Number of circuit layers")

    parser.add_argument("-t", "--trainability", action="store_true", default=False,
                        help="Flag for turning trainability exploration on. Not recommended for than 10 qubits")

    parser.add_argument("-e", "--expressibility", default=False, choices=["only", "detailed", "ordered", "True"],
                        nargs="+", type=str,
                        help="Flag for turning expressibility exploration on. Choices are 'True': turns on "
                             "expressibility exploration, 'only': performs only the expressibility exploration before "
                             "exciting the code, 'detailed': show plots of Haar distribution which are normally "
                             "hidden, 'ordered': show the results ordered the same way as in the original paper.")

    parser.add_argument("-o", "--optimizer", choices=["SLSQP", "COBYLA"], default="COBYLA", type=str,
                        help="Optimizer for the Quantum Algorithms. Currently only SLSQP and COBYLA are implemented "
                             "here")

    parser.add_argument("-d", "--directory", default=os.getcwd(), type=str,
                        help="Location where the output folders are generated. Default is the current working "
                             "directory")

    parser.add_argument("--max_iter", default=4000, type=int, help="Maximum iterations of the optimizer")

    parser.add_argument("-pp", "--pretraining_precision", default=8e-01, type=float,
                        help="Precision for pretraining process. Default is 8e-01")

    parser.add_argument("-fp", "--final_precision", default=1e-06, type=float,
                        help="Precision for optimizer to conclude the optimization process in the final layer")

    parser.add_argument("--verbose", default=False, action="store_true",
                        help="Flag to control the console print outs. Default is False")

    parser.add_argument("--num_shots", type=int, default=200, help="Number of shots used in for the quantum "
                                                                   "simulations in the main process")

    parser.add_argument("--num_shots_expressibility", type=int, default=1000,
                        help="Number of shots used in the quantum simulations for the calculation of the expressibility"
                        )

    parser.add_argument("--num_expressibility_params", type=int, default=5000,
                        help="Number of parameters used in the creation of the histogram for the calculation of the "
                             "expressibility")
    parser.add_argument("--graph_complexity", default=False, action="store_true",
                        help="Prints out the complexity of each graph out. It is not advised to run this method for "
                             "large graphs, as it tries to find the proper colorable solutions by brute force")

    method = parser.add_mutually_exclusive_group(required=True)

    method.add_argument("--SVQE", action="store_true", help="Flag for using the standard VQE")
    method.add_argument("--LVQE", action="store_true", help="Flag for using the Layer VQE")
    method.add_argument("--LL", action="store_true", help="Flag for using Layerwise Learning")
    method.add_argument("--QAOA", action="store_true", help="Flag for using QAOA")
    method.add_argument("--SHA", type=str, choices=["random", "sequential", "node_wise"], default=False,
                        help="Flag for using SHA. Choices for the SHA methods are random, sequential and node_wise")

    method.add_argument("--SHA_LVQE", type=str, choices=["random", "sequential", "node_wise"], default=False,
                        help="Flag for combining SHA and LVQE. Choices for the SHA methods are random, sequential and "
                             "node_wise")
    method.add_argument("--SHA_LL", type=str, choices=["random", "sequential", "node_wise"], default=False,
                        help="Flag for combining SHA and LL. Choices for the SHA methods are random, sequential and "
                             "node_wise"
                        )
    method.add_argument("--SHA_QAOA", type=str, choices=["random", "sequential", "node_wise"], default=False)

    parser.add_argument("--num_SHA_layers", type=int, default=8,
                        help="Number of SHA layers used. Only has an effect if a method which uses SHA has been chosen."
                        )

    parser.add_argument("--LL_params", type=int, nargs="+", default=[1, 1, 100],
                        help="Flag for setting the Layerwise Learning parameters of p, q, r. Only has an effect if "
                             "LL has been chosen. Input parameters in form ' p q r' as integers")

    parser.add_argument("--multi_process", default=False, action="store_true",
                        help="use a for multiprocessing optimized variant of the code")

    parser.add_argument("--name_of_folder", type=str, default="result",
                        help="Name of te folder where all results will be saved. Default called 'results'")

    parser.add_argument("--num_qubits_expressibility", type=int, default=4, help="Number of Qubits of the circuits used"
                                                                                 " in their expressibility exploration")

    parser.add_argument("--num_expressibility_layers", type=int, default=5, help="Number of circuit layers used in "
                                                                                 "their expressibility exploration")

    parser.add_argument("--print_circuits", default=False, type=int,
                        help="Prints out the layouts for the used circuits. Input defines for how many qubits the "
                             "circuits are going to be printed out")

    parser.add_argument("--stat", type=int, default=1)

    args = parser.parse_args()
    print(bcolors.BOLD + bcolors.OKBLUE + "Arguments used in the calculation are:" + bcolors.ENDC)
    print(args)

    if args.graph is None:
        graphs = [nx.fast_gnp_random_graph(8, p=0.9, seed=7), nx.fast_gnp_random_graph(8, p=0.55, seed=8),
                  nx.fast_gnp_random_graph(8, p=0.4, seed=9), nx.fast_gnp_random_graph(8, p=0.35, seed=10),
                  nx.fast_gnp_random_graph(8, p=0.35, seed=11), nx.fast_gnp_random_graph(8, p=0.4, seed=12),
                  nx.fast_gnp_random_graph(8, p=0.3, seed=13), nx.fast_gnp_random_graph(8, p=0.5, seed=14),
                  nx.fast_gnp_random_graph(8, p=0.9, seed=15), nx.fast_gnp_random_graph(8, p=0.4, seed=16)]
    else:
        graphs = []
        for i in range(0, len(args.graph), 2):
            graphs.append(nx.fast_gnp_random_graph(int(args.graph[i]), args.graph[i + 1]))

    FLAG_show_saved = False
    FLAG_show_graphs = False
    FLAG_show_graphs_reduced = False
    if args.show_plots is not False:
        for argument in args.show_plots:
            if argument == "all":
                FLAG_show_saved = True
                FLAG_show_graphs = True
                FLAG_show_graphs_reduced = True
                break
            if argument == "all_saved":
                FLAG_show_saved = True
            if argument == "graphs":
                FLAG_show_graphs = True
            if argument == "graphs_reduced":
                FLAG_show_graphs_reduced = True

    circuit_list = args.circuit
    repetitions = args.repetition
    colors = args.colors
    verbose = args.verbose

    FLAG_expressibility = False
    FLAG_expressibility_ordered = False
    FLAG_expressibility_only = False
    FLAG_expressibility_detailed = False
    if args.expressibility is not False:
        FLAG_expressibility = True
        for arguments in args.expressibility:
            if arguments == "ordered":
                FLAG_expressibility_ordered = True
            if arguments == "only":
                FLAG_expressibility_only = True
            if arguments == "detailed":
                FLAG_expressibility_detailed = True

    FLAG_trainability = args.trainability

    parent_directory = args.directory
    directory = os.path.join(parent_directory, args.name_of_folder)
    if not os.path.exists(directory):
        os.mkdir(directory)
        print(bcolors.WARNING + "Created folder" + directory + bcolors.ENDC)
    if not os.path.exists(parent_directory):
        print(bcolors.FAIL + parent_directory + " does not exist. Choose an existing directory or use the default "
                                                "current working directory" + bcolors.ENDC)
        sys.exit(bcolors.WARNING + "Process finished with exit code 1" + bcolors.ENDC)

    SVQE = args.SVQE
    LVQE = args.LVQE
    LL = args.LL
    SHA = False
    QAOA = args.QAOA
    SHA_LL = False
    SHA_LVQE = False
    SHA_QAOA = False
    assert len(args.LL_params) == 3, f"Only 3 parameters can be defined for the input. Number of parameters " \
                                     f"in the current input: {len(args.LL_params)}"
    for ll_param in args.LL_params:
        assert type(ll_param) == int, f"Layerwise Learning parameters must be given as an integer. " \
                                      f"Current input type {type(ll_param)}"
    p = args.LL_params[0]
    q = args.LL_params[1]
    r = args.LL_params[2]

    if args.SHA is not False:
        SHA = True
        SHA_method = args.SHA
    if args.SHA_LVQE is not False:
        SHA_LVQE = True
        SHA_method = args.SHA_LVQE
    if args.SHA_LL is not False:
        SHA_LL = True
        SHA_method = args.SHA_LL
    if args.SHA_QAOA is not False:
        SHA_QAOA = True

        SHA_method = args.SHA_QAOA


    print(bcolors.HEADER + "Start of program " + bcolors.ENDC)

    if FLAG_show_graphs:
        graph_title = 0
        for graph in graphs:
            plt.figure()
            nx.draw(graph)
            plt.title("Graph" + str(graph_title))
            plt.show()
            graph_title += 1

    if args.print_circuits is not False:
        for print_circs in circuit_list:
            print(bcolors.OKCYAN + "Layout of Circuit " + str(print_circs) + ":" + bcolors.ENDC)
            circuit(print_circs, args.print_circuits, repetitions, layering=False, print_circuit=True)

    if FLAG_expressibility:
        print(bcolors.OKBLUE + "Start of expressibility exploration for circuits " + str(circuit_list) + " using " +
              str(args.num_expressibility_params) + " parameters and " + str(args.num_shots_expressibility) +
              " shots" + bcolors.ENDC)

        compare_layer_express(circ_list=circuit_list, num_qubits=args.num_qubits_expressibility,
                              repetition=args.num_expressibility_layers, num_shots=args.num_shots_expressibility,
                              num_params=args.num_expressibility_params, directory=directory,
                              detailed=FLAG_expressibility_detailed, ordered=FLAG_expressibility_ordered,
                              FLAG_show_saved=FLAG_show_saved)

        if FLAG_expressibility_only:
            sys.exit(bcolors.OKBLUE + "Program concluded due to 'only' settings with exit code 1" + bcolors.ENDC)

    if FLAG_show_graphs_reduced:
        for graph_counter_r, graph_r in enumerate(graphs, start=1):
            reduced_A = AMreduce(nx.to_numpy_matrix(graph_r, weight=None))
            reduced_graph = nx.from_numpy_matrix(reduced_A)
            plt.subplot()
            nx.draw(reduced_graph)
            plt.title(f"Graph {graph_counter_r}")
            plt.show()
            if args.graph_complexity:
                ratio_g, counter_g = brute_force_coloring(reduced_A, colors)
                print(bcolors.OKBLUE + f"Current Graph ha s a complexity of {ratio_g} and {counter_g} "
                                       f"number of colorable solutions")


    def task(graph, graph_counter):
        for circuit in circuit_list:
            A = AMreduce(nx.to_numpy_matrix(graph, weight=None))

            if args.graph_complexity:
                ratio_g, counter_g = brute_force_coloring(A, colors)
                print(bcolors.OKBLUE + f"Current Graph ha s a complexity of {ratio_g} and {counter_g} "
                                       f"number of colorable solutions")
            result=None
            Num_qubits = int(A.shape[1] * np.floor(np.log2(colors)))

            optimization_method = Methods(A=A, circuit_type=circuit, repetitions=repetitions, colors=colors,
                                          directory=directory, graph_counter=graph_counter, num_qubits=Num_qubits,
                                          optimizer=args.optimizer, max_iter=args.max_iter, verbose=args.verbose,
                                          pretraining_precision=args.pretraining_precision,
                                          final_precision=args.final_precision,
                                          SHA_layers=args.num_SHA_layers, FLAG_show_saved=FLAG_show_saved,
                                          FLAG_trainability=FLAG_trainability, num_shots=args.num_shots, stat=args.stat)
            if SVQE:
                result = optimization_method.SVQE()

            if LVQE:
                result = optimization_method.LVQE()

            if SHA:
                result = optimization_method.SHA(SHA_method=SHA_method)

            if LL:
                result = optimization_method.LL(p=p, q=q, r=r)

            if SHA_LL:
                result = optimization_method.LLSHA(SHA_method=SHA_method, p=p, q=q, r=r)

            if QAOA:
                result = optimization_method.QAOA()

            if SHA_LVQE:
                result = optimization_method.LVSHA(SHA_method=SHA_method)

            if SHA_QAOA:
                result = optimization_method.QAOASHA(SHA_method=SHA_method)

            print(bcolors.OKGREEN + "Final result for Circuit " + str(circuit) + " in Graph " + str(graph_counter)
                  + " is:" + bcolors.ENDC)
            print(result)


    if args.multi_process:
        processes = [Process(target=task, args=(graphs[i], i)) for i in range(len(graphs))]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
    else:
        for graph_counter, graph in enumerate(graphs, start=1):
            task(graph=graph, graph_counter=graph_counter)


    print("End")
