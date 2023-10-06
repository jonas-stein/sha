Installation:

Install the requirements.txt ina virtual environment. The file contains all necessary liberies for to the Code to run properly.

Usage:

The Code has a variety of inputs which are defined as below. To run its base functionalities with default inputs, the
used method (SVQE, LVQE, LL, SHA, SHA + LVQE, SHA + LL, QAOA) needs to be defined solely. The default inputs are the
same values as the ones to procure the evalution results. It is not advised to run the code with default parameters
locally, as the code can take in this configurations days to conclude. Either execute it on a SLURM system or use
smaller input graphs, number of optimization etc. The result can be found either at the default location, which is the
same place where the code is saved, or at the location which can be defined by the --directory flag. All other flags
can be found here below or by using the --help flag:

usage: main.py [-h] [-g GRAPH [GRAPH ...]]
               [--show_plots {all,all_saved,graphs,graphs_reduced} [{all,all_saved,graphs,graphs_reduced} ...]]
               [-c COLORS]
               [-pqc {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19} [{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19} ...]]
               [-reps REPETITION] [-t]
               [-e {only,detailed,ordered,True} [{only,detailed,ordered,True} ...]]
               [-o {SLSQP,COBYLA}] [-d DIRECTORY] [--max_iter MAX_ITER]
               [-pp PRETRAINING_PRECISION] [-fp FINAL_PRECISION] [--verbose]
               [--num_shots NUM_SHOTS]
               [--num_shots_expressibility NUM_SHOTS_EXPRESSIBILITY]
               [--num_expressibility_params NUM_EXPRESSIBILITY_PARAMS]
               [--graph_complexity]
               (--SVQE | --LVQE | --LL | --QAOA | --SHA {random,sequential,node_wise} | --SHA_LVQE {random,sequential,node_wise} | --SHA_LL {random,sequential,node_wise})
               [--num_SHA_layers NUM_SHA_LAYERS]
               [--LL_params LL_PARAMS [LL_PARAMS ...]] [--multi_process]
               [--name_of_folder NAME_OF_FOLDER]
               [--num_qubits_expressibility NUM_QUBITS_EXPRESSIBILITY]
               [--num_expressibility_layers NUM_EXPRESSIBILITY_LAYERS]
               [--print_circuits PRINT_CIRCUITS]

optional arguments:
  -h, --help            show this help message and exit
  -g GRAPH [GRAPH ...], --graph GRAPH [GRAPH ...]
                        Custom input graph/graphs, which will be investigated.
                        Inputs are number of nodes n and connectivity p .
                        Enter as floats in the form of: n_0 p_0 n_1 p_1 ... .
                        If left empty the 10 predefined graphs will be used.
  --show_plots {all,all_saved,graphs,graphs_reduced} [{all,all_saved,graphs,graphs_reduced} ...]
                        controls detail level of shown plots. Options are:
                        all: shows all plots, all_saved: shows all saved
                        plots, graphs: shows plots of to be investigated
                        graphs, graphs_reduced: shows plots of reduced graphs.
                        Combination are possible Default is False
  -c COLORS, --colors COLORS
                        Number of colors used to color the Graph. Please use
                        powers of 2. Default number of colors is 4
  -pqc {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19} [{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19} ...], --circuit {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19} [{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19} ...]
                        Circuit templates used to evaluate graphs. For
                        information regarding the specific circuits consult my
                        Master thesis
  -reps REPETITION, --repetition REPETITION
                        Number of circuit layers
  -t, --trainability    Flag for turning trainability exploration on. Not
                        recommended for than 10 qubits
  -e {only,detailed,ordered,True} [{only,detailed,ordered,True} ...], --expressibility {only,detailed,ordered,True} [{only,detailed,ordered,True} ...]
                        Flag for turning expressibility exploration on.
                        Choices are 'True': turns on expressibility
                        exploration, 'only': performs only the expressibility
                        exploration before exciting the code, 'detailed': show
                        plots of Haar distribution which are normally hidden,
                        'ordered': show the results ordered the same way as in
                        the original paper.
  -o {SLSQP,COBYLA}, --optimizer {SLSQP,COBYLA}
                        Optimizer for the Quantum Algorithms. Currently only
                        SLSQP and COBYLA are implemented here
  -d DIRECTORY, --directory DIRECTORY
                        Location where the output folders are generated.
                        Default is the current working directory
  --max_iter MAX_ITER   Maximum iterations of the optimizer
  -pp PRETRAINING_PRECISION, --pretraining_precision PRETRAINING_PRECISION
                        Precision for pretraining process. Default is 8e-01
  -fp FINAL_PRECISION, --final_precision FINAL_PRECISION
                        Precision for optimizer to conclude the optimization
                        process in the final layer
  --verbose             Flag to control the console print outs. Default is
                        False
  --num_shots NUM_SHOTS
                        Number of shots used in for the quantum simulations in
                        the main process
  --num_shots_expressibility NUM_SHOTS_EXPRESSIBILITY
                        Number of shots used in the quantum simulations for
                        the calculation of the expressibility
  --num_expressibility_params NUM_EXPRESSIBILITY_PARAMS
                        Number of parameters used in the creation of the
                        histogram for the calculation of the expressibility
  --graph_complexity    Prints out the complexity of each graph out. It is not
                        advised to run this method for large graphs, as it
                        tries to find the proper colorable solutions by brute
                        force
  --SVQE                Flag for using the standard VQE
  --LVQE                Flag for using the Layer VQE
  --LL                  Flag for using Layerwise Learning
  --QAOA                Flag for using QAOA
  --SHA {random,sequential,node_wise}
                        Flag for using SHA. Choices for the SHA methods are
                        random, sequential and node_wise
  --SHA_LVQE {random,sequential,node_wise}
                        Flag for combining SHA and LVQE. Choices for the SHA
                        methods are random, sequential and node_wise
  --SHA_LL {random,sequential,node_wise}
                        Flag for combining SHA and LL. Choices for the SHA
                        methods are random, sequential and node_wise
  --num_SHA_layers NUM_SHA_LAYERS
                        Number of SHA layers used. Only has an effect if a
                        method which uses SHA has been chosen.
  --LL_params LL_PARAMS [LL_PARAMS ...]
                        Flag for setting the Layerwise Learning parameters of
                        p, q, r. Only has an effect if LL has been chosen.
                        Input parameters in form ' p q r' as integers
  --multi_process       use a for multiprocessing optimized variant of the
                        code
  --name_of_folder NAME_OF_FOLDER
                        Name of te folder where all results will be saved.
                        Default called 'results'
  --num_qubits_expressibility NUM_QUBITS_EXPRESSIBILITY
                        Number of Qubits of the circuits used in their
                        expressibility exploration
  --num_expressibility_layers NUM_EXPRESSIBILITY_LAYERS
                        Number of circuit layers used in their expressibility
                        exploration
  --print_circuits PRINT_CIRCUITS
                        Prints out the layouts for the used circuits. Input
                        defines for how many qubits the circuits are going to
                        be printed out