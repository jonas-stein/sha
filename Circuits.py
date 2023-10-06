import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit import QuantumRegister
from qiskit.circuit.library import TwoLocal


def circuit(number_of_circuit: int, num_qubits: int, num_layers: int, layering: bool = False,
            print_circuit: bool = False):
    def rotate(l, n):
        return l[n:] + l[:n]

    final_layer = False
    alternating_layer = False
    ansatz = []

    if number_of_circuit == 0:
        if not layering:
            entangler_map = []
            for qbits in range(0, num_qubits, 2):
                entangler_map.append([qbits, qbits + 1])
            repatition = num_layers

            rotation_gates = ["rz", "h", "rz", "crz", "rz"]
            entanglement_gates = "crz"

            ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, entangler_map, reps=repatition,
                                   insert_barriers=True, skip_final_rotation_layer=True))
        if layering:

            for layers in range(num_layers):

                entangler_map = []
                for qbits in range(0, num_qubits, 2):
                    entangler_map.append([qbits, qbits + 1])
                repatition = layers + 1

                rotation_gates = ["rz", "h", "rz", "crz", "rz"]
                entanglement_gates = "crz"

                ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, entangler_map, reps=repatition,
                                       insert_barriers=True, skip_final_rotation_layer=True))

    elif number_of_circuit == 1:
        if not layering:
            rotation_gates = ["rx", "rz"]
            entanglement_gates = 0

            ansatz.append(TwoLocal(num_qubits, rotation_gates, reps=num_layers, insert_barriers=True,
                                   skip_final_rotation_layer=True))
        if layering:
            for layers in range(num_layers):
                rotation_gates = ["rx", "rz"]
                entanglement_gates = 0

                ansatz.append(TwoLocal(num_qubits, rotation_gates, reps=layers + 1, insert_barriers=True,
                                       skip_final_rotation_layer=True))

    elif number_of_circuit == 2:
        if not layering:
            instructions = []
            rotation_gates = ["rx", "rz"]
            entanglement_gates = "cz"
            for j in reversed(range(1, num_qubits)):
                instructions.append((j, j - 1))

            ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=num_layers,
                                   insert_barriers=True, skip_final_rotation_layer=True))
        if layering:

            for layers in range(num_layers):

                instructions = []
                rotation_gates = ["rx", "rz"]
                entanglement_gates = "cz"
                for j in reversed(range(1, num_qubits)):
                    instructions.append((j, j - 1))

                ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=layers + 1,
                                       insert_barriers=True, skip_final_rotation_layer=True))

    elif number_of_circuit == 3:
        if not layering:
            instructions = []
            rotation_gates = ["rx", "rz"]
            entanglement_gates = "crz"
            for j in reversed(range(1, num_qubits)):
                instructions.append((j, j - 1))

            ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=num_layers,
                                   insert_barriers=True, skip_final_rotation_layer=True))
        if layering:

            for layers in range(num_layers):

                instructions = []
                rotation_gates = ["rx", "rz"]
                entanglement_gates = "crz"
                for j in reversed(range(1, num_qubits)):
                    instructions.append((j, j - 1))

                ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=layers + 1,
                                       insert_barriers=True, skip_final_rotation_layer=True))

    elif number_of_circuit == 4:
        if not layering:
            instructions = []
            rotation_gates = ["rx", "rz"]
            entanglement_gates = "crx"
            for j in reversed(range(1, num_qubits)):
                instructions.append((j, j - 1))

            ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=num_layers,
                                   insert_barriers=True, skip_final_rotation_layer=True))
        if layering:
            for layers in range(num_layers):

                instructions = []
                rotation_gates = ["rx", "rz"]
                entanglement_gates = "crx"
                for j in reversed(range(1, num_qubits)):
                    instructions.append((j, j - 1))

                ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=layers + 1,
                                       insert_barriers=True, skip_final_rotation_layer=True))

    elif number_of_circuit == 5:
        if not layering:
            instructions = []
            linlist = []
            rotation_gates = ["rx", "rz"]

            entanglement_gates = "crz"
            final_layer = True
            for i in (range(1, num_qubits)):
                linlist.append(i)
            linlist.reverse()

            for j in reversed(range(num_qubits)):

                linlist2 = linlist

                for k in range(0, len(linlist)):
                    if j != linlist2[k]:
                        instructions.append((j, linlist2[k]))

            ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=num_layers,
                                   insert_barriers=True, skip_final_rotation_layer=False))
        if layering:
            for layers in range(num_layers):

                instructions = []
                linlist = []
                rotation_gates = ["rx", "rz"]

                entanglement_gates = "crz"
                final_layer = True
                for i in (range(1, num_qubits)):
                    linlist.append(i)
                linlist.reverse()

                for j in reversed(range(num_qubits)):

                    linlist2 = linlist

                    for k in range(0, len(linlist)):
                        if j != linlist2[k]:
                            instructions.append((j, linlist2[k]))

                ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=layers + 1,
                                       insert_barriers=True, skip_final_rotation_layer=False))

    elif number_of_circuit == 6:
        if not layering:
            instructions = []
            linlist = []
            rotation_gates = ["rx", "rz"]

            entanglement_gates = "crx"
            final_layer = True
            for i in (range(1, num_qubits)):
                linlist.append(i)
            linlist.reverse()

            for j in reversed(range(num_qubits)):

                linlist2 = linlist

                for k in range(0, len(linlist)):
                    if j != linlist2[k]:
                        instructions.append((j, linlist2[k]))

            ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=num_layers,
                                   insert_barriers=True, skip_final_rotation_layer=False))
        if layering:
            for layers in range(num_layers):

                instructions = []
                linlist = []
                rotation_gates = ["rx", "rz"]

                entanglement_gates = "crx"
                final_layer = True
                for i in (range(1, num_qubits)):
                    linlist.append(i)
                linlist.reverse()

                for j in reversed(range(num_qubits)):

                    linlist2 = linlist

                    for k in range(0, len(linlist)):
                        if j != linlist2[k]:
                            instructions.append((j, linlist2[k]))

                ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=layers + 1,
                                       insert_barriers=True, skip_final_rotation_layer=False))

    elif number_of_circuit == 7:
        if not layering:
            instructions = []
            alternating_layer = True
            i = 0
            j = num_qubits - 1
            rotation_gates = ["rx", "rz"]
            entanglement_gates = "crz"
            for lays in range(int(np.floor(num_qubits / 2))):

                single_instructions = []
                if i + 1 != j:
                    single_instructions.append((i + 1, i))
                    single_instructions.append((j, j - 1))
                else:
                    single_instructions.append((i, i + 1))
                instructions.append(single_instructions)
                i += 1
                j -= 1

            ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions,
                                   reps=(num_layers) * int(np.floor(num_qubits / 2)), insert_barriers=True,
                                   skip_final_rotation_layer=True))
        if layering:
            for layers in range(num_layers):

                instructions = []
                alternating_layer = True
                i = 0
                j = num_qubits - 1
                rotation_gates = ["rx", "rz"]
                entanglement_gates = "crz"
                for lays in range(int(np.floor(num_qubits / 2))):

                    single_instructions = []
                    if i + 1 != j:
                        single_instructions.append((i + 1, i))
                        single_instructions.append((j, j - 1))
                    else:
                        single_instructions.append((i, i + 1))
                    instructions.append(single_instructions)
                    i += 1
                    j -= 1

                ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions,
                                       reps=(layers + 1) * int(np.floor(num_qubits / 2)), insert_barriers=True,
                                       skip_final_rotation_layer=True))

    elif number_of_circuit == 8:
        if not layering:

            instructions = []
            alternating_layer = True
            rotation_gates = ["rx", "rz"]
            entanglement_gates = "crx"
            i = 0
            j = num_qubits - 1
            for lays in range(int(np.floor(num_qubits / 2))):

                single_instructions = []
                if i + 1 != j:
                    single_instructions.append((i + 1, i))
                    single_instructions.append((j, j - 1))
                else:
                    single_instructions.append((i, i + 1))
                instructions.append(single_instructions)
                i += 1
                j -= 1

            ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions,
                                   reps=num_layers * int(np.floor(num_qubits / 2)), insert_barriers=True,
                                   skip_final_rotation_layer=True))
        if layering:
            for layers in range(num_layers):

                instructions = []
                alternating_layer = True
                rotation_gates = ["rx", "rz"]
                entanglement_gates = "crx"
                i = 0
                j = num_qubits - 1
                for lays in range(int(np.floor(num_qubits / 2))):

                    single_instructions = []
                    if i + 1 != j:
                        single_instructions.append((i + 1, i))
                        single_instructions.append((j, j - 1))
                    else:
                        single_instructions.append((i, i + 1))
                    instructions.append(single_instructions)
                    i += 1
                    j -= 1

                ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions,
                                       reps=(layers + 1) * int(np.floor(num_qubits / 2)), insert_barriers=True,
                                       skip_final_rotation_layer=True))

    elif number_of_circuit == 9:
        if not layering:

            rotation_gates = ["h", "rx"]
            entanglement_gates = ["cz"]
            instructions = []
            for j in reversed(range(1, num_qubits)):
                instructions.append((j, j - 1))

            param_vector = ParameterVector('θ', num_qubits * num_layers)

            qr = QuantumRegister(num_qubits)
            qc = QuantumCircuit(qr)
            counter = 0
            for layer in range(num_layers):
                for a in range(num_qubits):
                    qc.h(a)
                for instruct in instructions:
                    qc.cz(instruct[0], instruct[1])
                for b in range(num_qubits):
                    qc.rx(param_vector[counter], b)
                    counter += 1
            ansatz.append(qc)
        if layering:
            for layers in range(num_layers):
                rotation_gates = ["h", "rx"]
                entanglement_gates = ["cz"]
                instructions = []
                for j in reversed(range(1, num_qubits)):
                    instructions.append((j, j - 1))

                param_vector = ParameterVector('θ', num_qubits * num_layers)

                qr = QuantumRegister(num_qubits)
                qc = QuantumCircuit(qr)
                counter = 0
                for layer in range(layers + 1):
                    for a in range(num_qubits):
                        qc.h(a)
                    for instruct in instructions:
                        qc.cz(instruct[0], instruct[1])
                    for b in range(num_qubits):
                        qc.rx(param_vector[counter], b)
                        counter += 1
                ansatz.append(qc)

    elif number_of_circuit == 10:
        if not layering:

            rotation_gates = ["ry"]
            entanglement_gates = "cz"
            instructions = []
            final_layer = True
            for j in reversed(range(0, num_qubits)):
                instructions.append((j, j - 1))

            ansatz.append(TwoLocal(num_qubits, ["ry"], "cz", instructions, reps=num_layers, insert_barriers=True))
        if layering:
            for layers in range(num_layers):
                rotation_gates = ["ry"]
                entanglement_gates = "cz"
                instructions = []
                final_layer = True
                for j in reversed(range(0, num_qubits)):
                    instructions.append((j, j - 1))

                ansatz.append(TwoLocal(num_qubits, ["ry"], "cz", instructions, reps=layers + 1, insert_barriers=True))

    elif number_of_circuit == 11:
        if not layering:
            instructions = []
            i = 0
            j = num_qubits - 1
            rotation_gates = ["rx", "rz"]
            entanglement_gates = "cx"
            # construct instructions for entanglement
            for lays in range(int(np.floor(num_qubits / 2))):

                single_instructions = []
                if i + 1 != j:
                    single_instructions.append((i + 1, i))
                    single_instructions.append((j, j - 1))
                else:
                    single_instructions.append((i, i + 1))
                instructions.append(single_instructions)
                i += 1
                j -= 1

            # define number of parameters
            p = 0
            for k in range(int(np.floor(num_qubits / 2))):
                p += 2 * (num_qubits - 2 * k)
            if (num_qubits % 2) != 0:
                p += 2 * len(instructions)

            param_vector = ParameterVector('θ', p * num_layers)

            qr = QuantumRegister(num_qubits)
            qc = QuantumCircuit(qr)
            counter = 0
            for layer in range(num_layers):
                for w in range(len(instructions)):

                    for a in range(w, num_qubits - w):
                        qc.ry(param_vector[counter], a)
                        counter += 1

                    for b in range(w, num_qubits - w):
                        qc.rz(param_vector[counter], b)
                        counter += 1

                    for instruct in instructions[w]:
                        qc.cx(instruct[0], instruct[1])
            ansatz.append(qc)
        if layering:

            for layers in range(num_layers):
                instructions = []
                i = 0
                j = num_qubits - 1
                rotation_gates = ["rx", "rz"]
                entanglement_gates = "cx"
                # construct instructions for entanglement
                for lays in range(int(np.floor(num_qubits / 2))):

                    single_instructions = []
                    if i + 1 != j:
                        single_instructions.append((i + 1, i))
                        single_instructions.append((j, j - 1))
                    else:
                        single_instructions.append((i, i + 1))
                    instructions.append(single_instructions)
                    i += 1
                    j -= 1

                # define number of parameters
                p = 0
                for k in range(int(np.floor(num_qubits / 2))):
                    p += 2 * (num_qubits - 2 * k)
                if (num_qubits % 2) != 0:
                    p += 2 * len(instructions)

                param_vector = ParameterVector('θ', p * num_layers)

                qr = QuantumRegister(num_qubits)
                qc = QuantumCircuit(qr)
                counter = 0
                for layer in range(layers + 1):
                    for w in range(len(instructions)):

                        for a in range(w, num_qubits - w):
                            qc.ry(param_vector[counter], a)
                            counter += 1

                        for b in range(w, num_qubits - w):
                            qc.rz(param_vector[counter], b)
                            counter += 1

                        for instruct in instructions[w]:
                            qc.cx(instruct[0], instruct[1])
                ansatz.append(qc)

    elif number_of_circuit == 12:
        if not layering:
            instructions = []
            i = 0
            j = num_qubits - 1
            rotation_gates = ["rx", "rz"]
            entanglement_gates = "cz"
            # construct instructions for entanglement
            for lays in range(int(np.floor(num_qubits / 2))):

                single_instructions = []
                if i + 1 != j:
                    single_instructions.append((i + 1, i))
                    single_instructions.append((j, j - 1))
                else:
                    single_instructions.append((i, i + 1))
                instructions.append(single_instructions)
                i += 1
                j -= 1

            # define number of parameters
            p = 0
            for k in range(int(np.floor(num_qubits / 2))):
                p += 2 * (num_qubits - 2 * k)
            if (num_qubits % 2) != 0:
                p += 2 * len(instructions)

            param_vector = ParameterVector('θ', p * num_layers)

            qr = QuantumRegister(num_qubits)
            qc = QuantumCircuit(qr)
            counter = 0
            for layer in range(num_layers):
                for w in range(len(instructions)):

                    for a in range(w, num_qubits - w):
                        qc.ry(param_vector[counter], a)
                        counter += 1

                    for b in range(w, num_qubits - w):
                        qc.rz(param_vector[counter], b)
                        counter += 1

                    for instruct in instructions[w]:
                        qc.cz(instruct[0], instruct[1])
            ansatz.append(qc)
        if layering:
            ansatz = []
            for layers in range(num_layers):
                instructions = []
                i = 0
                j = num_qubits - 1
                rotation_gates = ["rx", "rz"]
                entanglement_gates = "cz"
                # construct instructions for entanglement
                for lays in range(int(np.floor(num_qubits / 2))):

                    single_instructions = []
                    if i + 1 != j:
                        single_instructions.append((i + 1, i))
                        single_instructions.append((j, j - 1))
                    else:
                        single_instructions.append((i, i + 1))
                    instructions.append(single_instructions)
                    i += 1
                    j -= 1

                # define number of parameters
                p = 0
                for k in range(int(np.floor(num_qubits / 2))):
                    p += 2 * (num_qubits - 2 * k)
                if (num_qubits % 2) != 0:
                    p += 2 * len(instructions)

                param_vector = ParameterVector('θ', p * num_layers)

                qr = QuantumRegister(num_qubits)
                qc = QuantumCircuit(qr)
                counter = 0
                for layer in range(layers + 1):
                    for w in range(len(instructions)):

                        for a in range(w, num_qubits - w):
                            qc.ry(param_vector[counter], a)
                            counter += 1

                        for b in range(w, num_qubits - w):
                            qc.rz(param_vector[counter], b)
                            counter += 1

                        for instruct in instructions[w]:
                            qc.cz(instruct[0], instruct[1])
                ansatz.append(qc)

    elif number_of_circuit == 13:
        if not layering:
            alternating_layer = True
            instructions = []
            singleinstructions = []
            auxlist = []
            for i in range(num_qubits):
                auxlist.append(i)
            auxlist.reverse()
            auxlist2 = rotate(auxlist, -1)

            for j in range(num_qubits):
                singleinstructions.append((auxlist[j], auxlist2[j]))
            instructions.append(singleinstructions)
            auxlist.reverse()
            auxlist3 = rotate(auxlist, -1)
            auxlist4 = rotate(auxlist, -2)
            singleinstructions = []

            for k in range(num_qubits ):
                singleinstructions.append((auxlist3[k], auxlist4[k]))
            instructions.append(singleinstructions)
            rotation_gates = ["ry"]
            entanglement_gates = "crz"

            ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions,
                                   reps=num_layers * 2, insert_barriers=True, skip_final_rotation_layer=True))
        if layering:
            for layers in range(num_layers):
                alternating_layer = True
                instructions = []
                singleinstructions = []
                auxlist = []
                for i in range(num_qubits):
                    auxlist.append(i)
                auxlist.reverse()
                auxlist2 = rotate(auxlist, -1)
                for j in range(num_qubits):
                    singleinstructions.append((auxlist[j], auxlist2[j]))
                instructions.append(singleinstructions)
                auxlist.reverse()
                auxlist3 = rotate(auxlist, -1)
                auxlist4 = rotate(auxlist, -2)
                singleinstructions = []

                for k in range(num_qubits):
                    singleinstructions.append((auxlist3[k], auxlist4[k]))
                instructions.append(singleinstructions)
                rotation_gates = ["ry"]
                entanglement_gates = "crz"
                ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions,
                                       reps=(layers + 1) * 2, insert_barriers=True, skip_final_rotation_layer=True))

    elif number_of_circuit == 14:
        if not layering:
            alternating_layer = True
            instructions = []
            singleinstructions = []
            auxlist = []
            for i in range(num_qubits):
                auxlist.append(i)
            auxlist.reverse()
            auxlist2 = rotate(auxlist, -1)
            for j in range(num_qubits):
                singleinstructions.append((auxlist[j], auxlist2[j]))
            instructions.append(singleinstructions)
            auxlist.reverse()
            auxlist3 = rotate(auxlist, -1)
            auxlist4 = rotate(auxlist, -2)
            singleinstructions = []

            for k in range(num_qubits):
                singleinstructions.append((auxlist3[k], auxlist4[k]))
            instructions.append(singleinstructions)
            rotation_gates = ["ry"]
            entanglement_gates = "crx"
            ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions,
                                   reps=num_layers * 2, insert_barriers=True, skip_final_rotation_layer=True))
        if layering:
            for layers in range(num_layers):
                alternating_layer = True
                instructions = []
                singleinstructions = []
                auxlist = []
                for i in range(num_qubits):
                    auxlist.append(i)
                auxlist.reverse()
                auxlist2 = rotate(auxlist, -1)
                for j in range(num_qubits):
                    singleinstructions.append((auxlist[j], auxlist2[j]))
                instructions.append(singleinstructions)
                auxlist.reverse()
                auxlist3 = rotate(auxlist, -1)
                auxlist4 = rotate(auxlist, -2)
                singleinstructions = []

                for k in range(num_qubits):
                    singleinstructions.append((auxlist3[k], auxlist4[k]))
                instructions.append(singleinstructions)
                rotation_gates = ["ry"]
                entanglement_gates = "crx"
                ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions,
                                       reps=(layers + 1) * 2, insert_barriers=True, skip_final_rotation_layer=True))

    elif number_of_circuit == 15:
        if not layering:
            alternating_layer = True
            instructions = []
            singleinstructions = []
            auxlist = []
            for i in range(num_qubits):
                auxlist.append(i)
            auxlist.reverse()
            auxlist2 = rotate(auxlist, -1)
            for j in range(num_qubits):
                singleinstructions.append((auxlist[j], auxlist2[j]))
            instructions.append(singleinstructions)
            auxlist.reverse()
            auxlist3 = rotate(auxlist, -1)
            auxlist4 = rotate(auxlist, -2)
            singleinstructions = []

            for k in range(num_qubits):
                singleinstructions.append((auxlist3[k], auxlist4[k]))
            instructions.append(singleinstructions)
            rotation_gates = ["ry"]
            entanglement_gates = "cx"
            ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions,
                                   reps=num_layers * 2, insert_barriers=True, skip_final_rotation_layer=True))
        if layering:
            for layers in range(num_layers):
                alternating_layer = True
                instructions = []
                singleinstructions = []
                auxlist = []
                for i in range(num_qubits):
                    auxlist.append(i)
                auxlist.reverse()
                auxlist2 = rotate(auxlist, -1)
                for j in range(num_qubits):
                    singleinstructions.append((auxlist[j], auxlist2[j]))
                instructions.append(singleinstructions)
                auxlist.reverse()
                auxlist3 = rotate(auxlist, -1)
                auxlist4 = rotate(auxlist, -2)
                singleinstructions = []

                for k in range(num_qubits):
                    singleinstructions.append((auxlist3[k], auxlist4[k]))
                instructions.append(singleinstructions)
                rotation_gates = ["ry"]
                entanglement_gates = "cx"
                ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions,
                                       reps=(layers + 1) * 2, insert_barriers=True, skip_final_rotation_layer=True))

    elif number_of_circuit == 16:
        if not layering:
            instructions = []
            i = 0
            j = num_qubits - 1
            for lays in range(int(np.floor(num_qubits / 2))):

                if i + 1 != j:
                    instructions.append((i + 1, i))
                    instructions.append((j, j - 1))
                else:
                    instructions.append((i, i + 1))
                i += 1
                j -= 1

            rotation_gates = ["rx", "rz"]
            entanglement_gates = "crz"
            ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=num_layers,
                                   insert_barriers=True,
                                   skip_final_rotation_layer=True))
        if layering:
            for layers in range(num_layers):
                instructions = []
                i = 0
                j = num_qubits - 1
                for lays in range(int(np.floor(num_qubits / 2))):

                    if i + 1 != j:
                        instructions.append((i + 1, i))
                        instructions.append((j, j - 1))
                    else:
                        instructions.append((i, i + 1))
                    i += 1
                    j -= 1

                rotation_gates = ["rx", "rz"]
                entanglement_gates = "crz"
                ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=layers + 1,
                                       insert_barriers=True, skip_final_rotation_layer=True))

    elif number_of_circuit == 17:
        if not layering:
            instructions = []
            i = 0
            j = num_qubits - 1
            for lays in range(int(np.floor(num_qubits / 2))):

                if i + 1 != j:
                    instructions.append((i + 1, i))
                    instructions.append((j, j - 1))
                else:
                    instructions.append((i, i + 1))
                i += 1
                j -= 1

            rotation_gates = ["rx", "rz"]
            entanglement_gates = "crx"
            ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=num_layers,
                                   insert_barriers=True,
                                   skip_final_rotation_layer=True))
        if layering:
            for layers in range(num_layers):
                instructions = []
                i = 0
                j = num_qubits - 1
                for lays in range(int(np.floor(num_qubits / 2))):

                    if i + 1 != j:
                        instructions.append((i + 1, i))
                        instructions.append((j, j - 1))
                    else:
                        instructions.append((i, i + 1))
                    i += 1
                    j -= 1

                rotation_gates = ["rx", "rz"]
                entanglement_gates = "crx"
                ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=layers + 1,
                                       insert_barriers=True,
                                       skip_final_rotation_layer=True))

    elif number_of_circuit == 18:
        if not layering:
            instructions = []
            auxlist = []
            for i in range(num_qubits):
                auxlist.append(i)
            auxlist = rotate(auxlist, 1)
            for j in reversed(range(0, num_qubits)):
                instructions.append((auxlist[j - 1], auxlist[j]))

            rotation_gates = ["rx", "rz"]
            entanglement_gates = "crz"
            ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=num_layers,
                                   insert_barriers=True, skip_final_rotation_layer=True))
        if layering:
            for layers in range(num_layers):
                instructions = []
                auxlist = []
                for i in range(num_qubits):
                    auxlist.append(i)
                auxlist = rotate(auxlist, 1)
                for j in reversed(range(0, num_qubits)):
                    instructions.append((auxlist[j - 1], auxlist[j]))

                rotation_gates = ["rx", "rz"]
                entanglement_gates = "crz"
                ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=layers + 1,
                                       insert_barriers=True, skip_final_rotation_layer=True))

    elif number_of_circuit == 19:
        if not layering:
            instructions = []
            auxlist = []
            for i in range(num_qubits):
                auxlist.append(i)
            auxlist = rotate(auxlist, 1)
            for j in reversed(range(0, num_qubits)):
                instructions.append((auxlist[j - 1], auxlist[j]))

            rotation_gates = ["rx", "rz"]
            entanglement_gates = "crx"
            ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=num_layers,
                                   insert_barriers=True, skip_final_rotation_layer=True))
        if layering:
            for layers in range(num_layers):
                instructions = []
                auxlist = []
                for i in range(num_qubits):
                    auxlist.append(i)
                auxlist = rotate(auxlist, 1)
                for j in reversed(range(0, num_qubits)):
                    instructions.append((auxlist[j - 1], auxlist[j]))

                rotation_gates = ["rx", "rz"]
                entanglement_gates = "crx"
                ansatz.append(TwoLocal(num_qubits, rotation_gates, entanglement_gates, instructions, reps=layers + 1,
                                       insert_barriers=True, skip_final_rotation_layer=True))

    else:
        raise ValueError("Unkown circuit number")

    if print_circuit:
        print(ansatz[-1].decompose().draw())

    return ansatz, rotation_gates, entanglement_gates, final_layer, alternating_layer
