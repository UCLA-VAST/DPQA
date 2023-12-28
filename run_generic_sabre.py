import os
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import (
    SabreLayout,
    SabreSwap,
)
from qiskit.transpiler import PassManager, CouplingMap
import json

def grid10():
    coupling = []
    for i in range(10):
        for j in range(10):
            if j + 1 in range(10):
                coupling.append([10 * i + j, 10 * i + j + 1])
                coupling.append([10 * i + j + 1, 10 * i + j])
            if i + 1 in range(10):
                coupling.append([10 * i + j, 10 * (i + 1) + j])
                coupling.append([10 * (i + 1) + j, 10 * i + j])
    return coupling

GRID10_COUPLING = [
    [0, 1], [1, 0], [0, 10], [10, 0], [1, 2], [2, 1], [1, 11], [11, 1],
    [2, 3], [3, 2], [2, 12], [12, 2], [3, 4], [4, 3], [3, 13], [13, 3],
    [4, 5], [5, 4], [4, 14], [14, 4], [5, 6], [6, 5], [5, 15], [15, 5],
    [6, 7], [7, 6], [6, 16], [16, 6], [7, 8], [8, 7], [7, 17], [17, 7],
    [8, 9], [9, 8], [8, 18], [18, 8], [9, 19], [19, 9], [10, 11], [11, 10],
    [10, 20], [20, 10], [11, 12], [12, 11], [11, 21], [21, 11], [12, 13],
    [13, 12], [12, 22], [22, 12], [13, 14], [14, 13], [13, 23], [23, 13],
    [14, 15], [15, 14], [14, 24], [24, 14], [15, 16], [16, 15], [15, 25],
    [25, 15], [16, 17], [17, 16], [16, 26], [26, 16], [17, 18], [18, 17],
    [17, 27], [27, 17], [18, 19], [19, 18], [18, 28], [28, 18], [19, 29],
    [29, 19], [20, 21], [21, 20], [20, 30], [30, 20], [21, 22], [22, 21],
    [21, 31], [31, 21], [22, 23], [23, 22], [22, 32], [32, 22], [23, 24],
    [24, 23], [23, 33], [33, 23], [24, 25], [25, 24], [24, 34], [34, 24],
    [25, 26], [26, 25], [25, 35], [35, 25], [26, 27], [27, 26], [26, 36],
    [36, 26], [27, 28], [28, 27], [27, 37], [37, 27], [28, 29], [29, 28],
    [28, 38], [38, 28], [29, 39], [39, 29], [30, 31], [31, 30], [30, 40],
    [40, 30], [31, 32], [32, 31], [31, 41], [41, 31], [32, 33], [33, 32],
    [32, 42], [42, 32], [33, 34], [34, 33], [33, 43], [43, 33], [34, 35],
    [35, 34], [34, 44], [44, 34], [35, 36], [36, 35], [35, 45], [45, 35],
    [36, 37], [37, 36], [36, 46], [46, 36], [37, 38], [38, 37], [37, 47],
    [47, 37], [38, 39], [39, 38], [38, 48], [48, 38], [39, 49], [49, 39],
    [40, 41], [41, 40], [40, 50], [50, 40], [41, 42], [42, 41], [41, 51],
    [51, 41], [42, 43], [43, 42], [42, 52], [52, 42], [43, 44], [44, 43],
    [43, 53], [53, 43], [44, 45], [45, 44], [44, 54], [54, 44], [45, 46],
    [46, 45], [45, 55], [55, 45], [46, 47], [47, 46], [46, 56], [56, 46],
    [47, 48], [48, 47], [47, 57], [57, 47], [48, 49], [49, 48], [48, 58],
    [58, 48], [49, 59], [59, 49], [50, 51], [51, 50], [50, 60], [60, 50],
    [51, 52], [52, 51], [51, 61], [61, 51], [52, 53], [53, 52], [52, 62],
    [62, 52], [53, 54], [54, 53], [53, 63], [63, 53], [54, 55], [55, 54],
    [54, 64], [64, 54], [55, 56], [56, 55], [55, 65], [65, 55], [56, 57],
    [57, 56], [56, 66], [66, 56], [57, 58], [58, 57], [57, 67], [67, 57],
    [58, 59], [59, 58], [58, 68], [68, 58], [59, 69], [69, 59], [60, 61],
    [61, 60], [60, 70], [70, 60], [61, 62], [62, 61], [61, 71], [71, 61],
    [62, 63], [63, 62], [62, 72], [72, 62], [63, 64], [64, 63], [63, 73],
    [73, 63], [64, 65], [65, 64], [64, 74], [74, 64], [65, 66], [66, 65],
    [65, 75], [75, 65], [66, 67], [67, 66], [66, 76], [76, 66], [67, 68],
    [68, 67], [67, 77], [77, 67], [68, 69], [69, 68], [68, 78], [78, 68],
    [69, 79], [79, 69], [70, 71], [71, 70], [70, 80], [80, 70], [71, 72],
    [72, 71], [71, 81], [81, 71], [72, 73], [73, 72], [72, 82], [82, 72],
    [73, 74], [74, 73], [73, 83], [83, 73], [74, 75], [75, 74], [74, 84],
    [84, 74], [75, 76], [76, 75], [75, 85], [85, 75], [76, 77], [77, 76],
    [76, 86], [86, 76], [77, 78], [78, 77], [77, 87], [87, 77], [78, 79],
    [79, 78], [78, 88], [88, 78], [79, 89], [89, 79], [80, 81], [81, 80],
    [80, 90], [90, 80], [81, 82], [82, 81], [81, 91], [91, 81], [82, 83],
    [83, 82], [82, 92], [92, 82], [83, 84], [84, 83], [83, 93], [93, 83],
    [84, 85], [85, 84], [84, 94], [94, 84], [85, 86], [86, 85], [85, 95],
    [95, 85], [86, 87], [87, 86], [86, 96], [96, 86], [87, 88], [88, 87],
    [87, 97], [97, 87], [88, 89], [89, 88], [88, 98], [98, 88], [89, 99],
    [99, 89], [90, 91], [91, 90], [91, 92], [92, 91], [92, 93], [93, 92],
    [93, 94], [94, 93], [94, 95], [95, 94], [95, 96], [96, 95], [96, 97],
    [97, 96], [97, 98], [98, 97], [98, 99], [99, 98]
    ]

def prepare_for_olsq(circuit):
    gates = []
    for gate in circuit.get_instructions('cx'):
        gates.append((gate[1][0].index, gate[1][1].index))
    return gates

def pushLeftDepth(list_gate_qubits,
                  count_program_qubit: int) -> int:
    """calculate the depth of circuit pushing every gate as left as possible.

    Args:
        list_gate_qubits (Sequence[Sequence[int]]):
        count_program_qubit (int):

    Returns:
        int: depth of the ASAP (as left as possible) circuit
    """

    push_forward_depth = [0 for _ in range(count_program_qubit)]
    for qubits in list_gate_qubits:
        if len(qubits) == 1:
            push_forward_depth[qubits[0]] += 1
        else:
            tmp_depth = push_forward_depth[qubits[0]]
            if tmp_depth < push_forward_depth[qubits[1]]:
                tmp_depth = push_forward_depth[qubits[1]]
            push_forward_depth[qubits[1]] = tmp_depth + 1
            push_forward_depth[qubits[0]] = tmp_depth + 1
    return max(push_forward_depth)

def run_sabre(circuit, coupling):
    sabre = PassManager([
        SabreLayout(coupling_map=CouplingMap(couplinglist=coupling)),
        SabreSwap(coupling_map=CouplingMap(couplinglist=coupling)),
    ])
    circuit = sabre.run(circuit)
    counts = circuit.count_ops()
    return counts['cx'] + (3 * counts['swap'] if 'swap' in counts else 0)

if __name__ == "__main__":
    with open("sabre_2q", "w") as f:
        f.write("name, num_2q, opt_depth\n")
    only_2q_gates = {}
    for bench in os.listdir("qasm_benchmarks/"):
        circuit = QuantumCircuit.from_qasm_file("qasm_benchmarks/" + bench)
        gates = prepare_for_olsq(circuit)
        only_2q_gates["qasm_benchmarks/" + bench] = gates
        dependency = pushLeftDepth(gates, circuit.width())
        num_2q = run_sabre(circuit, GRID10_COUPLING)
        with open("sabre_2q", "a") as f:
            f.write(f"{bench.split('.')[0]}, {num_2q}, {dependency}\n")
    with open("only_2q.json", "w") as f:
        json.dump(only_2q_gates, f)