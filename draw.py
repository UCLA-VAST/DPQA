import matplotlib
import json
import matplotlib.pyplot as plt
from math import exp, log
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('scale', help='small or large', type=str)
parser.add_argument('--dir', help='directory of DPQA output', type=str)
parser.add_argument('--suffix', type=str,
                    help='suffix to the file name.')
args = parser.parse_args()

matplotlib.rcParams.update({'font.size': 7})
matplotlib.rcParams['font.sans-serif'] = "CMU Sans Serif"
matplotlib.rcParams['font.family'] = "sans-serif"

INFID_CZ_PER_QUBIT = 0.005/2  # per Evered et al.

scale = args.scale
if scale == 'small':
    ns = list(range(10, 24, 2))
elif scale == 'large':
    ns = list(range(30, 100, 10))
else:
    raise ValueError('small or large.')


with open(f"results/stats/tket_Ngate.json", 'r') as f:
    tketjson = json.load(f)

with open(f"results/stats/sabre_Ngate.json", 'r') as f:
    sabrejson = json.load(f)

with open(f"results/stats/tbolsq2_Ngate.json", 'r') as f:
    tbolsqjson = json.load(f)

with open("results/stats/dpqa_duration.json") as f:
    dpqajson = json.load(f)

with open("results/stats/dpqa_depth.json") as f:
    dpqa_depthjson = json.load(f)

ticks = []
# for each data set, compute average and standard deviation
tket_avg = []
tket_err = []
sabre_avg = []
sabre_err = []
dpqa_avg = []
tbolsq_avg = []
tbolsq_err = []
dpqa_equiv_avg = []
dpqa_equiv_err = []

for i in ns:  # 2Q gates number figures
    available = [int(key) for key in dpqajson[str(i)].keys()]

    tket_avg.append(sum([tketjson[str(i)][j]
                    for j in available]) / len(available))
    tket_err.append((sum([(tketjson[str(i)][j]-tket_avg[-1])
                    ** 2 for j in available]) / (len(available)-1))**0.5)

    sabre_avg.append(sum([sabrejson[str(i)][j]
                     for j in available]) / len(available))
    sabre_err.append((sum([(sabrejson[str(i)][j]-sabre_avg[-1])
                     ** 2 for j in available]) / (len(available)-1))**0.5)

    if scale == 'small':
        # TB-OLSQ2 data only available for small size graphs
        tbolsq = list(tbolsqjson[str(i)].values())
        tbolsq_avg.append(sum(tbolsq) / len(tbolsq))
        tbolsq_err.append(
            (sum(
                [(n-tbolsq_avg[-1])**2 for n in tbolsq]
            ) / (len(tbolsq)-1))**0.5)

    # since DPQA does not insert SWAP gates, #2Q-gate= #Q*3/2 for 3-reg graphs
    dpqa_avg.append((i // 2) * 3)

    if scale == 'large':
        # dpqa_equiv data only used in figure for large graphs
        dpqa_equiv = [
            i * dpqa_depthjson[str(i)][str(j)] // 2 for j in available]
        dpqa_equiv_avg.append(sum(dpqa_equiv) / len(dpqa_equiv))
        dpqa_equiv_err.append(
            (sum(
                [(n-dpqa_equiv_avg[-1])**2 for n in dpqa_equiv]
            ) / (len(available)-1))**0.5)

    ticks.append(str(i))

if scale == 'large':
    plt.figure(figsize=(4, 2.5))  # unit inch
elif scale == 'small':
    plt.figure(figsize=(4, 3))

plt.errorbar(ns, tket_avg, tket_err, label='planar+t|ket>',
             c='tab:purple', marker='s')
plt.errorbar(ns, sabre_avg, sabre_err,
             label='planar+sabre', c='tab:blue', marker='x')
if scale == 'small':
    plt.errorbar(ns, tbolsq_avg, tbolsq_err,
                 label='planar+TB-OLSQ2', c='tab:orange', marker='d')
plt.errorbar(ns, dpqa_avg, label='OLSQ-DPQA' if scale ==
             'small' else 'OLSQ-DPQA (hybrid)', c='tab:green', marker='*')
if scale == 'large':
    plt.errorbar(
        ns,
        dpqa_equiv_avg,
        dpqa_equiv_err,
        label='OLSQ-DPQA\n(hybrid, global Rydberg)\n#qubits X #stages / 2',
        c='tab:red',
        marker='.')

plt.legend()
plt.grid(True, 'major', 'y')
plt.xlabel('Number of qubits in a 3-regular graph circuit')
plt.ylabel('Number of two-qubit gates')
plt.savefig(f"./results/figures/draw_{scale}_Ngate.pdf", bbox_inches='tight')


print(f'#2Q-gates tket/dpqa: {tket_avg[-1]/dpqa_avg[-1]}')
print(f'#2Q-gates sabre/dpqa: {sabre_avg[-1]/dpqa_avg[-1]}')
if scale == 'small':
    print(f'#2Q-gates tbolsq/dpqa: {tbolsq_avg[-1]/dpqa_avg[-1]}')
if scale == 'large':
    print(f'#2Q-gates sabre/dpqa-equiv: {sabre_avg[-1]/dpqa_equiv_avg[-1]}')
    print('for log-log fitting:')
    print(f'log of size 30 to 100: {[log(i) for i in range(30, 100, 10)]}')
    print(f'log of sabre #2Q-gates: {[log(i) for i in sabre_avg]}')


def geomean(data: list):
    tmp = 1
    for datum in data:
        tmp *= datum
    tmp **= (1/len(data))
    return tmp


if scale == 'small':  # infidelity figure
    ticks = []
    mov_inf_avg = []
    mov_inf_err = []
    for i in ns:
        available = [int(key) for key in dpqajson[str(i)].keys()]
        # infidelity = 1 - exp(- #Q * percentage of duration in coherence time)
        mov_inf_avg.append(sum(
            [
                1-exp(-1.0 * i * dpqajson[str(i)][str(j)]) for j in available
            ]) / len(available))
        mov_inf_err.append((sum(
            [(
                1-exp(-1.0 * i * dpqajson[str(i)][str(j)]) - mov_inf_avg[-1]
            )**2 for j in available
            ]) / (len(available)-1))**0.5)

    plt.figure(figsize=(2, 1.1))  # unit inch
    plt.plot(ns, [1-(1-INFID_CZ_PER_QUBIT)**(3*n)
             for n in ns], label='two-qubit gates', marker='x')
    plt.errorbar(ns, mov_inf_avg, mov_inf_err,
                 label='AOD movements', marker='.')
    tmp = sum([(1-(1-INFID_CZ_PER_QUBIT)**(3*ns[i])) / mov_inf_avg[i]
               for i in range(len(ns))]) / len(ns)
    print(f'infidelity 2q / mov: {tmp}')

    plt.xlabel('Number of qubits in a 3-regular graph circuit', loc='left')
    plt.yscale('log')
    plt.ylabel('Infidelity')
    plt.legend(bbox_to_anchor=(1, 0), loc="lower left")
    plt.savefig(
        f"./results/figures/draw_{scale}_infid.pdf", bbox_inches='tight')

if scale == 'large':  # compiler runtime figure
    # ticks = []
    # depth = []
    # for i in ns:
    #     available = [int(key) for key in dpqa_depthjson[str(i)].keys()]
    #     depth.append([dpqa_depthjson[str(i)][str(j)] for j in available])
    #     ticks.append(str(i))

    # plt.figure(figsize=(3.4, 2.5))  # unit inch
    # plt.boxplot(depth, positions=list(
    #     range(len(dpqa_avg))), sym='r.', widths=0.5)
    # plt.xlabel('Number of qubits in a 3-regular graph circuit')
    # plt.xticks(list(range(7)), ticks)
    # plt.savefig(
    #     f"./results/figures/draw_{scale}_depth.pdf", bbox_inches='tight')

    plt.figure(figsize=(3.4, 2.5))  # unit inch
    with open('./results/stats/dpqa_runtime.json', 'r') as f:
        runtimes = json.load(f)

    dpqa_time = []
    dpqa_tick = list(range(10, 24, 2))
    for n_qubit in dpqa_tick:
        time_size = []
        for i_run in range(10):
            if str(i_run) not in runtimes[str(n_qubit)]:
                continue
            time_sec = float(runtimes[str(n_qubit)][str(i_run)])
            time_size.append(time_sec)
        dpqa_time.append(time_size)

        plt.scatter([n_qubit for _ in range(len(time_size))],
                    time_size, c='tab:red', marker='x', s=9)

    hybrid_time = []
    hybrid_tick = list(range(30, 100, 10))
    for n_qubit in hybrid_tick:
        time_size = []
        for i_run in range(10):
            if str(i_run) not in runtimes[str(n_qubit)]:
                continue
            time_size.append(float(runtimes[str(n_qubit)][str(i_run)]))
        hybrid_time.append(time_size)
        plt.scatter([n_qubit for _ in range(len(time_size))],
                    time_size, c='tab:blue', marker='.')

    plt.plot([], c='tab:red', label='optimal')
    plt.plot([], c='tab:blue', label='hybrid')
    plt.legend()
    ticks = list(range(10, 100, 10))
    plt.xticks(ticks, ticks)
    plt.xlim([0, 100])
    plt.yscale('log')
    plt.xlabel('Number of qubits in a 3-regular graph circuit')
    plt.ylabel('Compiler runtime in seconds')
    plt.savefig(f"./results/figures/hybrid_original_time.pdf",
                bbox_inches='tight')
