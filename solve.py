from typing import Dict
from z3 import Int, Bool, sat, And, Implies, Solver, Not, is_true, Or, is_true, Then
from networkx import max_weight_matching, Graph
import time
import json

PYSAT_ENCODING = 2


def collisionExtract(list_gate_qubits):
    """Extract collision relations between the gates,
    If two gates g_1 and g_2 both acts on a qubit (at different time),
    we say that g_1 and g_2 collide on that qubit, which means that
    (1,2) will be in collision list.
    Args:
        list_gate_qubits: a list of gates in OLSQ IR

    Returns:
        list_collision: a list of collisions between the gates
    """

    list_collision = list()
    # We sweep through all the gates.  For each gate, we sweep through all the
    # gates after it, if they both act on some qubit, append them in the list.
    for g in range(len(list_gate_qubits)):
        for gg in range(g + 1, len(list_gate_qubits)):

            if list_gate_qubits[g][0] == list_gate_qubits[gg][0]:
                list_collision.append((g, gg))

            if len(list_gate_qubits[gg]) == 2:
                if list_gate_qubits[g][0] == list_gate_qubits[gg][1]:
                    list_collision.append((g, gg))

            if len(list_gate_qubits[g]) == 2:
                if list_gate_qubits[g][1] == list_gate_qubits[gg][0]:
                    list_collision.append((g, gg))
                if len(list_gate_qubits[gg]) == 2:
                    if list_gate_qubits[g][1] == list_gate_qubits[gg][1]:
                        list_collision.append((g, gg))

    return tuple(list_collision)


def maxDegree(list_gate_qubits, count_program_qubit):
    cnt = [0 for _ in range(count_program_qubit)]
    for g in list_gate_qubits:
        cnt[g[0]] += 1
        if len(g) == 2:
            cnt[g[1]] += 1
    return max(cnt)


def dependencyExtract(list_gate_qubits, count_program_qubit: int):
    """Extract dependency relations between the gates.
    If two gates g_1 and g_2 both acts on a qubit *and there is no gate
    between g_1 and g_2 that act on this qubit*, we then say that
    g2 depends on g1, which means that (1,2) will be in dependency list.
    Args:
        list_gate_qubits: a list of gates in OLSQ IR
        count_program_qubit: the number of logical/program qubit

    Returns:
        list_dependency: a list of dependency between the gates
    """

    list_dependency = []
    list_last_gate = [-1 for i in range(count_program_qubit)]
    # list_last_gate records the latest gate that acts on each qubit.
    # When we sweep through all the gates, this list is updated and the
    # dependencies induced by the update is noted.
    for i, qubits in enumerate(list_gate_qubits):

        if list_last_gate[qubits[0]] >= 0:
            list_dependency.append((list_last_gate[qubits[0]], i))
        list_last_gate[qubits[0]] = i

        if len(qubits) == 2:
            if list_last_gate[qubits[1]] >= 0:
                list_dependency.append((list_last_gate[qubits[1]], i))
            list_last_gate[qubits[1]] = i

    return tuple(list_dependency)


def pushLeftDepth(list_gate_qubits, count_program_qubit):
    push_forward_depth = [0 for i in range(count_program_qubit)]
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


class DPQA:

    def __init__(self, ifOpt=False, dir: str = None):
        self.dir = dir
        self.n_t = 1
        self.n_q = 0
        self.all_commutable = False
        self.satisfiable = False
        self.ifOpt = ifOpt
        self.all_aod = False
        self.no_transfer = False
        self.pure_graph = False
        self.result_json = {}
        self.result_json['prefix'] = ''
        self.result_json['layers'] = []
        self.aod_q_tot = []
        self.slm_q_tot = []
        self.row_per_site = 3
        self.runtimes = {}
        self.cardenc = "pysat"
        self.optimal_ratio = None

    def set_n_q_limit(self, n_q_limit):
        self.n_q_limit = n_q_limit

    def setOptimalRatio(self, ratio):
        self.optimal_ratio = ratio

    def setArchitecture(self, bounds):
        # bounds = [left of X, right of X, top of Y, bottom of Y]
        self.aod_l, self.aod_r, self.aod_d, self.aod_u = bounds[:4]
        if len(bounds) == 8:
            self.coord_l, self.coord_r, self.coord_d, self.coord_u = bounds[4:]
        else:  # AOD and SLM bounds are the same
            self.coord_l, self.coord_r, self.coord_d, self.coord_u = bounds[:4]

    def setProgram(self, program, nqubit=None):
        # assume program is a iterable of pairs of qubits in CZ gate
        # assume that the qubit indices used are consecutively 0, 1, ...
        self.n_g = len(program)
        tmp = [(min(pair), max(pair)) for pair in program]
        self.g_q = tuple(tmp)
        self.g_s = tuple(['CZ' for _ in range(self.n_g)])
        if not nqubit:
            for gate in program:
                self.n_q = max(gate[0], self.n_q)
                self.n_q = max(gate[1], self.n_q)
            self.n_q += 1
        else:
            self.n_q = nqubit
        self.dependencies = dependencyExtract(self.g_q, self.n_q)
        self.n_t = pushLeftDepth(self.g_q, self.n_q)

        # for graph state circuit
        self.gate_index = {}
        for i in range(self.n_q):
            for j in range(i+1, self.n_q):
                self.gate_index[(i, j)] = -1
        for i in range(self.n_g):
            self.gate_index[self.g_q[i]] = i
        self.gate_index_original = self.gate_index

    def setCommutation(self):
        self.all_commutable = True
        self.collisions = collisionExtract(self.g_q)
        self.n_t = maxDegree(self.g_q, self.n_q)

    def setAOD(self):
        self.all_aod = True

    def setDepth(self, depth):
        self.n_t = depth

    def setDepth(self, depth: Int):
        self.n_t = depth

    def setPureGraph(self):
        self.pure_graph = True

    def setNoTransfer(self):
        self.no_transfer = True

    def setRowSite(self, row_per_site):
        self.row_per_site = row_per_site

    def addMetadata(self, metadata: Dict):
        self.result_json = {**metadata, **(self.result_json)}

    def addPrefix(self, prefix):
        self.result_json["prefix"] = prefix

    def writeSettingJson(self):
        self.result_json['sat'] = self.satisfiable
        self.result_json['n_t'] = self.n_t
        self.result_json['n_q'] = self.n_q
        self.result_json['all_commutable'] = self.all_commutable
        self.result_json['if_Opt'] = self.ifOpt
        self.result_json['all_aod'] = self.all_aod
        self.result_json['no_transfer'] = self.no_transfer
        self.result_json['pure_graph'] = self.pure_graph
        self.result_json['aod_l'] = self.aod_l
        self.result_json['aod_r'] = self.aod_r
        self.result_json['aod_d'] = self.aod_d
        self.result_json['aod_u'] = self.aod_u
        self.result_json['coord_l'] = self.coord_l
        self.result_json['coord_r'] = self.coord_r
        self.result_json['coord_d'] = self.coord_d
        self.result_json['coord_u'] = self.coord_u
        self.result_json['row_per_site'] = self.row_per_site
        self.result_json['n_g'] = self.n_g
        self.result_json['g_q'] = self.g_q
        self.result_json['g_s'] = self.g_s
        # self.result_json['runtimes'] = self.runtimes

    def remove_gates(self, gate_ids):
        new_g_q = []
        new_g_s = []
        for i in range(len(self.g_q)):
            if i not in gate_ids:
                new_g_q.append(self.g_q[i])
                new_g_s.append(self.g_s[i])
        self.g_q = tuple(new_g_q)
        self.g_s = tuple(new_g_s)

        # for graph state circuit
        for q0 in range(self.n_q):
            for q1 in range(q0+1, self.n_q):
                self.gate_index[(q0, q1)] = -1
        for g in range(len(self.g_q)):
            self.gate_index[self.g_q[g]] = g
        self.collisions = collisionExtract(self.g_q)

    def solver_init(self, num_stage: int = 2):
        # variables
        a = [[Bool(f"a_q{q}_t{t}") for t in range(num_stage)]
             for q in range(self.n_q)]
        # for col and row, the data does not matter if atom in SLM
        c = [[Int(f"c_q{q}_t{t}") for t in range(num_stage)]
             for q in range(self.n_q)]
        r = [[Int(f"r_q{q}_t{t}") for t in range(num_stage)]
             for q in range(self.n_q)]
        x = [[Int(f"x_q{q}_t{t}") for t in range(num_stage)]
             for q in range(self.n_q)]
        y = [[Int(f"y_q{q}_t{t}") for t in range(num_stage)]
             for q in range(self.n_q)]

        (self.dpqa) = Solver()
        if self.cardenc == "z3atleast":
            (self.dpqa) = Then('simplify', 'solve-eqs',
                               'card2bv', 'bit-blast', 'aig', 'sat').solver()

        # if all are in AOD
        if self.all_aod:
            for q in range(self.n_q):
                for s in range(num_stage):
                    (self.dpqa).add(a[q][s])

        # if atom transfer is not allowed
        if self.no_transfer:
            for q in range(self.n_q):
                for s in range(1, num_stage):
                    (self.dpqa).add(a[q][s] == a[q][0])

        # bounds
        for q in range(self.n_q):
            for s in range(1, num_stage):
                # starting from s=1 since the values with s=0 are loaded
                (self.dpqa).add(Implies(Not(a[q][s]), x[q][s] >= self.coord_l))
                (self.dpqa).add(Implies(Not(a[q][s]), x[q][s] < self.coord_r))
                (self.dpqa).add(Implies(Not(a[q][s]), y[q][s] >= self.coord_d))
                (self.dpqa).add(Implies(Not(a[q][s]), y[q][s] < self.coord_u))
                (self.dpqa).add(Implies(a[q][s], x[q][s] >= self.coord_l))
                (self.dpqa).add(Implies(a[q][s], x[q][s] < self.coord_r))
                (self.dpqa).add(Implies(a[q][s], y[q][s] >= self.coord_d))
                (self.dpqa).add(Implies(a[q][s], y[q][s] < self.coord_u))
            for s in range(num_stage):
                # starting from s=0 since the solver finds these values
                (self.dpqa).add(c[q][s] >= 0)
                (self.dpqa).add(c[q][s] < self.aod_r)
                (self.dpqa).add(r[q][s] >= 0)
                (self.dpqa).add(r[q][s] < self.aod_u)

        # SLM sites are fixed
        for q in range(self.n_q):
            for s in range(num_stage-1):
                (self.dpqa).add(Implies(Not(a[q][s]), x[q][s] == x[q][s+1]))
                (self.dpqa).add(Implies(Not(a[q][s]), y[q][s] == y[q][s+1]))

        # AOD columns/rows are moved together
        for q in range(self.n_q):
            for s in range(num_stage-1):
                (self.dpqa).add(Implies(a[q][s], c[q][s+1] == c[q][s]))
                (self.dpqa).add(Implies(a[q][s], r[q][s+1] == r[q][s]))
        for q0 in range(self.n_q):
            for q1 in range(q0+1, self.n_q):
                for s in range(num_stage-1):
                    (self.dpqa).add(Implies(
                        And(a[q0][s], a[q1][s], c[q0][s] == c[q1][s]), x[q0][s+1] == x[q1][s+1]))
                    (self.dpqa).add(Implies(
                        And(a[q0][s], a[q1][s], r[q0][s] == r[q1][s]), y[q0][s+1] == y[q1][s+1]))

        # AOD columns/rows cannot move across each other
        for q in range(self.n_q):
            for qq in range(self.n_q):
                for s in range(num_stage-1):
                    if q != qq:
                        (self.dpqa).add(Implies(And(a[q][s], a[qq][s], c[q][s] < c[qq][s]),
                                                x[q][s+1] <= x[qq][s+1]))
                        (self.dpqa).add(Implies(And(a[q][s], a[qq][s], r[q][s] < r[qq][s]),
                                                y[q][s+1] <= y[qq][s+1]))

        # row/col constraints when atom transfer from SLM to AOD
        for q in range(self.n_q):
            for qq in range(self.n_q):
                for s in range(num_stage):
                    if q != qq:
                        (self.dpqa).add(Implies(And(a[q][s], a[qq][s], x[q][s] < x[qq][s]),
                                                c[q][s] < c[qq][s]))
                        (self.dpqa).add(Implies(And(a[q][s], a[qq][s], y[q][s] < y[qq][s]),
                                                r[q][s] < r[qq][s]))

        # not too many AOD columns/rows can be together, default 3
        for q in range(self.n_q):
            for qq in range(self.n_q):
                for s in range(num_stage-1):
                    if q != qq:
                        (self.dpqa).add(Implies(And(a[q][s], a[qq][s], c[q][s]-c[qq][s] > self.row_per_site - 1),
                                                x[q][s+1] > x[qq][s+1]))
                        (self.dpqa).add(Implies(And(a[q][s], a[qq][s], r[q][s]-r[qq][s] > self.row_per_site - 1),
                                                y[q][s+1] > y[qq][s+1]))

        # not too many AOD columns/rows can be together, default 3, for initial stage
        for q in range(self.n_q):
            for qq in range(self.n_q):
                if q != qq:
                    (self.dpqa).add(Implies(And(a[q][0], a[qq][0], c[q][0]-c[qq][0] > self.row_per_site - 1),
                                            x[q][0] > x[qq][0]))
                    (self.dpqa).add(Implies(And(a[q][0], a[qq][0], r[q][0]-r[qq][0] > self.row_per_site - 1),
                                            y[q][0] > y[qq][0]))

        if self.pure_graph:
            # bound number of atoms in each site, needed if not double counting
            for q0 in range(self.n_q):
                for q1 in range(q0+1, self.n_q):
                    for s in range(num_stage):
                        (self.dpqa).add(Implies(And(a[q0][s], a[q1][s]), Or(c[q0][s] != c[q1][s],
                                                                            r[q0][s] != r[q1][s])))
                        (self.dpqa).add(Implies(And(Not(a[q0][s]), Not(a[q1][s])), Or(x[q0][s] != x[q1][s],
                                                                                      y[q0][s] != y[q1][s])))

        # no atom transfer if two atoms meet
        for q0 in range(self.n_q):
            for q1 in range(q0+1, self.n_q):
                for s in range(1, num_stage):
                    (self.dpqa).add(Implies(And(x[q0][s] == x[q1][s], y[q0][s] == y[q1][s]), And(
                        a[q0][s] == a[q0][s-1], a[q1][s] == a[q1][s-1])))

        # for q in range(self.n_q):
        #     (self.dpqa).add(a[q][num_stage-1] == a[q][num_stage-2])

        return a, c, r, x, y

    def add_gate_constraints(self, num_stage: int, c, r, x, y):
        num_gate = len(self.g_q)
        t = [Int(f"t_g{g}") for g in range(num_gate)]

        if len(self.result_json['layers']) > 0:
            vars = self.result_json['layers'][-1]['qubits']
            for q in range(self.n_q):
                # load location info
                if 'x' in vars[q]:
                    (self.dpqa).add(x[q][0] == vars[q]['x'])
                if 'y' in vars[q]:
                    (self.dpqa).add(y[q][0] == vars[q]['y'])
            # virtually putting everything down to acillary SLMs
            # let solver pick some qubits to AOD, so we don't set a_q,0
            # we also don't set c_q,0 and r_q,0, but enforce ordering when
            # two qubits are both in AOD last round, i.e., don't swap
            for q0 in range(self.n_q):
                for q1 in range(q0+1, self.n_q):
                    if vars[q0]['a'] == 1 and vars[q1]['a'] == 1:
                        if vars[q0]['x'] == vars[q1]['x']:
                            if vars[q0]['c'] < vars[q1]['c']:
                                (self.dpqa).add(c[q0][0] < c[q1][0])
                            if vars[q0]['c'] > vars[q1]['c']:
                                (self.dpqa).add(c[q0][0] > c[q1][0])
                            if vars[q0]['c'] == vars[q1]['c']:
                                (self.dpqa).add(c[q0][0] == c[q1][0])
                        if vars[q0]['y'] == vars[q1]['y']:
                            if vars[q0]['r'] < vars[q1]['r']:
                                (self.dpqa).add(r[q0][0] < r[q1][0])
                            if vars[q0]['r'] > vars[q1]['r']:
                                (self.dpqa).add(r[q0][0] > r[q1][0])
                            if vars[q0]['r'] == vars[q1]['r']:
                                (self.dpqa).add(r[q0][0] == r[q1][0])

        for g in range(num_gate):
            (self.dpqa).add(t[g] < num_stage)
            (self.dpqa).add(t[g] >= 0)

        # dependency/collision
        if self.all_commutable:
            for collision in self.collisions:
                (self.dpqa).add(Or(t[collision[0]] != t[collision[1]],
                                   t[collision[0]] == 0, t[collision[0]] == 0))
                # stage0 is the 'trash can' for gates where we don't impose
                # connectivity, so if a gate is in stage0, we can ignore all
                # its collisions. If both gates are not in stage0, we impose.
        else:
            raise ValueError(
                "Cannot process circuits that are not fully commutable")
            # for dependency in self.dependencies:
            #     (self.dpqa).add(t[dependency[0]] < t[dependency[1]])

        # connectivity
        for g in range(num_gate):
            for s in range(1, num_stage):
                if len(self.g_q[g]) == 2:
                    q0 = self.g_q[g][0]
                    q1 = self.g_q[g][1]
                    (self.dpqa).add(Implies(t[g] == s, x[q0][s] == x[q1][s]))
                    (self.dpqa).add(Implies(t[g] == s, y[q0][s] == y[q1][s]))

        if self.pure_graph:
            # global CZ switch (only works for graph state circuit)
            for q0 in range(self.n_q):
                for q1 in range(q0+1, self.n_q):
                    for s in range(1, num_stage):
                        if self.gate_index[(q0, q1)] == -1:
                            (self.dpqa).add(
                                Or(x[q0][s] != x[q1][s], y[q0][s] != y[q1][s]))
                        else:
                            (self.dpqa).add(Implies(And(x[q0][s] == x[q1][s],
                                                        y[q0][s] == y[q1][s]), t[self.gate_index[(q0, q1)]] == s))
        else:
            raise ValueError("Cannot process non pure graph circuits now")
            # # global CZ switch
            # (self.dpqa).add(self.n_g == sum([If(And(x[i][k] == x[j][k],
            #                                  y[i][k] == y[j][k]), 1, 0) for i in range(self.n_q)
            #                           for j in range(i+1, self.n_q) for k in range(self.n_t)]))

        return t

    def add_bound_gate(self, bound_gate: int, num_stage: int, t):
        method = self.cardenc
        num_gate = len(self.g_q)
        if method == "summation":
            # (self.dpqa).add(sum([If(t[g] == s, 1, 0) for g in range(num_gate)
            #                     for s in range(1, num_stage)]) >= bound_gate)
            raise ValueError()
        elif method == "z3atleast":
            # tmp = [(t[g] == s) for g in range(num_gate)
            #        for s in range(1, num_stage)]
            # (self.dpqa).add(AtLeast(*tmp, bound_gate))
            raise ValueError()
        elif method == "pysat":
            from pysat.card import CardEnc
            offset = num_gate - 1
            numvar = (num_stage-1)*num_gate+1
            ancillary = {}
            cnf = CardEnc.atleast(lits=range(
                1, numvar), top_id=numvar-1, bound=bound_gate, encoding=PYSAT_ENCODING)
            for conj in cnf:
                or_list = []
                for i in conj:
                    val = abs(i)
                    idx = val + offset
                    if i in range(1, numvar):
                        or_list.append(t[idx % num_gate] == (idx // num_gate))
                    elif i in range(-numvar+1, 0):
                        or_list.append(
                            Not(t[idx % num_gate] == (idx // num_gate)))
                    else:
                        if val not in ancillary.keys():
                            ancillary[val] = Bool("anx_{}".format(val))
                        if i < 0:
                            or_list.append(Not(ancillary[val]))
                        else:
                            or_list.append(ancillary[val])
                (self.dpqa).add(Or(*or_list))
        else:
            raise ValueError("cardinality method unknown")

    def print_partial(self, num_stage: int, a, c, r, x, y, t):
        model = (self.dpqa).model()
        layers = self.result_json['layers']
        old_len_layers = len(layers)

        # print and store stage 1, ...
        for s in range(num_stage):
            real_s = old_len_layers+s-1
            print(f"        stage {real_s}:")
            layer = {}
            layer['qubits'] = []
            # aod_q = []
            # slm_q = []
            for q in range(self.n_q):
                layer['qubits'].append({
                    'id': q,
                    'a': 1 if is_true(model[a[q][s]]) else 0,
                    'x': model[x[q][s]].as_long(),
                    'y': model[y[q][s]].as_long(),
                    'c': model[c[q][s]].as_long(),
                    'r': model[r[q][s]].as_long()})
                if is_true(model[a[q][s]]):
                    # aod_q.append(q)
                    print(f"        q_{q} is at ({model[x[q][s]].as_long()}, " +
                          f"{model[y[q][s]].as_long()})" + f" AOD c_{model[c[q][s]].as_long()}, r_{model[r[q][s]].as_long()}")
                    # if real_s > 0 and (q not in self.aod_q_tot[real_s-1]):
                    #     print("        !!!changed")
                    #     layer['qubits'][q]['transfer'] = True
                else:
                    # slm_q.append(q)
                    print(f"        q_{q} is at ({model[x[q][s]].as_long()}, " +
                          f"{model[y[q][s]].as_long()}) SLM" + f" c_{model[c[q][s]].as_long()}, r_{model[r[q][s]].as_long()}")
                    # if real_s > 0 and (q not in self.slm_q_tot[real_s-1]):
                    #     print("        !!!changed")
                    #     layer['qubits'][q]['transfer'] = True
            # self.aod_q_tot.append(aod_q)
            # self.slm_q_tot.append(slm_q)

            if s == 0 and self.result_json['layers']:
                for q in range(self.n_q):
                    self.result_json['layers'][-1]['qubits'][q]['a'] = 1 if is_true(
                        model[a[q][s]]) else 0
                    self.result_json['layers'][-1]['qubits'][q]['c'] = model[c[q][s]].as_long()
                    self.result_json['layers'][-1]['qubits'][q]['r'] = model[r[q][s]].as_long()

            if s > 0:
                layer['gates'] = []
                gates_done = []
                for g in range(len(self.g_q)):
                    if model[t[g]].as_long() == s:
                        print(
                            f"        CZ(q_{self.g_q[g][0]}, q_{self.g_q[g][1]})")
                        layer['gates'].append(
                            {'id': self.gate_index_original[(self.g_q[g][0], self.g_q[g][1])], 'q0': self.g_q[g][0], 'q1': self.g_q[g][1]})
                        gates_done.append(g)

                self.result_json['layers'].append(layer)

        self.remove_gates(gates_done)
        # print((self.dpqa).statistics())
        # return True

    def hybrid_strategy(self):
        if self.optimal_ratio == None:
            self.setOptimalRatio(1 if self.n_q < 30 else 0.05)
        if self.optimal_ratio == 1 and self.n_q < 30:
            self.setNoTransfer()

    def solve(self):

        total_g_q = len(self.g_q)
        self.writeSettingJson()
        self.t_s = time.time()
        # self.result_json['layers'] = []
        t_curr = 1
        step = 1
        a, c, r, x, y = self.solver_init(step+1)
        check_time = 0
        while len(self.g_q) > self.optimal_ratio * total_g_q:
            print(f"gate batch {t_curr}")

            (self.dpqa).push()  # gate related constraints
            t = self.add_gate_constraints(step+1, c, r, x, y)

            G = Graph()
            G.add_edges_from(self.g_q)
            bound_gate = len(max_weight_matching(G))

            (self.dpqa).push()  # gate bound
            self.add_bound_gate(bound_gate, step+1, t)

            now = time.time()
            solved_batch_gates = True if (self.dpqa).check() == sat else False
            check_time += time.time() - now

            while not solved_batch_gates:
                print(f"    no solution, bound_gate={bound_gate} too large")
                (self.dpqa).pop()  # pop to reduce gate bound
                bound_gate -= 1

                (self.dpqa).push()  # new gate bound
                self.add_bound_gate(bound_gate, step+1, t)

                now = time.time()
                solved_batch_gates = True if (
                    self.dpqa).check() == sat else False
                check_time += time.time() - now

            print(
                f"    found solution with {bound_gate} gates in {step} step")
            self.print_partial(step+1, a, c, r, x, y, t)
            (self.dpqa).pop()  # pop the card constraints for solved batch
            t_curr += 1
            (self.dpqa).pop()  # pop to change gate batch

        print(f"final {len(self.g_q) / total_g_q}")
        bound_gate = len(self.g_q)
        (self.dpqa).push()  # gate related constraints
        t = self.add_gate_constraints(step+1, c, r, x, y)

        (self.dpqa).push()  # gate bound
        self.add_bound_gate(bound_gate, step+1, t)

        now = time.time()
        solved_batch_gates = True if (self.dpqa).check() == sat else False
        check_time += time.time() - now

        while not solved_batch_gates:
            print(f"    no solution, step={step} too small")
            (self.dpqa).pop()  # pop to reduce gate bound
            step += 1

            a, c, r, x, y = self.solver_init(step+1)
            (self.dpqa).push()  # gate bound
            t = self.add_gate_constraints(step+1, c, r, x, y)
            print(self.g_q)

            (self.dpqa).push()  # new gate bound
            self.add_bound_gate(bound_gate, step+1, t)

            now = time.time()
            solved_batch_gates = True if (
                self.dpqa).check() == sat else False
            check_time += time.time() - now

        print(
            f"    found solution with {bound_gate} gates in {step} step")
        self.print_partial(step+1, a, c, r, x, y, t)

        self.result_json['timestamp'] = str(time.time())
        self.result_json['duration'] = str(time.time() - self.t_s)
        self.result_json['check_time'] = str(check_time)
        self.result_json['n_t'] = len(self.result_json['layers'])
        print(self.result_json['duration'])

        if not self.dir:
            self.dir = "./results/"
        with open(self.dir + f"{self.result_json['prefix']}.json", 'w') as file_object:
            json.dump(self.result_json, file_object)
        file_object.close()
