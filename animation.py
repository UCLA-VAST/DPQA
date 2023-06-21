from matplotlib.animation import FFMpegWriter, FuncAnimation
import matplotlib.pyplot as plt
import json
import matplotlib
import networkx as nx
import argparse


# physics constants
R_B = 6  # rydberg range
AOD_SEP = 2  # min AOD separation
RYD_SEP = 15  # sufficient distance to avoid Rydberg
SITE_SLMS = 2  # number of SLMs in a site
SLM_SEP = AOD_SEP  # separation of SLMs inside a site
SITE_WIDTH = 4  # total width of SLMs in a site
X_SITE_SEP = RYD_SEP + SITE_WIDTH  # separation of sites in X direction
Y_SITE_SEP = RYD_SEP  # separation of sites in Y direction
X_WAIT_OFFSET = -1  # x offset when rows finish moving for a stage
V_MOV = 0.5  # microns per micro seconds

# padding of the figure
X_LOW_PAD = 2*AOD_SEP
Y_LOW_PAD = 4*AOD_SEP
X_HIGH_PAD = 2*AOD_SEP
Y_HIGH_PAD = 4*AOD_SEP

# constants for animation
FPS = 24  # frames per second
INIT_FRM = 24  # initial empty frames
PT_MICRON = 8  # scaling factor: points per micron
MUS_PER_FRM = 8  # microseconds per frame
T_RYDBERG = MUS_PER_FRM*8  # microseconds for Rydberg
T_ACTIVATE = MUS_PER_FRM  # microseconds for (de)activating AOD


# class for physical entities: qubits and AOD rows/cols

class Qubit():
    def __init__(self, id):
        self.array = 'SLM'
        self.id = id
        self.c = -1
        self.r = -1
        self.x = -X_LOW_PAD-1
        self.y = -Y_LOW_PAD-1


class Row():
    def __init__(self, id):
        self.id = id
        self.active = False
        self.y = -Y_LOW_PAD-1


class Col():
    def __init__(self, id):
        self.id = id
        self.active = False
        self.x = -X_LOW_PAD-1


# class for basic instruction: Init, Move, Activate, Deactivate, Rydberg

# todo: add class Raman (single-qubit gates)
# todo: add a parent class 'Inst'

class Init():
    def __init__(self,
                 s: int,
                 col_objs: list,
                 row_objs: list,
                 qubit_objs: list,
                 slm_qubit_idx: list = [],
                 slm_qubit_xys: list = [],
                 aod_qubit_idx: list = [],
                 aod_qubit_crs: list = [],
                 aod_col_act_idx: list = [],
                 aod_col_xs: list = [],
                 aod_row_act_idx: list = [],
                 aod_row_ys: list = []):
        self.type = 'Init'
        self.name = self.type
        self.duration = INIT_FRM
        self.all_slms = []
        self.verify(slm_qubit_idx,
                    slm_qubit_xys,
                    aod_qubit_idx,
                    aod_qubit_crs,
                    aod_col_act_idx,
                    aod_col_xs,
                    aod_row_act_idx,
                    aod_row_ys)

        for i, q_id in enumerate(slm_qubit_idx):
            qubit_objs[q_id].array = 'SLM'
            (qubit_objs[q_id].x, qubit_objs[q_id].y) = slm_qubit_xys[i]
            self.all_slms.append(slm_qubit_xys[i])

        for i, col_id in enumerate(aod_col_act_idx):
            col_objs[col_id].activate = True
            col_objs[col_id].x = aod_col_xs[i]

        for i, row_id in enumerate(aod_col_act_idx):
            row_objs[row_id].activate = True
            row_objs[row_id].y = aod_row_ys[i]

        for i, q_id in enumerate(aod_qubit_idx):
            qubit_objs[q_id].array = 'AOD'
            (qubit_objs[q_id].c, qubit_objs[q_id].r) = aod_qubit_crs[i]
            qubit_objs[q_id].x = col_objs[qubit_objs[q_id].c].x
            qubit_objs[q_id].y = row_objs[qubit_objs[q_id].r].y

        self.code = {'type': self.type,
                     'name': self.name,
                     'duration': self.duration,
                     'slm_qubit_idx': slm_qubit_idx,
                     'slm_qubit_xys': slm_qubit_xys,
                     'aod_qubit_idx': aod_qubit_idx,
                     'aod_qubit_crs': aod_qubit_crs,
                     'aod_col_act_idx': aod_col_act_idx,
                     'aod_col_xs': aod_col_xs,
                     'aod_row_act_idx': aod_row_act_idx,
                     'aod_row_ys': aod_row_ys,
                     'state': self.state(col_objs, row_objs, qubit_objs)}

    def add_slms(self, slms: list):
        for slm in slms:
            if slm not in self.all_slms:
                self.all_slms.append(slm)

    def state(self, col_objs, row_objs, qubit_objs):
        curr = {}
        curr['qubits'] = [{'id': q.id, 'x': q.x, 'y': q.y,
                           'array': q.array, 'c': q.c, 'r': q.r} for q in qubit_objs]
        curr['cols'] = [{'id': c.id, 'active': c.active, 'x': c.x}
                        for c in col_objs]
        curr['rows'] = [{'id': r.id, 'active': r.active, 'y': r.y}
                        for r in row_objs]
        return curr

    def verify(self,
               slm_qubit_idx: list = [],
               slm_qubit_xys: list = [],
               aod_qubit_idx: list = [],
               aod_qubit_crs: list = [],
               aod_col_act_idx: list = [],
               aod_col_xs: list = [],
               aod_row_act_idx: list = [],
               aod_row_ys: list = []):
        a = len(slm_qubit_idx)
        b = len(slm_qubit_xys)
        if a != b:
            raise ValueError(
                f'{self.name}: SLM qubit arguments invalid {a} idx, {b} xys.')
        for i in slm_qubit_xys:
            if len(i) != 2:
                raise ValueError(f'{self.name}: SLM qubit xys {i} invalid.')
        for i in range(len(slm_qubit_xys)):
            for j in range(i+1, len(slm_qubit_xys)):
                if slm_qubit_xys[i] == slm_qubit_xys[j]:
                    raise ValueError(
                        f'{self.name}: SLM qubits {slm_qubit_idx[i]} and {slm_qubit_idx[j]} xys are the same.')

        # todo: if not all qubits in SLM, add checks

    def emit_full(self):
        self.code['all_slms'] = self.all_slms
        return self.code

    def emit(self):
        raw_code = {}
        for k in ['type', 'name', 'duration', 'slm_qubit_idx', 'slm_qubit_xys',
                  'aod_qubit_idx', 'aod_qubit_crs', 'aod_col_act_idx',
                  'aod_col_xs', 'aod_row_act_idx', 'aod_row_ys', 'n_q',
                  'x_high', 'y_high', 'c_high', 'r_high']:
            raw_code[k] = self.code[k]
        return raw_code


class Move():
    def __init__(self,
                 s: int,
                 col_objs: list,
                 row_objs: list,
                 qubit_objs: list,
                 col_idx: list = [],
                 col_begin: list = [],
                 col_end: list = [],
                 row_idx: list = [],
                 row_begin: list = [],
                 row_end: list = [],
                 prefix: str = ''):
        self.type = 'Move'
        self.stage = s
        self.name = prefix + ':' + self.type
        self.col_idx = col_idx
        self.col_begin = col_begin
        self.col_end = col_end
        self.row_idx = row_idx
        self.row_begin = row_begin
        self.row_end = row_end
        self.verify(col_objs, row_objs)

        self.code = {'type': self.type, 'name': self.name}

        self.code['cols'] = []
        max_distance = 0
        for i in range(len(self.col_idx)):
            distance = abs(col_end[i]-col_begin[i])
            if distance > 0:
                self.code['cols'].append(
                    {'id': col_idx[i],
                     'shift': col_end[i]-col_begin[i],
                     'begin': col_begin[i],
                     'end': col_end[i]})
                col_objs[col_idx[i]].x = col_end[i]
                max_distance = max(max_distance, distance)

        self.code['rows'] = []
        for i in range(len(self.row_idx)):
            distance = abs(row_end[i]-row_begin[i])
            if distance > 0:
                self.code['rows'].append(
                    {'id': row_idx[i],
                     'shift': row_end[i]-row_begin[i],
                     'begin': row_begin[i],
                     'end': row_end[i]})
                row_objs[row_idx[i]].y = row_end[i]
                max_distance = max(max_distance, distance)

        self.code['duration'] = 200*((max_distance/110)**(1/2))

        for qubit_obj in qubit_objs:
            if qubit_obj.array == 'AOD':
                qubit_obj.x = col_objs[qubit_obj.c].x
                qubit_obj.y = row_objs[qubit_obj.r].y

        self.code['state'] = self.state(col_objs, row_objs, qubit_objs)

    def state(self, col_objs, row_objs, qubit_objs):
        curr = {}
        curr['qubits'] = [{'id': q.id, 'x': q.x, 'y': q.y,
                           'array': q.array, 'c': q.c, 'r': q.r} for q in qubit_objs]
        curr['cols'] = [{'id': c.id, 'active': c.active, 'x': c.x}
                        for c in col_objs]
        curr['rows'] = [{'id': r.id, 'active': r.active, 'y': r.y}
                        for r in row_objs]
        return curr

    def verify(self, col_objs, row_objs):
        a = len(self.col_idx)
        b = len(self.col_begin)
        c = len(self.col_end)
        if not (a == b and a == c):
            raise ValueError(
                f'{self.name}: col arguments invalid {a} idx, {b} begin, {c} end.')
        a = len(self.row_idx)
        b = len(self.row_begin)
        c = len(self.row_end)
        if not (a == b and a == c):
            raise ValueError(
                f'{self.name}: row arguments invalid {a} idx, {b} begin, {c} end.')

        activated_col_idx = []
        activated_col_xs = []
        for col_obj in col_objs:
            if col_obj.active:
                if activated_col_idx and col_obj.x < activated_col_xs[-1]+AOD_SEP:
                    raise ValueError(
                        f'{self.name}: col beginning position invalid col {col_obj.id} at x={col_obj.x} while col {activated_col_idx[-1]} at x={activated_col_xs[-1]}.')
                activated_col_idx.append(col_obj.id)
                activated_col_xs.append(col_obj.x)
        for i, moving_col_id in enumerate(self.col_idx):
            if moving_col_id not in activated_col_idx:
                raise ValueError(
                    f'{self.name}: col {moving_col_id} to move is not activated.')
            j = activated_col_idx.index(moving_col_id)
            if self.col_begin[i] != activated_col_xs[j]:
                raise ValueError(
                    f'{self.name}: col {moving_col_id} beginning x not agree.')
            activated_col_xs[j] = self.col_end[i]
        for i in range(1, len(activated_col_xs)):
            if activated_col_xs[i-1]+AOD_SEP > activated_col_xs[i]:
                raise ValueError(
                    f'{self.name}: col ending position invalid col {activated_col_idx[i-1]} at x={activated_col_xs[i-1]} while col {activated_col_idx[i]} at x={activated_col_xs[i]}.')

        activated_row_idx = []
        activated_row_ys = []
        for row_obj in row_objs:
            if row_obj.active:
                if activated_row_idx and row_obj.y < activated_row_ys[-1] + AOD_SEP:
                    raise ValueError(
                        f'{self.name}: row beginning position invalid row {row_obj.id} at y={row_obj.y} while row {activated_row_idx[-1]} at y={activated_row_ys[-1]}.')
                activated_row_idx.append(row_obj.id)
                activated_row_ys.append(row_obj.y)
        for i, moving_row_id in enumerate(self.row_idx):
            if moving_row_id not in activated_row_idx:
                raise ValueError(
                    f'{self.name}: row {moving_row_id} to move is not activated.')
            j = activated_row_idx.index(moving_row_id)
            if self.row_begin[i] != activated_row_ys[j]:
                raise ValueError(
                    f'{self.name}: row {moving_row_id} beginning y not agree.')
            activated_row_ys[j] = self.row_end[i]
        for i in range(1, len(activated_row_ys)):
            if activated_row_ys[i-1]+AOD_SEP > activated_row_ys[i]:
                raise ValueError(
                    f'{self.name}: row ending position invalid row {activated_row_idx[i-1]} at y={activated_row_ys[i-1]} while row {activated_row_idx[i]} at y={activated_row_ys[i]}.')

    def emit_full(self):
        return self.code

    def emit(self):
        raw_code = {'type': self.type}
        raw_code['cols'] = [{k: v for k, v in tmp.items() if k in [
            'id', 'shift']} for tmp in self.code['cols']]
        raw_code['rows'] = [{k: v for k, v in tmp.items() if k in [
            'id', 'shift']} for tmp in self.code['rows']]
        return raw_code


class Activate():
    def __init__(self,
                 s: int,
                 col_objs: list,
                 row_objs: list,
                 qubit_objs: list,
                 col_idx: list = [],
                 col_xs: list = [],
                 row_idx: list = [],
                 row_ys: list = [],
                 pickup_qs: list = [],
                 prefix: str = ''):
        self.stage = s
        self.type = 'Activate'
        self.name = prefix + ':' + self.type
        self.col_idx = col_idx
        self.col_xs = col_xs
        self.row_idx = row_idx
        self.row_ys = row_ys
        self.pickup_qs = pickup_qs
        self.duration = T_ACTIVATE

        self.verify(col_objs, row_objs, qubit_objs)
        self.code = {'type': self.type,
                     'name': self.name,
                     'col_idx': self.col_idx,
                     'col_xs': self.col_xs,
                     'row_idx': self.row_idx,
                     'row_ys': self.row_ys,
                     'pickup_qs': self.pickup_qs,
                     'duration': self.duration}

        for i in range(len(self.col_idx)):
            col_objs[self.col_idx[i]].active = True
            col_objs[self.col_idx[i]].x = self.col_xs[i]
        for i in range(len(self.row_idx)):
            row_objs[self.row_idx[i]].active = True
            row_objs[self.row_idx[i]].y = self.row_ys[i]
        for q_id in self.pickup_qs:
            qubit_objs[q_id].array = 'AOD'
        self.code['state'] = self.state(col_objs, row_objs, qubit_objs)

    def state(self, col_objs, row_objs, qubit_objs):
        curr = {}
        curr['qubits'] = [{'id': q.id, 'x': q.x, 'y': q.y,
                           'array': q.array, 'c': q.c, 'r': q.r} for q in qubit_objs]
        curr['cols'] = [{'id': c.id, 'active': c.active, 'x': c.x}
                        for c in col_objs]
        curr['rows'] = [{'id': r.id, 'active': r.active, 'y': r.y}
                        for r in row_objs]
        return curr

    def verify(self, col_objs: list, row_objs: list, qubit_objs: list):
        a = len(self.col_idx)
        b = len(self.col_xs)
        if a != b:
            raise ValueError(
                f'{self.name}: col arguments invalid {a} idx, {b} xs')
        a = len(self.row_idx)
        b = len(self.row_ys)
        if a != b:
            raise ValueError(
                f'f{self.name}: row arguments invalid {a} idx, {b} ys.')

        for i in range(len(self.col_idx)):
            if col_objs[self.col_idx[i]].active:
                raise ValueError(
                    f'{self.name}: col {self.col_idx[i]} already activated.')
            for j in range(self.col_idx[i]):
                if col_objs[j].active and col_objs[j].x > self.col_xs[i]-AOD_SEP:
                    raise ValueError(
                        f'{self.name}: col {j} at x={col_objs[j].x} is too left for col {self.col_idx[i]} to activate at x={self.col_xs[i]}.')
            for j in range(self.col_idx[i]+1, len(col_objs)):
                if col_objs[j].active and col_objs[j].x-AOD_SEP < self.col_xs[i]:
                    raise ValueError(
                        f'{self.name}: col {j} at x={col_objs[j].x} is too right for col {self.col_idx[i]} to activate at x={self.col_xs[i]}.')
        for i in range(len(self.row_idx)):
            if row_objs[self.row_idx[i]].active:
                raise ValueError(
                    f'{self.name}: row {self.row_idx[i]} already activated.')
            for j in range(self.row_idx[i]):
                if row_objs[j].active and row_objs[j].y > self.row_ys[i]-AOD_SEP:
                    raise ValueError(
                        f'{self.name}: row {j} at y={row_objs[j].y} is too high for row {self.row_idx[i]} to activate at y={self.row_ys[i]}.')
            for j in range(self.row_idx[i]+1, len(row_objs)):
                if row_objs[j].active and row_objs[j].y-AOD_SEP < self.row_ys[i]:
                    raise ValueError(
                        f'{self.name}: row {j} at y={col_objs[j].y} is too low for row {self.row_idx[i]} to activate at y={self.row_ys[i]}.')

        active_xys = []  # the traps that are newly activted by this Activate
        active_xs = [col.x for col in col_objs if col.active]
        active_ys = [row.y for row in row_objs if row.active]
        for x in active_xs:
            for y in self.row_ys:
                active_xys.append((x, y))
        for y in active_ys:
            for x in self.col_xs:
                active_xys.append((x, y))
        for x in self.col_xs:
            for y in self.row_ys:
                active_xys.append((x, y))

        for q_id in range(len(qubit_objs)):
            if q_id in self.pickup_qs:
                if (qubit_objs[q_id].x, qubit_objs[q_id].y) not in active_xys:
                    raise ValueError(
                        f'{self.name}: q {q_id} not picked up by col {qubit_objs[q_id].c} row {qubit_objs[q_id].r} at x={qubit_objs[q_id].x} y={qubit_objs[q_id].y}.')
            else:
                if (qubit_objs[q_id].x, qubit_objs[q_id].y) in active_xys:
                    raise ValueError(
                        f'{self.name}: q {q_id} wrongfully picked up by col {qubit_objs[q_id].c} row {qubit_objs[q_id].r} at x={qubit_objs[q_id].x} y={qubit_objs[q_id].y}.')

    def emit_full(self):
        return self.code

    def emit(self):
        return {k: v for k, v in self.code.items() if k in ['type',
                                                            'col_idx',
                                                            'row_idx']}


class Deactivate():
    def __init__(self,
                 s: int,
                 col_objs: list,
                 row_objs: list,
                 qubit_objs: list,
                 col_idx: list = [],
                 col_xs: list = [],
                 row_idx: list = [],
                 row_ys: list = [],
                 dropoff_qs: list = [],
                 prefix: str = ''):
        self.stage = s
        self.type = 'Deactivate'
        self.name = prefix + ':' + self.type
        self.col_idx = col_idx
        self.col_xs = col_xs
        self.row_idx = row_idx
        self.row_ys = row_ys
        self.dropoff_qs = dropoff_qs
        self.duration = T_ACTIVATE

        self.verify(col_objs, row_objs, qubit_objs)
        self.code = {'type': self.type,
                     'name': self.name,
                     'col_idx': self.col_idx,
                     'col_xs': self.col_xs,
                     'row_idx': self.row_idx,
                     'row_ys': self.row_ys,
                     'dropoff_qs': self.dropoff_qs,
                     'duration': self.duration}

        for i in range(len(self.col_idx)):
            col_objs[self.col_idx[i]].active = False
        for i in range(len(self.row_idx)):
            row_objs[self.row_idx[i]].active = False
        for q_id in self.dropoff_qs:
            qubit_objs[q_id].array = 'SLM'
        self.code['state'] = self.state(col_objs, row_objs, qubit_objs)

    def state(self, col_objs, row_objs, qubit_objs):
        curr = {}
        curr['qubits'] = [{'id': q.id, 'x': q.x, 'y': q.y,
                           'array': q.array, 'c': q.c, 'r': q.r} for q in qubit_objs]
        curr['cols'] = [{'id': c.id, 'active': c.active, 'x': c.x}
                        for c in col_objs]
        curr['rows'] = [{'id': r.id, 'active': r.active, 'y': r.y}
                        for r in row_objs]
        return curr

    def verify(self, col_objs: list, row_objs: list, qubit_objs: list):
        a = len(self.col_idx)
        b = len(self.col_xs)
        if a != b:
            raise ValueError(
                f'{self.name}: col arguments invalid {a} idx, {b} xs')
        a = len(self.row_idx)
        b = len(self.row_ys)
        if a != b:
            raise ValueError(
                f'f{self.name}: row arguments invalid {a} idx, {b} ys.')

        for i in range(len(self.col_idx)):
            if not col_objs[self.col_idx[i]].active:
                raise ValueError(
                    f'{self.name}: col {self.col_idx[i]} already dectivated.')
            for j in range(self.col_idx[i]):
                if col_objs[j].active and col_objs[j].x > self.col_xs[i]-AOD_SEP:
                    raise ValueError(
                        f'{self.name}: col {j} at x={col_objs[j].x} is too left for col {self.col_idx[i]} to deactivate at x={self.col_xs[i]}.')
            for j in range(self.col_idx[i]+1, len(col_objs)):
                if col_objs[j].active and col_objs[j].x-AOD_SEP < self.col_xs[i]:
                    raise ValueError(
                        f'{self.name}: col {j} at x={col_objs[j].x} is too right for col {self.col_idx[i]} to deactivate at x={self.col_xs[i]}.')
        for i in range(len(self.row_idx)):
            if not row_objs[self.row_idx[i]].active:
                raise ValueError(
                    f'{self.name}: row {self.row_idx[i]} already deactivated.')
            for j in range(self.row_idx[i]):
                if row_objs[j].active and row_objs[j].y > self.row_ys[i]-AOD_SEP:
                    raise ValueError(
                        f'{self.name}: row {j} at y={row_objs[j].y} is too high for row {self.row_idx[i]} to deactivate at y={self.row_ys[i]}.')
            for j in range(self.row_idx[i]+1, len(row_objs)):
                if row_objs[j].active and row_objs[j].y-AOD_SEP < self.row_ys[i]:
                    raise ValueError(
                        f'{self.name}: row {j} at y={col_objs[j].y} is too low for row {self.row_idx[i]} to deactivate at y={self.row_ys[i]}.')

        deactive_xys = []
        active_xs = [col.x for col in col_objs if col.active]
        for x in active_xs:
            for y in self.row_ys:
                deactive_xys.append((x, y))

        for q_id in range(len(qubit_objs)):
            if q_id in self.dropoff_qs:
                if (qubit_objs[q_id].x, qubit_objs[q_id].y) not in deactive_xys:
                    raise ValueError(
                        f'{self.name}: q {q_id} not dropped off from col {qubit_objs[q_id].c} row {qubit_objs[q_id].r} at x={qubit_objs[q_id].x} y={qubit_objs[q_id].y}.')
            elif qubit_objs[q_id].array == 'AOD':
                if (qubit_objs[q_id].x, qubit_objs[q_id].y) in deactive_xys:
                    raise ValueError(
                        f'{self.name}: q {q_id} wrongfully dropped off from col {qubit_objs[q_id].c} row {qubit_objs[q_id].r} at x={qubit_objs[q_id].x} y={qubit_objs[q_id].y}.')

    def emit_full(self):
        return self.code

    def emit(self):
        return {k: v for k, v in self.code.items() if k in ['type',
                                                            'col_idx',
                                                            'row_idx']}


class Rydberg():
    def __init__(self,
                 s: int,
                 col_objs: list,
                 row_objs: list,
                 qubit_objs: list,
                 gates: list):
        self.type = 'Rydberg'
        self.stage = s
        self.name = self.type + '_' + str(self.stage)
        self.duration = T_RYDBERG
        self.verify(gates, qubit_objs)
        self.code = {'type': self.type,
                     'name': self.name,
                     'gates': gates,
                     'duration': self.duration,
                     'state': self.state(col_objs, row_objs, qubit_objs)}

    def state(self, col_objs, row_objs, qubit_objs):
        curr = {}
        curr['qubits'] = [{'id': q.id, 'x': q.x, 'y': q.y,
                           'array': q.array, 'c': q.c, 'r': q.r} for q in qubit_objs]
        curr['cols'] = [{'id': c.id, 'active': c.active, 'x': c.x}
                        for c in col_objs]
        curr['rows'] = [{'id': r.id, 'active': r.active, 'y': r.y}
                        for r in row_objs]
        return curr

    def verify(self, gates: list, qubit_objs: list):
        # for g in gates:
        #     q0 = {'id': qubit_objs[g['q0']].id,
        #           'x': qubit_objs[g['q0']].x,
        #           'y': qubit_objs[g['q0']].y}
        #     q1 = {'id': qubit_objs[g['q1']].id,
        #           'x': qubit_objs[g['q1']].x,
        #           'y': qubit_objs[g['q1']].y}
        #     if (q0['x']-q1['x'])**2 + (q0['y']-q1['y'])**2 > R_B**2:
        #         raise ValueError(
        #             f"{self.name}: q{q0['id']} at x={q0['x']} y={q0['y']} and q{q1['id']} at x={q1['x']} y={q1['y']} are farther away than Rydberg range.")

        return

    def emit(self):
        return {'type': self.type}

    def emit_full(self):
        return self.code


# class for big operations: ReloadRow, Reload, OffloadRow, Offload
# internally, these are lists of basic operations

# todo: add verify for big operations

class ReloadRow():
    def __init__(self,
                 s: int,
                 r: int,
                 prefix: str = ''):
        self.type = 'ReloadRow'
        self.stage = s
        self.row_id = r
        self.name = prefix + ':' + self.type + '_' + str(self.row_id)
        self.moving_cols_id = []
        self.moving_cols_begin = []
        self.moving_cols_end = []

    def add_col_shift(self, id: int, begin: int, end: int):
        self.moving_cols_id.append(id)
        self.moving_cols_begin.append(begin)
        self.moving_cols_end.append(end)

    def generate_col_shift(self, col_objs: list, row_objs: list, qubit_objs: list):
        self.col_shift = Move(s=self.stage,
                              col_objs=col_objs,
                              row_objs=row_objs,
                              qubit_objs=qubit_objs,
                              col_idx=self.moving_cols_id,
                              col_begin=self.moving_cols_begin,
                              col_end=self.moving_cols_end,
                              row_idx=[],
                              row_begin=[],
                              row_end=[],
                              prefix=self.name + ':ColShift')

    def generate_row_activate(self, col_objs: list, row_objs: list, qubit_objs: list, cols: list, xs: list, y: int, pickup_qs: list):
        self.row_activate = Activate(s=self.stage,
                                     col_objs=col_objs,
                                     row_objs=row_objs,
                                     qubit_objs=qubit_objs,
                                     col_idx=cols,
                                     col_xs=xs,
                                     row_idx=[self.row_id, ],
                                     row_ys=[y, ],
                                     pickup_qs=pickup_qs,
                                     prefix=self.name)

    def generate_parking(self, col_objs, row_objs, qubit_objs: list, shift_down: int, col_idx: list = [], col_begin: list = [], col_end: list = []):
        self.row_shift = Move(s=self.stage,
                              col_objs=col_objs,
                              row_objs=row_objs,
                              qubit_objs=qubit_objs,
                              col_idx=col_idx,
                              col_begin=col_begin,
                              col_end=col_end,
                              row_idx=[self.row_id, ],
                              row_begin=[row_objs[self.row_id].y],
                              row_end=[row_objs[self.row_id].y - shift_down],
                              prefix=self.name + ':Parking')

    def emit(self):
        return [self.col_shift.emit(), self.row_activate.emit(), self.row_shift.emit()]

    def emit_full(self):
        return [self.col_shift.emit_full(), self.row_activate.emit_full(), self.row_shift.emit_full()]


class Reload():
    def __init__(self, s: int):
        self.type = 'Reload'
        self.stage = s
        self.name = self.type + f'_{s}'
        self.row_reloads = []

    def add_row_reload(self, r: int):
        self.row_reloads.append(ReloadRow(self.stage, r, prefix=self.name))
        return self.row_reloads[-1]

    def emit_full(self):
        code_full = []
        for reloadRow in self.row_reloads:
            code_full += reloadRow.emit_full()
        return code_full

    def emit(self):
        code = []
        for reloadRow in self.row_reloads:
            code += reloadRow.emit()
        return code


class OffloadRow():
    def __init__(self,
                 s: int,
                 r: int,
                 prefix: str = ''):
        self.type = 'OffloadRow'
        self.stage = s
        self.row_id = r
        self.name = prefix + ':' + self.type + '_' + str(self.row_id)
        self.moving_cols_id = []
        self.moving_cols_begin = []
        self.moving_cols_end = []

    def add_col_shift(self, id: int, begin: int, end: int):
        self.moving_cols_id.append(id)
        self.moving_cols_begin.append(begin)
        self.moving_cols_end.append(end)

    def generate_col_shift(self, col_objs: list, row_objs: list, qubit_objs: list):
        self.col_shift = Move(s=self.stage,
                              col_objs=col_objs,
                              row_objs=row_objs,
                              qubit_objs=qubit_objs,
                              col_idx=self.moving_cols_id,
                              col_begin=self.moving_cols_begin,
                              col_end=self.moving_cols_end,
                              row_idx=[],
                              row_begin=[],
                              row_end=[],
                              prefix=self.name + ':ColShift')

    def generate_row_shift(self, col_objs, row_objs, qubit_objs: list, site_y: int):
        self.row_shift = Move(s=self.stage,
                              col_objs=col_objs,
                              row_objs=row_objs,
                              qubit_objs=qubit_objs,
                              col_idx=[],
                              col_begin=[],
                              col_end=[],
                              row_idx=[self.row_id, ],
                              row_begin=[row_objs[self.row_id].y],
                              row_end=[site_y*Y_SITE_SEP, ],
                              prefix=self.name + ':RowDownShift')

    def generate_row_deactivate(self, col_objs: list, row_objs: list, qubit_objs: list, dropoff_qs: list):
        self.row_deactivate = Deactivate(s=self.stage,
                                         col_objs=col_objs,
                                         row_objs=row_objs,
                                         qubit_objs=qubit_objs,
                                         col_idx=[],
                                         col_xs=[],
                                         row_idx=[self.row_id, ],
                                         row_ys=[row_objs[self.row_id].y, ],
                                         dropoff_qs=dropoff_qs,
                                         prefix=self.name)

    def emit(self):
        return [self.col_shift.emit(), self.row_shift.emit(), self.row_deactivate.emit()]

    def emit_full(self):
        return [self.col_shift.emit_full(), self.row_shift.emit_full(), self.row_deactivate.emit_full()]


class Offload():
    def __init__(self, s: int):
        self.type = 'Offload'
        self.stage = s
        self.name = self.type + f'_{s}'
        self.row_offloads = []

    def add_row_offload(self, r: int):
        self.row_offloads.append(OffloadRow(self.stage, r, prefix=self.name))
        return self.row_offloads[-1]

    def emit_full(self):
        code_full = []
        for offloadRow in self.row_offloads:
            code_full += offloadRow.emit_full()
        code_full.append(self.col_deactivate.emit_full())
        return code_full

    def emit(self):
        code = []
        for offloadRow in self.row_offloads:
            code += offloadRow.emit()
        code.append(self.col_deactivate.emit())
        return code

    def all_cols_deactivate(self, col_objs, row_objs, qubit_objs):
        self.col_deactivate = Deactivate(s=self.stage,
                                         col_objs=col_objs,
                                         row_objs=row_objs,
                                         qubit_objs=qubit_objs,
                                         col_idx=[
                                             c.id for c in col_objs if c.active],
                                         col_xs=[
                                             c.x for c in col_objs if c.active],
                                         row_idx=[],
                                         row_ys=[],
                                         dropoff_qs=[],
                                         prefix=self.name)


class SwapPair():
    def __init__(self,
                 s: int,
                 col_objs: list,
                 row_objs: list,
                 qubit_objs: list,
                 left_q_id: int,
                 right_q_id: int,
                 prefix: str = ''):
        self.type = 'SwapPair'
        self.stage = s
        self.name = prefix + ':' + self.type + f'({left_q_id},{right_q_id})'
        left_x = qubit_objs[left_q_id].x
        right_x = qubit_objs[right_q_id].x
        y = qubit_objs[left_q_id].y
        qubit_objs[left_q_id].c = 0
        qubit_objs[left_q_id].r = 0
        qubit_objs[right_q_id].c = 0
        qubit_objs[right_q_id].r = 1
        self.objs = [
            Activate(s,
                     col_objs=col_objs,
                     row_objs=row_objs,
                     qubit_objs=qubit_objs,
                     col_idx=[0, ],
                     col_xs=[left_x, ],
                     row_idx=[0, ],
                     row_ys=[y, ],
                     pickup_qs=[left_q_id, ],
                     prefix=self.name + f':PickUp_q{left_q_id}'),
            Move(s,
                 col_objs=col_objs,
                 row_objs=row_objs,
                 qubit_objs=qubit_objs,
                 col_idx=[0, ],
                 col_begin=[left_x, ],
                 col_end=[right_x, ],
                 row_idx=[0, ],
                 row_begin=[y, ],
                 row_end=[y-AOD_SEP, ],
                 prefix=self.name + f':tmp<-q@{left_x}'),
            Activate(s,
                     col_objs=col_objs,
                     row_objs=row_objs,
                     qubit_objs=qubit_objs,
                     col_idx=[],
                     col_xs=[],
                     row_idx=[1, ],
                     row_ys=[y, ],
                     pickup_qs=[right_q_id, ],
                     prefix=self.name + f':PickUp_{right_q_id}'),
            Move(s,
                 col_objs=col_objs,
                 row_objs=row_objs,
                 qubit_objs=qubit_objs,
                 col_idx=[0, ],
                 col_begin=[right_x, ],
                 col_end=[left_x, ],
                 row_idx=[],
                 row_begin=[],
                 row_end=[],
                 prefix=self.name + f':q@{left_x}<-q@{right_x}'),
            Deactivate(s,
                       col_objs=col_objs,
                       row_objs=row_objs,
                       qubit_objs=qubit_objs,
                       col_idx=[],
                       col_xs=[],
                       row_idx=[1, ],
                       row_ys=[y, ],
                       dropoff_qs=[right_q_id, ],
                       prefix=self.name + f':DropOff_q{right_q_id}'),
            Move(s,
                 col_objs=col_objs,
                 row_objs=row_objs,
                 qubit_objs=qubit_objs,
                 col_idx=[0, ],
                 col_begin=[left_x, ],
                 col_end=[right_x, ],
                 row_idx=[0, ],
                 row_begin=[y-AOD_SEP, ],
                 row_end=[y, ],
                 prefix=self.name + f':q@{right_x}<-tmp'),
            Deactivate(s,
                       col_objs=col_objs,
                       row_objs=row_objs,
                       qubit_objs=qubit_objs,
                       col_idx=[0, ],
                       col_xs=[right_x, ],
                       row_idx=[0, ],
                       row_ys=[y, ],
                       dropoff_qs=[left_q_id, ],
                       prefix=self.name + f':DropOff_q{left_q_id}'),
        ]

    def emit(self):
        return [obj.emit() for obj in self.objs]

    def emit_full(self):
        return [obj.emit_full() for obj in self.objs]


class Swap():
    def __init__(self, s: int):
        self.type = 'Swap'
        self.stage = s
        self.name = self.type + f'_{s}'
        self.code = []
        self.code_full = []

    def add_swap_pair(self, col_objs, row_objs, qubit_objs, q_id, qq_id):
        left_q_id = q_id
        right_q_id = qq_id
        if qubit_objs[q_id].x > qubit_objs[qq_id].x:
            left_q_id = qq_id
            right_q_id = q_id
        obj = SwapPair(self.stage, col_objs, row_objs,
                       qubit_objs, left_q_id, right_q_id, self.name)
        self.code += obj.emit()
        self.code_full += obj.emit_full()

    def emit(self):
        return self.code

    def emit_full(self):
        return self.code_full


class CodeGen():
    def __init__(self, file_name: str, no_transfer: bool = False, dir: str = None):
        self.no_transfer = no_transfer
        self.compiled_file = file_name
        if not dir:
            dir = './results/code/'
        self.code_file = dir + \
            (file_name.split('/')[-1]).replace('.json', '_code.json')
        self.code_full_file = dir + \
            (file_name.split('/')[-1]).replace('.json', '_code_full.json')
        self.read_compiled()
        self.code_full = []
        self.code = []
        self.qubits = [Qubit(i) for i in range(self.n_q)]
        self.rows = [Row(i) for i in range(self.r_high)]
        self.cols = [Col(i) for i in range(self.c_high)]

        self.builder()
        self.remove_empty_moves()
        self.code.insert(0, self.init.emit())
        self.code_full.insert(0, self.init.emit_full())
        with open(self.code_file, 'w') as f:
            json.dump(self.code, f)
        with open(self.code_full_file, 'w') as f:
            json.dump(self.code_full, f)

    def read_compiled(self):
        with open(self.compiled_file, 'r') as f:
            data = json.load(f)
        self.n_q = data['n_q']
        self.x_high = data['coord_r']
        self.y_high = data['coord_u']
        self.c_high = data['aod_r']
        self.r_high = data['aod_u']
        self.layers = data['layers']
        self.n_t = len(self.layers)
        for i in range(self.n_t-1, 0, -1):
            for q in range(self.n_q):
                self.layers[i]['qubits'][q]['a'] = self.layers[i -
                                                               1]['qubits'][q]['a']
                self.layers[i]['qubits'][q]['c'] = self.layers[i -
                                                               1]['qubits'][q]['c']
                self.layers[i]['qubits'][q]['r'] = self.layers[i -
                                                               1]['qubits'][q]['r']

    def remove_empty_moves(self):
        vacuous_insts = [i for i in range(
            len(self.code)) if self.code_full[i]['duration'] == 0]
        self.code = [inst for i, inst in enumerate(
            self.code) if i not in vacuous_insts]
        self.code_full = [inst for i, inst in enumerate(
            self.code_full) if i not in vacuous_insts]

    def builder(self):
        self.aod_from_compiled()
        self.builder_init()
        for s in range(1, len(self.layers)):
            self.builder_swap(s)
            if (not self.no_transfer) or s == 1:
                self.builder_reload(s)
            self.builder_move(s)
            if not self.no_transfer:
                self.builder_offload(s)
            self.builder_rydberg(s)

    def aod_from_compiled(self):
        for s, layer in enumerate(self.layers):
            if s == 0:
                continue
            layer['row'] = [{'id': i, 'qs': []}
                            for i in range(self.r_high-0)]
            layer['col'] = [{'id': i, 'qs': []}
                            for i in range(self.c_high-0)]
            prev_layer = self.layers[s-1]
            for i, q in enumerate(layer['qubits']):
                if layer['qubits'][i]['a']:
                    layer['row'][q['r']]['y_begin'] = prev_layer['qubits'][i]['y']
                    layer['row'][q['r']]['y_end'] = q['y']
                    layer['row'][q['r']]['qs'].append(q['id'])
                    layer['col'][q['c']]['x_begin'] = prev_layer['qubits'][i]['x']
                    layer['col'][q['c']]['x_end'] = q['x']
                    layer['col'][q['c']]['qs'].append(q['id'])

            for case in ['_begin', '_end']:
                x_cols = []
                for x in range(self.x_high):
                    cols_at_x = []
                    for c in range(self.c_high):
                        if layer['col'][c]['qs'] and layer['col'][c]['x'+case] == x:
                            cols_at_x.append(c)
                    for i, c in enumerate(cols_at_x):
                        layer['col'][c]['offset'+case] = i
                    x_cols.append(cols_at_x)
                layer['x_cols'+case] = x_cols
                y_rows = []
                for y in range(self.y_high):
                    rows_at_y = []
                    for r in range(self.r_high):
                        if layer['row'][r]['qs'] and layer['row'][r]['y'+case] == y:
                            rows_at_y.append(r)
                    for i, r in enumerate(rows_at_y):
                        layer['row'][r]['offset'+case] = i
                    y_rows.append(rows_at_y)
                layer['y_rows'+case] = y_rows

    def builder_init(self):
        slm_qubit_idx = list(range(self.n_q))  # all qubits in SLM
        slm_qubit_xys = [(X_SITE_SEP*self.layers[0]['qubits'][i]['x'],
                          Y_SITE_SEP*self.layers[0]['qubits'][i]['y']) for i in range(self.n_q)]
        for g in self.layers[0]['gates']:
            a0 = self.layers[1]['qubits'][g['q0']]['a']
            a1 = self.layers[1]['qubits'][g['q1']]['a']
            x_left = X_SITE_SEP*self.layers[0]['qubits'][g['q0']]['x']
            x_right = x_left + SITE_WIDTH
            y = Y_SITE_SEP*self.layers[0]['qubits'][g['q0']]['y']

            if a0 == 1 and a1 == 1 and self.layers[1]['qubits'][g['q0']]['c'] > self.layers[1]['qubits'][g['q1']]['c']:
                slm_qubit_xys[g['q0']] = (x_right, y)
            else:
                slm_qubit_xys[g['q1']] = (x_right, y)

        self.init = Init(0, self.cols, self.rows, self.qubits,
                         slm_qubit_idx=slm_qubit_idx,
                         slm_qubit_xys=slm_qubit_xys)
        self.init.code['n_q'] = self.n_q
        self.init.code['x_high'] = self.x_high
        self.init.code['y_high'] = self.y_high
        self.init.code['c_high'] = self.c_high
        self.init.code['r_high'] = self.r_high
        self.builder_rydberg(0)

    def builder_rydberg(self, s: int):
        obj = Rydberg(
            s, self.cols, self.rows, self.qubits, self.layers[s]['gates'])
        self.code.append(obj.emit())
        self.code_full.append(obj.emit_full())
        self.init.add_slms([(q.x, q.y) for q in self.qubits])

    def builder_swap(self, s: int):
        swap_obj = Swap(s)
        prev_layer = self.layers[s-1]
        this_layer = self.layers[s]
        for q0_id in range(self.n_q):
            for q1_id in range(q0_id+1, self.n_q):
                q0_a = this_layer['qubits'][q0_id]['a']
                q1_a = this_layer['qubits'][q1_id]['a']
                # if two qubits are at the same site and both being picked up
                if q0_a == 1 and q1_a == 1:
                    q0_x = prev_layer['qubits'][q0_id]['x']
                    q1_x = prev_layer['qubits'][q1_id]['x']
                    q0_y = prev_layer['qubits'][q0_id]['y']
                    q1_y = prev_layer['qubits'][q1_id]['y']
                    q0_c = this_layer['qubits'][q0_id]['c']
                    q1_c = this_layer['qubits'][q1_id]['c']
                    q0_r = this_layer['qubits'][q0_id]['r']
                    q1_r = this_layer['qubits'][q1_id]['r']
                    if (q0_x, q0_y, q0_r) == (q1_x, q1_y, q1_r):
                        # if their position and column indeces are in reverse order
                        if (q0_c > q1_c and self.qubits[q0_id].x < self.qubits[q1_id].x) or (q0_c < q1_c and self.qubits[q0_id].x > self.qubits[q1_id].x):
                            swap_obj.add_swap_pair(
                                self.cols, self.rows, self.qubits, q0_id, q1_id)
        self.code += swap_obj.emit()
        self.code_full += swap_obj.emit_full()

    def builder_reload(self, s: int):
        layer = self.layers[s]
        prev_layer = self.layers[s-1]
        reload_obj = Reload(s)
        # reload row by row
        for row_id in range(self.r_high):
            if layer['row'][row_id]['qs']:  # there are qubits to load
                reloadRow_obj = reload_obj.add_row_reload(row_id)
                pickup_qs = []
                cols_to_active = []
                x_to_activate = []
                # consider the movements site by site
                for site_x in range(self.x_high):
                    site_qs = []
                    for q_id in range(len(self.qubits)):
                        if layer['qubits'][q_id]['a'] == 1 and prev_layer['qubits'][q_id]['x'] == site_x and layer['qubits'][q_id]['r'] == row_id:
                            site_qs.append(q_id)
                            self.qubits[q_id].r = row_id
                            self.qubits[q_id].c = layer['qubits'][q_id]['c']
                    if len(site_qs) == 1:
                        q_id = site_qs[0]
                        col_id_left = layer['qubits'][q_id]['c']
                        col_id_right = col_id_left
                        lower_offset = layer['col'][col_id_left]['offset_begin']
                        upper_offset = lower_offset
                        lower_x = self.qubits[q_id].x
                        upper_x = lower_x
                        if not self.cols[col_id_left].active:
                            cols_to_active.append(col_id_left)
                            x_to_activate.append(lower_x)
                        else:  # col already active, shift it to align with q_id
                            reloadRow_obj.add_col_shift(
                                id=col_id_left, begin=self.cols[col_id_left].x, end=lower_x)
                    elif len(site_qs) == 2:
                        [q_id_left, q_id_right] = site_qs
                        if layer['qubits'][q_id_left]['c'] > layer['qubits'][q_id_right]['c']:
                            tmp = q_id_left
                            q_id_left = q_id_right
                            q_id_right = tmp
                        col_id_left = layer['qubits'][q_id_left]['c']
                        col_id_right = layer['qubits'][q_id_right]['c']
                        lower_offset = layer['col'][col_id_left]['offset_begin']
                        upper_offset = layer['col'][col_id_right]['offset_begin']
                        lower_x = self.qubits[q_id_left].x
                        upper_x = self.qubits[q_id_right].x
                        if not self.cols[col_id_left].active:
                            cols_to_active.append(col_id_left)
                            x_to_activate.append(lower_x)
                        else:
                            reloadRow_obj.add_col_shift(
                                id=col_id_left, begin=self.cols[col_id_left].x, end=lower_x)
                        if not self.cols[col_id_right].active:
                            cols_to_active.append(col_id_right)
                            x_to_activate.append(upper_x)
                        else:
                            reloadRow_obj.add_col_shift(
                                id=col_id_right, begin=self.cols[col_id_right].x, end=upper_x)
                    elif len(site_qs) > 2:
                        raise ValueError(
                            f"builder reload {s} row {row_id} site {site_x}: more than 2 qubits")
                    else:
                        continue

                    # which site each col is in corresponds to 'x_cols_begin'
                    # '_begin' is wrt the movement between stage s-1 and s
                    for col_id in layer['x_cols_begin'][site_x]:
                        if self.cols[col_id].active and col_id != col_id_left and col_id != col_id_right:
                            # if there is a col on the right of the cols for loading
                            if layer['col'][col_id]['offset_begin'] > upper_offset:
                                reloadRow_obj.add_col_shift(
                                    id=col_id, begin=self.cols[col_id].x, end=upper_x+AOD_SEP*(layer['col'][col_id]['offset_begin']-upper_offset)+1)
                            # if there is a col on the left of the cols for loading
                            elif layer['col'][col_id]['offset_begin'] < lower_offset:
                                reloadRow_obj.add_col_shift(
                                    id=col_id, begin=self.cols[col_id].x, end=lower_x+AOD_SEP*(layer['col'][col_id]['offset_begin']-lower_offset)-1)
                            # if there is a col in the middle of the cols for loading
                            else:
                                reloadRow_obj.add_col_shift(
                                    id=col_id, begin=self.cols[col_id].x, end=lower_x+AOD_SEP*(layer['col'][col_id]['offset_begin']-lower_offset))

                    pickup_qs += site_qs

                # collect all col shifts for this row and apply
                reloadRow_obj.generate_col_shift(
                    self.cols, self.rows, self.qubits)
                reloadRow_obj.generate_row_activate(self.cols, self.rows, self.qubits,
                                                    cols_to_active, x_to_activate, layer['row'][row_id]['y_begin']*Y_SITE_SEP, pickup_qs)
                # the number of rows in the same site_y
                num_rows = len(layer['y_rows_begin']
                               [layer['row'][row_id]['y_begin']])
                shift_down = (
                    num_rows - layer['row'][row_id]['offset_begin'])*AOD_SEP
                col_idx = [col_id for col_id,
                           col in enumerate(self.cols) if col.active]
                col_begin = [self.cols[col_id].x for col_id in col_idx]
                col_end = [1+layer['col'][col_id]['x_begin']*X_SITE_SEP+AOD_SEP *
                           layer['col'][col_id]['offset_begin'] for col_id in col_idx]
                reloadRow_obj.generate_parking(
                    self.cols, self.rows, self.qubits, shift_down, col_idx, col_begin, col_end)

        self.code += reload_obj.emit()
        self.code_full += reload_obj.emit_full()

    def builder_move(self, s: int):

        col_idx = []
        col_begin = []
        col_end = []
        for col_id in range(self.c_high):
            if self.cols[col_id].active:
                col_idx.append(col_id)
                col_begin.append(self.cols[col_id].x)
                site_x = self.layers[s]['col'][col_id]['x_end']
                offset = self.layers[s]['col'][col_id]['offset_end']
                col_end.append(1+site_x*X_SITE_SEP + AOD_SEP*offset)

        row_idx = []
        row_begin = []
        row_end = []
        for row_id in range(self.r_high):
            if self.rows[row_id].active:
                row_idx.append(row_id)
                row_begin.append(self.rows[row_id].y)
                site_y = self.layers[s]['row'][row_id]['y_end']
                offset = self.layers[s]['row'][row_id]['offset_end']
                row_end.append(site_y*Y_SITE_SEP + AOD_SEP*(1+offset))

        obj = Move(s=s,
                   col_objs=self.cols,
                   row_objs=self.rows,
                   qubit_objs=self.qubits,
                   col_idx=col_idx,
                   col_begin=col_begin,
                   col_end=col_end,
                   row_idx=row_idx,
                   row_begin=row_begin,
                   row_end=row_end,
                   prefix=f'BigMove_{s}')
        self.code.append(obj.emit())
        self.code_full.append(obj.emit_full())

    def builder_offload(self, s: int):
        offload_obj = Offload(s)
        layer = self.layers[s]
        for row_id in range(self.r_high):
            if self.rows[row_id].active:
                dropoff_qs = []
                offloadRow_obj = offload_obj.add_row_offload(row_id)
                for site_x in range(self.x_high):
                    site_q_slm = []
                    site_q_aod = []
                    for q_id, q in enumerate(layer['qubits']):
                        if (q['x'], q['y']) == (site_x, layer['row'][row_id]['y_end']):
                            if self.qubits[q_id].array == 'AOD' and q['r'] == row_id:
                                dropoff_qs.append(q_id)
                                site_q_aod.append(q_id)
                            if self.qubits[q_id].array == 'SLM':
                                site_q_slm.append(q_id)
                    if len(site_q_aod) == 1:
                        q_id = site_q_aod[0]
                        col_id_left = layer['qubits'][q_id]['c']
                        col_id_right = col_id_left
                        lower_offset = layer['col'][col_id_left]['offset_end']
                        upper_offset = lower_offset
                        lower_x = X_SITE_SEP*site_x
                        if site_q_slm:
                            lower_x = 2*X_SITE_SEP*site_x + \
                                SITE_WIDTH - self.qubits[site_q_slm[0]].x
                        upper_x = lower_x
                        offloadRow_obj.add_col_shift(
                            id=col_id_left, begin=self.qubits[q_id].x, end=lower_x)
                    elif len(site_q_aod) == 2:
                        [q_id_left, q_id_right] = site_q_aod
                        if layer['qubits'][q_id_left]['c'] > layer['qubits'][q_id_right]['c']:
                            tmp = q_id_left
                            q_id_left = q_id_right
                            q_id_right = tmp
                        col_id_left = layer['qubits'][q_id_left]['c']
                        col_id_right = layer['qubits'][q_id_right]['c']
                        lower_offset = layer['col'][col_id_left]['offset_end']
                        upper_offset = layer['col'][col_id_right]['offset_end']
                        lower_x = X_SITE_SEP*site_x
                        upper_x = X_SITE_SEP*site_x + SITE_WIDTH

                        offloadRow_obj.add_col_shift(
                            id=col_id_left, begin=self.qubits[q_id_left].x, end=lower_x)
                        offloadRow_obj.add_col_shift(
                            id=col_id_right, begin=self.qubits[q_id_right].x, end=upper_x)
                    elif len(site_q_aod) > 2:
                        raise ValueError(
                            f"builder offload {s} row {row_id} site {site_x}: more than 2 qubits")
                    else:
                        continue

                    for col_id in layer['x_cols_end'][site_x]:
                        if self.cols[col_id].active and col_id != col_id_left and col_id != col_id_right:
                            if layer['col'][col_id]['offset_end'] > upper_offset:
                                offloadRow_obj.add_col_shift(
                                    id=col_id, begin=self.cols[col_id].x, end=upper_x+AOD_SEP*(layer['col'][col_id]['offset_end']-upper_offset)+1)
                            elif layer['col'][col_id]['offset_end'] < lower_offset:
                                offloadRow_obj.add_col_shift(
                                    id=col_id, begin=self.cols[col_id].x, end=lower_x+AOD_SEP*(layer['col'][col_id]['offset_end']-lower_offset)-1)
                            else:
                                offloadRow_obj.add_col_shift(
                                    id=col_id, begin=self.cols[col_id].x, end=lower_x+AOD_SEP*(layer['col'][col_id]['offset_end']-lower_offset))

                offloadRow_obj.generate_col_shift(
                    self.cols, self.rows, self.qubits)
                offloadRow_obj.generate_row_shift(
                    self.cols, self.rows, self.qubits, layer['row'][row_id]['y_end'])
                offloadRow_obj.generate_row_deactivate(
                    self.cols, self.rows, self.qubits, dropoff_qs)
        offload_obj.all_cols_deactivate(self.cols, self.rows, self.qubits)
        self.code += offload_obj.emit()
        self.code_full += offload_obj.emit_full()


class Animator():
    def __init__(self, code_file_name, scaling_factor=PT_MICRON, font=10, ffmpeg='ffmpeg', real_speed=False, show_graph=False, edges=None, dir: str = None):
        matplotlib.use('Agg')
        matplotlib.rcParams.update({'font.size': font})
        plt.rcParams['animation.ffmpeg_path'] = ffmpeg
        self.scaling = scaling_factor
        self.code_file = code_file_name
        self.animation_file = dir + (code_file_name.replace(
            '.json', '.mp4')).split('/')[-1]
        self.real_speed = real_speed
        self.show_graph = show_graph
        if show_graph:
            self.graph_edges = edges
            self.graph_alpha = [1 for _ in range(len(self.graph_edges))]
            self.graph_color = ['black' for _ in range(len(self.graph_edges))]
            self.previous_edges = None
        self.read_files()

        self.keyframes = []
        self.setup_canvas()
        self.create_master_schedule()

        anim = FuncAnimation(self.fig, self.update, init_func=self.update_init, frames=self.keyframes[-1],
                             interval=1000/FPS)
        anim.save(self.animation_file, writer=FFMpegWriter(FPS))

    def read_files(self):
        with open(self.code_file, 'r') as f:
            self.code = json.load(f)

        self.n_q = self.code[0]['n_q']
        self.x_high = self.code[0]['x_high']
        self.y_high = self.code[0]['y_high']
        self.c_high = self.code[0]['c_high']
        self.r_high = self.code[0]['r_high']

    def create_master_schedule(self):
        frame = 0
        for inst in self.code:
            if not self.real_speed:
                if inst['type'] == 'Move' and 'BigMove' not in inst['name']:
                    inst['duration'] = 1
            inst['f_begin'] = frame
            new_frame = frame + int((inst['duration'] +
                                     MUS_PER_FRM - 1)/MUS_PER_FRM)
            inst['f_end'] = new_frame-1
            self.keyframes.append(new_frame)
            frame = new_frame

    def setup_canvas(self):
        px = 1/plt.rcParams['figure.dpi'] * self.scaling
        self.X_LOW = -X_LOW_PAD
        self.X_HIGH = SITE_WIDTH + \
            (self.x_high-0-1)*X_SITE_SEP + X_HIGH_PAD
        self.Y_LOW = -Y_LOW_PAD
        self.Y_HIGH = (self.y_high-0-1)*Y_SITE_SEP + Y_HIGH_PAD
        self.fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=((self.X_HIGH-self.X_LOW)*px*4/3,
                                                                                         (self.Y_HIGH-self.Y_LOW)*px))
        self.fig.tight_layout()
        self.ax = ax[0]
        self.network_ax = ax[1]
        self.network = nx.Graph()
        self.ax.set_xlim([self.X_LOW, self.X_HIGH])
        self.ax.set_xticks(
            [SITE_WIDTH/2+X_SITE_SEP*i for i in range(self.x_high)])
        self.ax.set_xticklabels([i for i in range(self.x_high)])
        self.ax.set_ylim(
            [self.Y_LOW, self.Y_HIGH])
        self.ax.set_yticks(
            [Y_SITE_SEP*i for i in range(self.y_high)])
        self.ax.set_yticklabels([i for i in range(self.y_high)])
        if self.show_graph:
            self.network_ax.set_title('The 3-regular graph')
        self.title = self.ax.set_title('')

        # slm_xs = [slm[0] for slm in self.code[0]['all_slms']]
        # slm_ys = [slm[1] for slm in self.code[0]['all_slms']]
        # self.ax.scatter(slm_xs, slm_ys, marker='o', s=50,
        #                 facecolor='none', edgecolor=(0, 0, 1, 0.5))

    def update_init(self):
        slm_xs = [xy[0] for xy in self.code[0]["slm_qubit_xys"]]
        slm_ys = [xy[1] for xy in self.code[0]["slm_qubit_xys"]]
        self.qubit_scat = self.ax.scatter(slm_xs, slm_ys, c='b')
        self.col_plots = [self.ax.axvline(0, self.Y_LOW, self.Y_HIGH, c=(
            1, 0, 0, 0), ls='--') for _ in range(self.c_high)]
        self.row_plots = [self.ax.axhline(0, self.X_LOW, self.X_HIGH, c=(
            1, 0, 0, 0), ls='--') for _ in range(self.c_high)]

        if self.show_graph:
            self.network.add_edges_from(self.graph_edges)
            self.pos = nx.spring_layout(self.network)
            nx.draw(self.network, pos=self.pos, ax=self.network_ax,
                    with_labels=True, edgecolors='black', node_color='lightgray')
        return

    def update(self, f: int):  # f is the frame
        if f < self.keyframes[0]:
            return
        for i in range(len(self.code)):
            if f >= self.keyframes[i-1] and f < self.keyframes[i]:
                inst = self.code[i]
                if inst['type'] == 'Rydberg':
                    return self.update_rydberg(f, inst)
                elif inst['type'] == 'Move':
                    return self.update_move(f, inst, self.code[i-1]['state'])
                elif inst['type'] == 'Activate':
                    return self.update_activate(f, inst)
                elif inst['type'] == 'Deactivate':
                    return self.update_deactivate(f, inst)
                elif inst['type'] == 'Init':
                    return
                else:
                    raise ValueError(f"unknown inst type {inst['type']}")

    # s is the stage like in compiled results
    # f is the relative frame in this frame group
    def update_rydberg(self, f: int, inst: dict):
        edges = [(g['q0'], g['q1']) for g in inst['gates']]
        if f == inst['f_begin']:
            self.title.set_text(inst['name'])
            self.texts = []
            active_qubits = []
            for g in inst['gates']:
                active_qubits += [g['q0'], g['q1']]
            for q_id in active_qubits:
                self.texts.append(
                    self.ax.text(inst['state']['qubits'][q_id]['x']+1, inst['state']['qubits'][q_id]['y']+1, q_id))
            # adjust_text(self.texts)
            self.ax.set_facecolor((0, 0, 1, 0.2))
            if f == 0:
                self.qubit_scat.set_offsets(
                    [(q['x'], q['y']) for q in inst['state']['qubits']])
                self.qubit_scat.set_color((0, 0, 1, 1))
            # print([q['x'] for q in inst['state']['qubits']])
            # print([q['y'] for q in inst['state']['qubits']])

        if self.show_graph and f == int((inst['f_begin'] + inst['f_end'])/2):
            nx.draw_networkx_edges(
                self.network, pos=self.pos, edgelist=edges, edge_color='blue', ax=self.network_ax, width=4)
            # print(edges)
            # if self.previous_edges:
            #     nx.draw_networkx_edges(
            #     self.network, pos=self.pos, edgelist=self.previous_edges, edge_color='white', ax=self.network_ax, width=4)
            #     nx.draw_networkx_edges(
            #     self.network, pos=self.pos, edgelist=self.previous_edges, edge_color='black', ax=self.network_ax, width=1)

        if f == inst['f_end']:
            self.ax.set_facecolor('w')
            for text in self.texts:
                text.remove()
            self.previous_edges = edges

    def interpolate(self, progress: int, duration: int, begin: int, end: int):
        D = end - begin
        if D == 0:
            return begin
        r = (1+progress)/duration
        return begin + 3*D*(r**2) - 2*D*(r**3)

    def update_move(self, f: int, inst: dict, prev_state: dict):
        if f == inst['f_begin']:
            self.title.set_text(inst['name'])

        progress = f - inst['f_begin']
        duration = inst['f_end'] - inst['f_begin'] + 1
        col_xs = [col['x'] for col in prev_state['cols']]
        for col in inst['cols']:
            curr_x = self.interpolate(
                progress, duration, col['begin'], col['end'])
            col_xs[col['id']] = curr_x
            self.col_plots[col['id']].set_xdata(col_xs[col['id']])

        row_ys = [row['y'] for row in prev_state['rows']]
        for row in inst['rows']:
            curr_y = self.interpolate(
                progress, duration, row['begin'], row['end'])
            row_ys[row['id']] = curr_y
            self.row_plots[row['id']].set_ydata(row_ys[row['id']])

        q_xs = [q['x'] for q in prev_state['qubits']]
        q_ys = [q['y'] for q in prev_state['qubits']]
        for q_id in range(self.n_q):
            if prev_state['qubits'][q_id]['array'] == 'AOD':
                q_xs[q_id] = col_xs[prev_state['qubits'][q_id]['c']]
                q_ys[q_id] = row_ys[prev_state['qubits'][q_id]['r']]
        self.qubit_scat.set_offsets([(q_xs[i], q_ys[i])
                                    for i in range(self.n_q)])
        # if 'BigMove' in inst['name'] and (f == inst['f_begin'] or f == inst['f_end']):
        #     print(q_xs)
        #     print(q_ys)
        #     print([q['array'] for q in prev_state['qubits']])

        return self.col_plots

    def update_activate(self, f: int, inst: dict):
        if f == inst['f_begin']:
            self.title.set_text(inst['name'])
            for id, col in enumerate(inst['col_idx']):
                self.col_plots[col].set_xdata(inst['col_xs'][id])
                self.col_plots[col].set_color((1, 0, 0, 0.2))
            for id, row in enumerate(inst['row_idx']):
                self.row_plots[row].set_ydata(inst['row_ys'][id])
                self.row_plots[row].set_color((1, 0, 0, 0.2))

        # if f == inst['f_begin'] + int(inst['duration']/2):
            self.qubit_scat.set_color(
                ['b' if q['array'] == 'SLM' else 'r' for q in inst['state']['qubits']])

    def update_deactivate(self, f: int, inst: dict):
        if f == inst['f_end']:
            self.title.set_text(inst['name'])
            for col in inst['col_idx']:
                self.col_plots[col].set_color((1, 0, 0, 0))
            for row in inst['row_idx']:
                self.row_plots[row].set_color((1, 0, 0, 0))

        # if f == inst['f_begin'] + int(inst['duration']/2):
            self.qubit_scat.set_color(
                ['b' if q['array'] == 'SLM' else 'r' for q in inst['state']['qubits']])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument(
        '--scaling', help='scaling factor of the animation', type=int)
    parser.add_argument('--font', help='font size in the animation', type=int)
    parser.add_argument('--ffmpeg', help='custom ffmpeg path', type=str)
    parser.add_argument(
        '--realSpeed', help='use real speed in the animation of reload and offload procedures', action='store_true')
    parser.add_argument(
        '--noGraph', help='do not show graph on the side', action='store_true')
    parser.add_argument('--dir', help='working directory', type=str)
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        data = json.load(f)
    codegen = CodeGen(
        args.input_file,
        no_transfer=data['no_transfer'],
        dir=args.dir if args.dir else './results/code/'
    )
    Animator(
        codegen.code_full_file,
        scaling_factor=args.scaling if args.scaling else PT_MICRON,
        font=args.font if args.font else 10,
        ffmpeg=args.ffmpeg if args.ffmpeg else 'ffmpeg',
        real_speed=args.realSpeed,
        show_graph=not args.noGraph,
        edges=data["g_q"] if not args.noGraph else [],
        dir=args.dir if args.dir else './results/animations/'
    )
