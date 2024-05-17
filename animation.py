from matplotlib.animation import FFMpegWriter, FuncAnimation
from typing import Sequence, Mapping, Any, Union
import matplotlib.pyplot as plt
import json
import matplotlib
import networkx as nx
import argparse
from abc import ABC, abstractmethod


# physics constants
R_B = 2 #6  # rydberg range
AOD_SEP = 1 #2  # min AOD separation
RYD_SEP = 10 #15  # sufficient distance to avoid Rydberg
SITE_WIDTH = 2 * AOD_SEP #4  # total width of SLMs in a site
X_SITE_SEP = RYD_SEP + SITE_WIDTH  # separation of sites in X direction
Y_SITE_SEP = RYD_SEP  # separation of sites in Y direction

# padding of the figure
X_LOW_PAD = 2 * AOD_SEP
Y_LOW_PAD = 4 * AOD_SEP
X_HIGH_PAD = 2 * AOD_SEP
Y_HIGH_PAD = 4 * AOD_SEP

# constants for animation
FPS = 24  # frames per second
INIT_FRM = 24  # initial empty frames
PT_MICRON = 8  # scaling factor: points per micron
MUS_PER_FRM = 8  # microseconds per frame
T_RYDBERG = 0.15  # microseconds for Rydberg
T_ACTIVATE = 50  # microseconds for (de)activating AOD


# class for physical entities: qubits and AOD rows/cols

class Qubit():
    def __init__(self, id: int):
        self.array = 'SLM'
        self.id = id
        self.c = -1  # AOD coloumn index
        self.r = -1  # AOD row index
        self.x = -X_LOW_PAD-1  # real X coordinates in um
        self.y = -Y_LOW_PAD-1  # real Y coordinates in um


class Row():
    def __init__(self, id: int):
        self.id = id
        self.active = False
        self.y = -Y_LOW_PAD-1  # real Y coordinates in um


class Col():
    def __init__(self, id: int):
        self.id = id
        self.active = False
        self.x = -X_LOW_PAD-1  # real X coordinates in um


class Inst(ABC):
    """abstract class of DPQA instructions.

    In general, the __init__ of specific instruction classes looks like
        def __init__(self, *):
            super().__init__(*)
            self.verify(*)
            self.operate(*)
            super().write_code(*)
    """

    def __init__(
            self,
            type: str,
            prefix: Union[str, None] = None,
            stage: int = -1,
            reduced_keys: Sequence[str] = [],
    ):
        """init method for instructions.

        Args:
            type (str): 
            prefix (str | None, optional): provide the big operation.
                this Inst belongs to. Defaults to None.
            stage (int, optional): stage the Inst belongs to. Defaults to -1.
            reduced_keys (Sequence[str], optional): data to keep in emit()
                from emit_full(). Defaults to [].
        """
        self.type = type
        self.name = prefix + ':' + type if prefix else type
        self.stage = stage
        self.reduced_keys = reduced_keys + ['type', ]
        self.duration = -1
        self.code = {'type': self.type, 'name': self.name, }

    def write_code(
            self,
            col_objs: Sequence[Col],
            row_objs: Sequence[Row],
            qubit_objs: Sequence[Qubit],
            data: Mapping[str, Any],
    ):
        """write self.code with provided info.

        Args:
            col_objs (Sequence[Col]): Col objects used.
            row_objs (Sequence[Row]): Row objects used.
            qubit_objs (Sequence[Qubit]): Qubit objects used.
            data (Mapping[str, Any]): other info for the Inst.
        """
        for k, v in data.items():
            self.code[k] = v

        # get the current state of DPQA
        curr = {}
        curr['qubits'] = [
            {
                'id': q.id,
                'x': q.x,
                'y': q.y,
                'array': q.array,
                'c': q.c,
                'r': q.r
            } for q in qubit_objs
        ]
        curr['cols'] = [
            {
                'id': c.id,
                'active': c.active,
                'x': c.x
            } for c in col_objs
        ]
        curr['rows'] = [
            {
                'id': r.id,
                'active': r.active,
                'y': r.y
            } for r in row_objs
        ]
        self.code['state'] = curr

    @abstractmethod
    def verify(self):
        """verification of instructions. This is abstract because we require
        each child class to provide its own verification method.
        """
        pass

    def operate(self):
        """perform operation of instructions on Col, Row, and Qubit objects."""
        pass

    def emit(self) -> Sequence[Mapping[str, Any]]:
        """emit the minimum code for executing instructions.

        Returns:
            Sequence[Mapping[str, Any]]: code in a dict.
        """
        return ({k: self.code[k] for k in self.reduced_keys}, )

    def emit_full(self) -> Sequence[Mapping[str, Any]]:
        """emit the code with full info for any purpose.

        Returns:
            Sequence[Mapping[str, Any]]: code with full info in a dict.
        """
        return (self.code, )

    def is_trivial(self) -> bool:
        return True if self.duration == 0 else False

    def remove_trivial_insts(self):
        # this is used in ComboInst. Added here for convience.
        pass


# classes for basic instructions: Init, Move, Activate, Deactivate, Rydberg
# todo: add class Raman (single-qubit gates)


class Init(Inst):
    def __init__(self,
                 col_objs: Sequence[Col],
                 row_objs: Sequence[Row],
                 qubit_objs: Sequence[Qubit],
                 slm_qubit_idx: Sequence[int] = [],
                 slm_qubit_xys: Sequence[Sequence[int]] = [],
                 aod_qubit_idx: Sequence[int] = [],
                 aod_qubit_crs: Sequence[Sequence[int]] = [],
                 aod_col_act_idx: Sequence[int] = [],
                 aod_col_xs: Sequence[int] = [],
                 aod_row_act_idx: Sequence[int] = [],
                 aod_row_ys: Sequence[int] = [],
                 data: Mapping[str, Any] = {},):
        super().__init__(
            'Init',
            reduced_keys=[
                'slm_qubit_idx', 'slm_qubit_xys', 'aod_qubit_idx',
                'aod_qubit_crs', 'aod_col_act_idx', 'aod_col_xs',
                'aod_row_act_idx', 'aod_row_ys', 'n_q',
                'x_high', 'y_high', 'c_high', 'r_high'
            ]
        )
        for k, v in data.items():
            self.code[k] = v
        self.all_slms = []
        self.verify(slm_qubit_idx,
                    slm_qubit_xys,
                    aod_qubit_idx,
                    aod_qubit_crs,
                    aod_col_act_idx,
                    aod_col_xs,
                    aod_row_act_idx,
                    aod_row_ys,)
        self.operate(col_objs,
                     row_objs,
                     qubit_objs,
                     slm_qubit_idx,
                     slm_qubit_xys,
                     aod_qubit_idx,
                     aod_qubit_crs,
                     aod_col_act_idx,
                     aod_col_xs,
                     aod_row_act_idx,
                     aod_row_ys,)
        super().write_code(
            col_objs,
            row_objs,
            qubit_objs,
            {
                'duration': INIT_FRM,
                'slm_qubit_idx': slm_qubit_idx,
                'slm_qubit_xys': slm_qubit_xys,
                'aod_qubit_idx': aod_qubit_idx,
                'aod_qubit_crs': aod_qubit_crs,
                'aod_col_act_idx': aod_col_act_idx,
                'aod_col_xs': aod_col_xs,
                'aod_row_act_idx': aod_row_act_idx,
                'aod_row_ys': aod_row_ys,
            })

    def add_slms(self, slms: Sequence[Sequence[int]]):
        for slm in slms:
            if slm not in self.all_slms:
                self.all_slms.append(slm)

    def verify(
            self,
            slm_qubit_idx: Sequence[int],
            slm_qubit_xys: Sequence[Sequence[int]],
            aod_qubit_idx: Sequence[int],
            aod_qubit_crs: Sequence[Sequence[int]],
            aod_col_act_idx: Sequence[int],
            aod_col_xs: Sequence[int],
            aod_row_act_idx: Sequence[int],
            aod_row_ys: Sequence[int],
    ):
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
                        f'{self.name}: SLM qubits {slm_qubit_idx[i]} '
                        f'and {slm_qubit_idx[j]} xys are the same.'
                    )
        # todo: the case when not all atoms are in SLM

    def operate(
            self,
            col_objs: Sequence[Col],
            row_objs: Sequence[Row],
            qubit_objs: Sequence[Qubit],
            slm_qubit_idx: Sequence[int],
            slm_qubit_xys: Sequence[Sequence[int]],
            aod_qubit_idx: Sequence[int],
            aod_qubit_crs: Sequence[Sequence[int]],
            aod_col_act_idx: Sequence[int],
            aod_col_xs: Sequence[int],
            aod_row_act_idx: Sequence[int],
            aod_row_ys: Sequence[int],
    ):
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

    def emit_full(self):
        # all the used SLMs are counted during the whole codegen process,
        # so the emit_full of Init needs to add this info
        self.code['all_slms'] = self.all_slms
        return super().emit_full()


class Move(Inst):
    def __init__(self,
                 s: int,
                 col_objs: Sequence[Col],
                 row_objs: Sequence[Row],
                 qubit_objs: Sequence[Qubit],
                 col_idx: Sequence[int] = [],
                 col_begin: Sequence[int] = [],
                 col_end: Sequence[int] = [],
                 row_idx: Sequence[int] = [],
                 row_begin: Sequence[int] = [],
                 row_end: Sequence[int] = [],
                 prefix: str = ''):
        super().__init__('Move', prefix=prefix, stage=s)
        self.verify(
            col_objs,
            row_objs,
            col_idx,
            col_begin,
            col_end,
            row_idx,
            row_begin,
            row_end,
        )
        data = self.operate(
            col_objs,
            row_objs,
            qubit_objs,
            col_idx,
            col_begin,
            col_end,
            row_idx,
            row_begin,
            row_end,)
        super().write_code(col_objs, row_objs, qubit_objs, data)

    def operate(
            self,
            col_objs: Sequence[Col],
            row_objs: Sequence[Row],
            qubit_objs: Sequence[Qubit],
            col_idx: Sequence[int],
            col_begin: Sequence[int],
            col_end: Sequence[int],
            row_idx: Sequence[int],
            row_begin: Sequence[int],
            row_end: Sequence[int],
    ) -> Mapping[str, Any]:
        data = {}
        # calculate the max  move distance of columns
        data['cols'] = []
        max_distance = 0
        for i in range(len(col_idx)):
            distance = abs(col_end[i]-col_begin[i])
            if distance > 0:
                data['cols'].append(
                    {'id': col_idx[i],
                     'shift': col_end[i]-col_begin[i],
                     'begin': col_begin[i],
                     'end': col_end[i]})
                col_objs[col_idx[i]].x = col_end[i]
                max_distance = max(max_distance, distance)

        # calculate the max  move distance of rows
        data['rows'] = []
        for i in range(len(row_idx)):
            distance = abs(row_end[i]-row_begin[i])
            if distance > 0:
                data['rows'].append(
                    {'id': row_idx[i],
                     'shift': row_end[i]-row_begin[i],
                     'begin': row_begin[i],
                     'end': row_end[i]})
                row_objs[row_idx[i]].y = row_end[i]
                max_distance = max(max_distance, distance)

        # movement time per Bluvstein et al. units are us and um.
        self.duration = 200*((max_distance/110)**(1/2))
        data['duration'] = self.duration

        for qubit_obj in qubit_objs:
            if qubit_obj.array == 'AOD':
                qubit_obj.x = col_objs[qubit_obj.c].x
                qubit_obj.y = row_objs[qubit_obj.r].y

        return data

    def verify(
            self,
            col_objs: Sequence[Col],
            row_objs: Sequence[Row],
            col_idx: Sequence[int],
            col_begin: Sequence[int],
            col_end: Sequence[int],
            row_idx: Sequence[int],
            row_begin: Sequence[int],
            row_end: Sequence[int],
    ):
        a = len(col_idx)
        b = len(col_begin)
        c = len(col_end)
        if not (a == b and a == c):
            raise ValueError(
                f'{self.name}: col arguments invalid'
                f' {a} idx, {b} begin, {c} end.'
            )
        a = len(row_idx)
        b = len(row_begin)
        c = len(row_end)
        if not (a == b and a == c):
            raise ValueError(
                f'{self.name}: row arguments invalid'
                f' {a} idx, {b} begin, {c} end.'
            )

        activated_col_idx = []
        activated_col_xs = []
        for col_obj in col_objs:
            if col_obj.active:
                if (activated_col_idx
                        and col_obj.x < activated_col_xs[-1] + AOD_SEP):
                    raise ValueError(
                        f'{self.name}: col beginning position invalid'
                        f' col {col_obj.id} at x={col_obj.x} while '
                        f'col {activated_col_idx[-1]} at'
                        f' x={activated_col_xs[-1]}.'
                    )
                activated_col_idx.append(col_obj.id)
                activated_col_xs.append(col_obj.x)
        for i, moving_col_id in enumerate(col_idx):
            if moving_col_id not in activated_col_idx:
                raise ValueError(
                    f'{self.name}: col {moving_col_id} to move'
                    f' is not activated.'
                )
            j = activated_col_idx.index(moving_col_id)
            if col_begin[i] != activated_col_xs[j]:
                raise ValueError(
                    f'{self.name}: col {moving_col_id} beginning x not agree.')
            activated_col_xs[j] = col_end[i]
        for i in range(1, len(activated_col_xs)):
            if activated_col_xs[i - 1] + AOD_SEP > activated_col_xs[i]:
                raise ValueError(
                    f'{self.name}: col ending position invalid'
                    f' col {activated_col_idx[i-1]} at '
                    f'x={activated_col_xs[i-1]} while '
                    f'col {activated_col_idx[i]} at x={activated_col_xs[i]}.')

        activated_row_idx = []
        activated_row_ys = []
        for row_obj in row_objs:
            if row_obj.active:
                if (activated_row_idx
                        and row_obj.y < activated_row_ys[-1] + AOD_SEP):
                    raise ValueError(
                        f'{self.name}: row beginning position invalid '
                        f'row {row_obj.id} at y={row_obj.y} while '
                        f'row {activated_row_idx[-1]} at '
                        f'y={activated_row_ys[-1]}.'
                    )
                activated_row_idx.append(row_obj.id)
                activated_row_ys.append(row_obj.y)
        for i, moving_row_id in enumerate(row_idx):
            if moving_row_id not in activated_row_idx:
                raise ValueError(
                    f'{self.name}: row {moving_row_id} to move '
                    f'is not activated.'
                )
            j = activated_row_idx.index(moving_row_id)
            if row_begin[i] != activated_row_ys[j]:
                raise ValueError(
                    f'{self.name}: row {moving_row_id} beginning y not agree.')
            activated_row_ys[j] = row_end[i]
        for i in range(1, len(activated_row_ys)):
            if activated_row_ys[i - 1] + AOD_SEP > activated_row_ys[i]:
                raise ValueError(
                    f'{self.name}: row ending position invalid '
                    f'row {activated_row_idx[i-1]} at '
                    f'y={activated_row_ys[i-1]} while '
                    f'row {activated_row_idx[i]} at y={activated_row_ys[i]}.')

    def emit(self):
        code = {'type': self.type}
        code['cols'] = [{k: v for k, v in tmp.items() if k in [
            'id', 'shift']} for tmp in self.code['cols']]
        code['rows'] = [{k: v for k, v in tmp.items() if k in [
            'id', 'shift']} for tmp in self.code['rows']]
        return (code,)


class Activate(Inst):
    def __init__(self,
                 s: int,
                 col_objs: Sequence[Col],
                 row_objs: Sequence[Row],
                 qubit_objs: Sequence[Qubit],
                 col_idx: Sequence[int] = [],
                 col_xs: Sequence[int] = [],
                 row_idx: Sequence[int] = [],
                 row_ys: Sequence[int] = [],
                 pickup_qs: Sequence[int] = [],
                 prefix: str = '',):
        super().__init__(
            'Activate',
            prefix=prefix,
            stage=s,
            reduced_keys=['col_idx', 'col_xs', 'row_idx', 'row_ys']
        )
        self.verify(
            col_objs,
            row_objs,
            qubit_objs,
            col_idx,
            col_xs,
            row_idx,
            row_ys,
            pickup_qs,
        )
        self.operate(
            col_objs,
            row_objs,
            qubit_objs,
            col_idx,
            col_xs,
            row_idx,
            row_ys,
            pickup_qs,
        )
        super().write_code(
            col_objs,
            row_objs,
            qubit_objs,
            {
                'col_idx': col_idx,
                'col_xs': col_xs,
                'row_idx': row_idx,
                'row_ys': row_ys,
                'pickup_qs': pickup_qs,
                'duration': T_ACTIVATE
            }
        )

    def operate(
            self,
            col_objs: Sequence[Col],
            row_objs: Sequence[Row],
            qubit_objs: Sequence[Qubit],
            col_idx: Sequence[int],
            col_xs: Sequence[int],
            row_idx: Sequence[int],
            row_ys: Sequence[int],
            pickup_qs: Sequence[int],
    ):
        for i in range(len(col_idx)):
            col_objs[col_idx[i]].active = True
            col_objs[col_idx[i]].x = col_xs[i]
        for i in range(len(row_idx)):
            row_objs[row_idx[i]].active = True
            row_objs[row_idx[i]].y = row_ys[i]
        for q_id in pickup_qs:
            qubit_objs[q_id].array = 'AOD'

    def verify(
            self,
            col_objs: Sequence[Col],
            row_objs: Sequence[Row],
            qubit_objs: Sequence[Qubit],
            col_idx: Sequence[int],
            col_xs: Sequence[int],
            row_idx: Sequence[int],
            row_ys: Sequence[int],
            pickup_qs: Sequence[int],
    ):
        print("in verify 1")
        print("active_xs: ", [col.x for col in col_objs if col.active])
        print("active_xs_id: ", [i for i, col in enumerate(col_objs) if col.active])
        a = len(col_idx)
        b = len(col_xs)
        if a != b:
            raise ValueError(
                f'{self.name}: col arguments invalid {a} idx, {b} xs')
        a = len(row_idx)
        b = len(row_ys)
        if a != b:
            raise ValueError(
                f'f{self.name}: row arguments invalid {a} idx, {b} ys.')

        for i in range(len(col_idx)):
            if col_objs[col_idx[i]].active:
                raise ValueError(
                    f'{self.name}: col {col_idx[i]} already activated.')
            for j in range(col_idx[i]):
                if (col_objs[j].active
                        and col_objs[j].x > col_xs[i] - AOD_SEP):
                    raise ValueError(
                        f'{self.name}: col {j} at x={col_objs[j].x} is '
                        f'too left for col {col_idx[i]}'
                        f' to activate at x={col_xs[i]}.'
                    )
            for j in range(col_idx[i] + 1, len(col_objs)):
                if (col_objs[j].active
                        and col_objs[j].x - AOD_SEP < col_xs[i]):
                    raise ValueError(
                        f'{self.name}: col {j} at x={col_objs[j].x} is '
                        f'too right for col {col_idx[i]} '
                        f'to activate at x={col_xs[i]}.'
                    )
        for i in range(len(row_idx)):
            if row_objs[row_idx[i]].active:
                raise ValueError(
                    f'{self.name}: row {row_idx[i]} already activated.')
            for j in range(row_idx[i]):
                if (row_objs[j].active
                        and row_objs[j].y > row_ys[i] - AOD_SEP):
                    raise ValueError(
                        f'{self.name}: row {j} at y={row_objs[j].y} is '
                        f'too high for row {row_idx[i]} '
                        f'to activate at y={row_ys[i]}.'
                    )
            for j in range(row_idx[i] + 1, len(row_objs)):
                if (row_objs[j].active
                        and row_objs[j].y-AOD_SEP < row_ys[i]):
                    raise ValueError(
                        f'{self.name}: row {j} at y={col_objs[j].y} is '
                        f'too low for row {row_idx[i]} '
                        f'to activate at y={row_ys[i]}.'
                    )

        active_xys = []  # the traps that are newly activted by this Activate
        active_xs = [col.x for col in col_objs if col.active]
        active_ys = [row.y for row in row_objs if row.active]
        print("in verify 2")
        print("active_xs: ", active_xs)
        print("active_xs_id: ", [i for i, col in enumerate(col_objs) if col.active])
        
        for x in active_xs:
            for y in row_ys:
                active_xys.append((x, y))
        for y in active_ys:
            for x in col_xs:
                active_xys.append((x, y))
        for x in col_xs:
            for y in row_ys:
                active_xys.append((x, y))

        for q_id in range(len(qubit_objs)):
            if q_id in pickup_qs:
                if (qubit_objs[q_id].x, qubit_objs[q_id].y) not in active_xys:
                    raise ValueError(
                        f'{self.name}: q {q_id} not picked up '
                        f'by col {qubit_objs[q_id].c} '
                        f'row {qubit_objs[q_id].r} at '
                        f'x={qubit_objs[q_id].x} y={qubit_objs[q_id].y}.'
                    )
            else:
                # !
                if (qubit_objs[q_id].x, qubit_objs[q_id].y) in active_xys:
                    raise ValueError(
                        f'{self.name}: q {q_id} wrongfully picked up by '
                        f'col {qubit_objs[q_id].c} row {qubit_objs[q_id].r}'
                        f' at x={qubit_objs[q_id].x} y={qubit_objs[q_id].y}.')


class Deactivate(Inst):
    def __init__(self,
                 s: int,
                 col_objs: Sequence[Col],
                 row_objs: Sequence[Row],
                 qubit_objs: Sequence[Qubit],
                 col_idx: Sequence[int] = [],
                 col_xs: Sequence[int] = [],
                 row_idx: Sequence[int] = [],
                 row_ys: Sequence[int] = [],
                 dropoff_qs: Sequence[int] = [],
                 prefix: str = '',):
        super().__init__(
            'Deactivate',
            prefix=prefix,
            stage=s,
            reduced_keys=['col_idx', 'row_idx']
        )
        self.verify(
            col_objs,
            row_objs,
            qubit_objs,
            col_idx,
            col_xs,
            row_idx,
            row_ys,
            dropoff_qs
        )
        self.operate(
            col_objs,
            row_objs,
            qubit_objs,
            col_idx,
            row_idx,
            dropoff_qs
        )
        super().write_code(
            col_objs,
            row_objs,
            qubit_objs,
            {
                'col_idx': col_idx,
                'col_xs': col_xs,
                'row_idx': row_idx,
                'row_ys': row_ys,
                'dropoff_qs': dropoff_qs,
                'duration': T_ACTIVATE
            }
        )

    def operate(
        self,
        col_objs: Sequence[Col],
        row_objs: Sequence[Row],
        qubit_objs: Sequence[Qubit],
        col_idx: Sequence[int],
        row_idx: Sequence[int],
        dropoff_qs: Sequence[int],
    ):
        for i in range(len(col_idx)):
            col_objs[col_idx[i]].active = False
        for i in range(len(row_idx)):
            row_objs[row_idx[i]].active = False
        for q_id in dropoff_qs:
            qubit_objs[q_id].array = 'SLM'

    def verify(
            self,
            col_objs: Sequence[Col],
            row_objs: Sequence[Row],
            qubit_objs: Sequence[Qubit],
            col_idx: Sequence[int],
            col_xs: Sequence[int],
            row_idx: Sequence[int],
            row_ys: Sequence[int],
            dropoff_qs: Sequence[int],
    ):
        a = len(col_idx)
        b = len(col_xs)
        if a != b:
            raise ValueError(
                f'{self.name}: col arguments invalid {a} idx, {b} xs')
        a = len(row_idx)
        b = len(row_ys)
        if a != b:
            raise ValueError(
                f'{self.name}: row arguments invalid {a} idx, {b} ys.')

        for i in range(len(col_idx)):
            if not col_objs[col_idx[i]].active:
                raise ValueError(
                    f'{self.name}: col {col_idx[i]} already dectivated.')
            for j in range(col_idx[i]):
                if (col_objs[j].active
                        and col_objs[j].x > col_xs[i] - AOD_SEP):
                    raise ValueError(
                        f'{self.name}: col {j} at x={col_objs[j].x} is '
                        f'too left for col {col_idx[i]} '
                        f'to deactivate at x={col_xs[i]}.')
            for j in range(col_idx[i]+1, len(col_objs)):
                if (col_objs[j].active
                        and col_objs[j].x - AOD_SEP < col_xs[i]):
                    raise ValueError(
                        f'{self.name}: col {j} at x={col_objs[j].x} is '
                        f'too right for col {col_idx[i]} '
                        f'to deactivate at x={col_xs[i]}.')
        for i in range(len(row_idx)):
            if not row_objs[row_idx[i]].active:
                raise ValueError(
                    f'{self.name}: row {row_idx[i]} already deactivated.')
            for j in range(row_idx[i]):
                if (row_objs[j].active
                        and row_objs[j].y > row_ys[i] - AOD_SEP):
                    raise ValueError(
                        f'{self.name}: row {j} at y={row_objs[j].y} is '
                        f'too high for row {row_idx[i]} '
                        f'to deactivate at y={row_ys[i]}.'
                    )
            for j in range(row_idx[i]+1, len(row_objs)):
                if (row_objs[j].active
                        and row_objs[j].y-AOD_SEP < row_ys[i]):
                    raise ValueError(
                        f'{self.name}: row {j} at y={col_objs[j].y} is '
                        f'too low for row {row_idx[i]} '
                        f'to deactivate at y={row_ys[i]}.'
                    )

        deactive_xys = []
        active_xs = [col.x for col in col_objs if col.active]
        for x in active_xs:
            for y in row_ys:
                deactive_xys.append((x, y))

        for q_id in range(len(qubit_objs)):
            if q_id in dropoff_qs:
                if (qubit_objs[q_id].x, qubit_objs[q_id].y) not in deactive_xys:
                    raise ValueError(
                        f'{self.name}: q {q_id} not dropped off from '
                        f'col {qubit_objs[q_id].c} row {qubit_objs[q_id].r} '
                        f'at x={qubit_objs[q_id].x} y={qubit_objs[q_id].y}.'
                    )
            elif qubit_objs[q_id].array == 'AOD':
                if (qubit_objs[q_id].x, qubit_objs[q_id].y) in deactive_xys:
                    raise ValueError(
                        f'{self.name}: q {q_id} wrongfully dropped off from '
                        f'col {qubit_objs[q_id].c} row {qubit_objs[q_id].r} '
                        f'at x={qubit_objs[q_id].x} y={qubit_objs[q_id].y}.'
                    )


class Rydberg(Inst):
    def __init__(self,
                 s: int,
                 col_objs: Sequence[Col],
                 row_objs: Sequence[Row],
                 qubit_objs: Sequence[Qubit],
                 gates: Sequence[Mapping[str, int]]):
        super().__init__('Rydberg', prefix=f'Rydberg_{s}', stage=s)
        self.verify(gates, qubit_objs)
        super().write_code(
            col_objs,
            row_objs,
            qubit_objs,
            {'gates': gates, 'duration': T_RYDBERG, }
        )

    def verify(
            self,
            gates: Sequence[Mapping[str, int]],
            qubit_objs: Sequence[Qubit]):
        # for g in gates:
        #     q0 = {'id': qubit_objs[g['q0']].id,
        #           'x': qubit_objs[g['q0']].x,
        #           'y': qubit_objs[g['q0']].y}
        #     q1 = {'id': qubit_objs[g['q1']].id,
        #           'x': qubit_objs[g['q1']].x,
        #           'y': qubit_objs[g['q1']].y}
        #     if (q0['x']-q1['x'])**2 + (q0['y']-q1['y'])**2 > R_B**2:
        #         raise ValueError(
        #             f"{self.name}: q{q0['id']} at x={q0['x']} y={q0['y']} "
        #             f"and q{q1['id']} at x={q1['x']} y={q1['y']} "
        #             f"are farther away than Rydberg range."
        #         )
        return


# class for big ops: ReloadRow, Reload, OffloadRow, Offload, SwapPair, Swap
# internally, these are lists of basic operations.

# todo: is there verification needed on the ComboInst lebvel?
class ComboInst:
    pass


class ComboInst():
    """class for combined instructions which is a sequence of combined
    instructions or DPQA instructions.
    """

    def __init__(
            self,
            type: str,
            prefix: Union[str, None] = None,
            suffix: Union[str, None] = None,
            stage: int = -1,
    ):
        """init method for combined instructions.

        Args:
            type (str): 
            prefix (str | None, optional): Defaults to None.
            suffix (str | None, optional): Defaults to None.
            stage (int, optional): Defaults to -1.
        """
        self.type = type
        self.name = (f'{prefix}:' if prefix else '') + \
            type + (f'_{suffix}' if suffix else '')
        self.stage = stage
        self.duration = -1
        self.insts = []

    def emit(self) -> Sequence[Mapping[str, Any]]:
        """combine the code of each Inst inside this ComboInst and return."""
        code = []
        for inst in self.insts:
            code += inst.emit()
        return code

    def emit_full(self) -> Sequence[Mapping[str, Any]]:
        code = []
        for inst in self.insts:
            code += inst.emit_full()
        return code

    def append_inst(self, inst: Union[Inst, ComboInst]):
        self.insts.append(inst)

    def prepend_inst(self, inst: Union[Inst, ComboInst]):
        self.insts.insert(0, inst)

    def is_trivial(self) -> bool:
        for inst in self.insts:
            if not inst.is_trivial():
                return False
        return True

    def remove_trivial_insts(self):
        nontrivial_insts = []
        for inst in self.insts:
            inst.remove_trivial_insts()
            if not inst.is_trivial():
                nontrivial_insts.append(inst)
        self.insts = nontrivial_insts


class ReloadRow(ComboInst):
    def __init__(self, s: int, r: int, prefix: str = ''):
        super().__init__('ReloadRow', prefix=prefix, suffix=str(r), stage=s)
        self.row_id = r
        self.moving_cols_id = []
        self.moving_cols_begin = []
        self.moving_cols_end = []

    def add_col_shift(self, id: int, begin: int, end: int):
        print("add col shift")
        self.moving_cols_id.append(id)
        self.moving_cols_begin.append(begin)
        self.moving_cols_end.append(end)

    def generate_col_shift(
            self,
            col_objs: Sequence[Col],
            row_objs: Sequence[Row],
            qubit_objs: Sequence[Qubit],
    ):  
        print("in generate_col_shift")
        print("self.moving_cols_id: ", self.moving_cols_id)
        print("self.moving_cols_begin: ", self.moving_cols_begin)
        print("self.moving_cols_end: ", self.moving_cols_end)
        self.insts.append(
            Move(s=self.stage,
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
        )

    def generate_row_activate(
            self,
            col_objs: Sequence[Col],
            row_objs: Sequence[Row],
            qubit_objs: Sequence[Qubit],
            cols: Sequence[int],
            xs: Sequence[int],
            y: int,
            pickup_qs: Sequence[int],
    ):
        self.insts.append(
            Activate(s=self.stage,
                     col_objs=col_objs,
                     row_objs=row_objs,
                     qubit_objs=qubit_objs,
                     col_idx=cols,
                     col_xs=xs,
                     row_idx=[self.row_id, ],
                     row_ys=[y, ],
                     pickup_qs=pickup_qs,
                     prefix=self.name)
        )

    def generate_parking(
            self,
            col_objs: Sequence[Col],
            row_objs: Sequence[Row],
            qubit_objs: Sequence[Qubit],
            shift_down: int,
            col_idx: Sequence[int] = [],
            col_begin: Sequence[int] = [],
            col_end: Sequence[int] = []
    ):
        self.insts.append(
            Move(s=self.stage,
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
        )


class Reload(ComboInst):
    def __init__(self, s: int):
        super().__init__('Reload', suffix=str(s), stage=s)

    def add_row_reload(self, r: int):
        self.insts.append(ReloadRow(self.stage, r, prefix=self.name))
        return self.insts[-1]


class OffloadRow(ComboInst):
    def __init__(self, s: int, r: int, prefix: str = ''):
        super().__init__('OffloadRow', prefix=prefix, suffix=str(r), stage=s)
        self.row_id = r
        self.moving_cols_id = []
        self.moving_cols_begin = []
        self.moving_cols_end = []

    def add_col_shift(self, id: int, begin: int, end: int):
        self.moving_cols_id.append(id)
        self.moving_cols_begin.append(begin)
        self.moving_cols_end.append(end)

    def generate_col_shift(
            self,
            col_objs: Sequence[Col],
            row_objs: Sequence[Row],
            qubit_objs: Sequence[Qubit],
    ):
        self.insts.append(
            Move(s=self.stage,
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
        )

    def generate_row_shift(
            self,
            col_objs: Sequence[Col],
            row_objs: Sequence[Row],
            qubit_objs: Sequence[Qubit],
            site_y: int,
    ):
        self.insts.append(
            Move(s=self.stage,
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
        )

    def generate_row_deactivate(
            self,
            col_objs: Sequence[Col],
            row_objs: Sequence[Row],
            qubit_objs: Sequence[Qubit],
            dropoff_qs: Sequence[int],
    ):
        self.insts.append(
            Deactivate(s=self.stage,
                       col_objs=col_objs,
                       row_objs=row_objs,
                       qubit_objs=qubit_objs,
                       col_idx=[],
                       col_xs=[],
                       row_idx=[self.row_id, ],
                       row_ys=[row_objs[self.row_id].y, ],
                       dropoff_qs=dropoff_qs,
                       prefix=self.name)
        )


class Offload(ComboInst):
    def __init__(self, s: int):
        super().__init__('Offload', suffix=str(s), stage=s)

    def add_row_offload(self, r: int):
        self.insts.append(OffloadRow(self.stage, r, prefix=self.name))
        return self.insts[-1]

    def all_cols_deactivate(
            self,
            col_objs: Sequence[Col],
            row_objs: Sequence[Row],
            qubit_objs: Sequence[Qubit],
    ):
        self.insts.append(
            Deactivate(
                s=self.stage,
                col_objs=col_objs,
                row_objs=row_objs,
                qubit_objs=qubit_objs,
                col_idx=[c.id for c in col_objs if c.active],
                col_xs=[c.x for c in col_objs if c.active],
                row_idx=[],
                row_ys=[],
                dropoff_qs=[],
                prefix=self.name
            )
        )


class SwapPair(ComboInst):
    """swap a pair of atoms A and B at the same interaction site
       O stands for an empty trap

            A  B

        step1: activate row 0 and col 0 on A to pick it up
            |
         ---A--B---
            |

        step2: move A to below B
               |
            O  B
               |
           ----A-----
               |

        step3: activate row 1 to pick up B
               |
         ---O--B---
               |
         ------A---
               |

        step4: move A and B to the place A used to be
            |
        ----B--O--
            |
        ----A-----
            |

        step5: deactivate row 1 to drop off B
            |
            B  O
            |
        ----A-----
            |

        step6: move A to the place B used to be
               |
         ---B--A---
               |

        step7: deactivate row 0 and col 0 to drop off A

            B  A
    """

    def __init__(self,
                 s: int,
                 col_objs: Sequence[Col],
                 row_objs: Sequence[Row],
                 qubit_objs: Sequence[Qubit],
                 left_q_id: int,
                 right_q_id: int,
                 prefix: str = ''
                 ):
        super().__init__(
            'SwapPair',
            prefix=prefix,
            suffix=f'({left_q_id},{right_q_id})',
            stage=s
        )

        # precondition
        left_x = qubit_objs[left_q_id].x
        right_x = qubit_objs[right_q_id].x
        y = qubit_objs[left_q_id].y
        qubit_objs[left_q_id].c = 0
        qubit_objs[left_q_id].r = 0
        qubit_objs[right_q_id].c = 0
        qubit_objs[right_q_id].r = 1

        self.insts = [
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


class Swap(ComboInst):
    def __init__(self, s: int):
        super().__init__('Swap', suffix=str(s), stage=s)

    def add_swap_pair(
            self,
            col_objs: Sequence[Col],
            row_objs: Sequence[Row],
            qubit_objs: Sequence[Qubit],
            q_id: int,
            qq_id: int
    ):
        left_q_id = q_id
        right_q_id = qq_id
        if qubit_objs[q_id].x > qubit_objs[qq_id].x:
            left_q_id = qq_id
            right_q_id = q_id
        self.insts.append(
            SwapPair(self.stage, col_objs, row_objs,
                     qubit_objs, left_q_id, right_q_id, self.name)
        )


class CodeGen():
    """Generate code files: json containing a list of dict, each one 
    corresponding to a DPQA instruction defined above. 
    """

    def __init__(
            self,
            file_name: str,
            no_transfer: bool = False,
            dir: str = None
    ):
        self.read_compiled(file_name)
        program = self.builder(no_transfer)

        if not dir:
            dir = './results/code/'
        self.code_full_file = dir + \
            (file_name.split('/')[-1]).replace('.json', '_code_full.json')
        with open(self.code_full_file, 'w') as f:
            json.dump(program.emit_full(), f)
        with open(self.code_full_file.replace('_code_full', '_code'), 'w') as f:
            json.dump(program.emit(), f)

    def read_compiled(self, compiled_file: str):
        with open(compiled_file, 'r') as f:
            data = json.load(f)
        self.n_q = data['n_q']
        self.x_high = data['n_x']
        self.y_high = data['n_y']
        self.c_high = data['n_c']
        self.r_high = data['n_r']
        self.layers = data['layers']
        self.n_t = len(self.layers)

        """change of convention. In solve() and the SMT model, a/c/r_s govern
        the movement from stage s to stage s+1, i.e.,
            ----------- x/y_0 
            | a/c/r_0 |
            ----------- x/y_1
            | a/c/r_1 |
            ----------- x/y_2
            | a/c/r_2 |

        However, the initial stage is very special in codegen and animation,
        so we generate code in this way:
           Rydberg_0  (special)  <------- x/y_0
           
           Swap_1 (optional)  <---. 
           Reload_1  <----------\  \ 
           BigMove_1  <--------- a/c/r_1
           Offload_1  <---------/ 
           Rydberg_1  <----------- x/y_1

               ...

        Thus, the movement between stage s and s+1 should be govened by a/c/r
        with subscript s+1, e.g., BigMove_1 uses a/c/r_1 above.
        So we need to shift the a/c/r variable values here.
        """
        for i in range(self.n_t-1, 0, -1):
            for q in range(self.n_q):
                self.layers[i]['qubits'][q]['a'] =\
                    self.layers[i - 1]['qubits'][q]['a']
                self.layers[i]['qubits'][q]['c'] =\
                    self.layers[i - 1]['qubits'][q]['c']
                self.layers[i]['qubits'][q]['r'] =\
                    self.layers[i - 1]['qubits'][q]['r']

        # infer some info of the AODs
        self.aod_from_compiled()

    def aod_from_compiled(self):
        for s, layer in enumerate(self.layers):
            if s == 0:
                continue
            layer['row'] = [{'id': i, 'qs': []}
                            for i in range(self.r_high-0)]
            layer['col'] = [{'id': i, 'qs': []}
                            for i in range(self.c_high-0)]
            prev_layer = self.layers[s-1]

            # figure out in the movement from stage s-1 to s:
            # - before the movement, where is each row 'y_begin'
            # - after the movement, where is each row 'y_end'
            # - what qubits are in this row 'qs'
            # similar for each AOD column
            for i, q in enumerate(layer['qubits']):
                if layer['qubits'][i]['a']:
                    layer['row'][q['r']]['y_begin'] =\
                        prev_layer['qubits'][i]['y']
                    layer['row'][q['r']]['y_end'] = q['y']
                    layer['row'][q['r']]['qs'].append(q['id'])
                    layer['col'][q['c']]['x_begin'] =\
                        prev_layer['qubits'][i]['x']
                    layer['col'][q['c']]['x_end'] = q['x']
                    layer['col'][q['c']]['qs'].append(q['id'])

            # figure out in the movement from stage s-1 to s:
            # - before the movement, which columns have site coord = X
            # - for all these cols, what is the relevant order from left 'offset'
            # - after the movement, which columns have site coord = X
            # - for all these cols, what is the relevant order from left 'offset'
            # similar for the AOD rows
            for case in ['_begin', '_end']:
                x_cols = []
                for x in range(self.x_high):
                    cols_at_x = []
                    for c in range(self.c_high):
                        if (layer['col'][c]['qs']
                                and layer['col'][c]['x' + case] == x):
                            cols_at_x.append(c)
                    for i, c in enumerate(cols_at_x):
                        layer['col'][c]['offset' + case] = i
                    x_cols.append(cols_at_x)
                layer['x_cols' + case] = x_cols
                y_rows = []
                for y in range(self.y_high):
                    rows_at_y = []
                    for r in range(self.r_high):
                        if (layer['row'][r]['qs']
                                and layer['row'][r]['y' + case] == y):
                            rows_at_y.append(r)
                    for i, r in enumerate(rows_at_y):
                        layer['row'][r]['offset' + case] = i
                    y_rows.append(rows_at_y)
                layer['y_rows' + case] = y_rows

    def builder(self, no_transfer: bool):
        qubits = [Qubit(i) for i in range(self.n_q)]
        rows = [Row(i) for i in range(self.r_high)]
        cols = [Col(i) for i in range(self.c_high)]
        program = ComboInst('Program')

        # read to comment in read_compiled() for structure of this method.
        init = self.builder_init(cols, rows, qubits, program)  # has Rydberg_0

        for s in range(1, len(self.layers)):
            print("s: ", s)
            print("self.layers[s]: ", self.layers[s])
            if s == 12:
                print("self.program]: ", program.emit_full()[-1])
            
            self.builder_swap(s, cols, rows, qubits, program)

            if (not no_transfer) or s == 1:
                # if we know there is not atom transfer, we can simply skip the
                # reload and offload procedures. However, we keep the first one
                # just for convenience of generating animations.
                self.builder_reload(s, cols, rows, qubits, program)

            self.builder_move(s, cols, rows, qubits, program)

            if not no_transfer:
                self.builder_offload(s, cols, rows, qubits, program)

            self.builder_rydberg(s, cols, rows, qubits, program, init)

        program.remove_trivial_insts()
        program.prepend_inst(init)
        return program

    def builder_init(
            self,
            cols: Sequence[Col],
            rows: Sequence[Row],
            qubits: Sequence[Qubit],
            program: ComboInst,
    ) -> Inst:
        slm_qubit_idx = list(range(self.n_q))  # put all qubits in SLM
        slm_qubit_xys = [(
            X_SITE_SEP * self.layers[0]['qubits'][i]['x'],
            Y_SITE_SEP * self.layers[0]['qubits'][i]['y']
        ) for i in range(self.n_q)]  # put all qubits in the left trap

        # when there are more than one qubit in a site at the beginning,
        # need to put one of them in the right trap.
        for g in self.layers[0]['gates']:
            a0 = self.layers[1]['qubits'][g['q0']]['a']
            a1 = self.layers[1]['qubits'][g['q1']]['a']
            x_left = X_SITE_SEP * self.layers[0]['qubits'][g['q0']]['x']
            x_right = x_left + SITE_WIDTH
            y = Y_SITE_SEP * self.layers[0]['qubits'][g['q0']]['y']

            # if both atoms are in AOD, use their column indices to decide
            # which one to put in the left trap and which one to the right
            # if they have the same col index, the order does not matter
            # in Reload, we will pick them up in different rows.
            if (a0 == 1
                and a1 == 1
                and self.layers[1]['qubits'][g['q0']]['c'] >
                    self.layers[1]['qubits'][g['q1']]['c']):
                slm_qubit_xys[g['q0']] = (x_right, y)
            else:
                slm_qubit_xys[g['q1']] = (x_right, y)

        init = Init(cols, rows, qubits,
                    slm_qubit_idx=slm_qubit_idx,
                    slm_qubit_xys=slm_qubit_xys,
                    data={
                        'n_q': self.n_q,
                        'x_high': self.x_high,
                        'y_high': self.y_high,
                        'c_high': self.c_high,
                        'r_high': self.r_high,
                    })

        self.builder_rydberg(0, cols, rows, qubits, program, init)

        return init

    def builder_rydberg(
            self,
            s: int,
            cols: Sequence[Col],
            rows: Sequence[Row],
            qubits: Sequence[Qubit],
            program: ComboInst,
            init: Inst
    ):
        program.append_inst(
            Rydberg(s, cols, rows, qubits, self.layers[s]['gates'])
        )
        init.add_slms([(q.x, q.y) for q in qubits])

    def builder_swap(
            self,
            s: int,
            cols: Sequence[Col],
            rows: Sequence[Row],
            qubits: Sequence[Qubit],
            program: ComboInst,
    ):
        swap_obj = Swap(s)
        prev_layer = self.layers[s-1]
        this_layer = self.layers[s]
        for q0_id in range(self.n_q):
            for q1_id in range(q0_id+1, self.n_q):
                q0_a = this_layer['qubits'][q0_id]['a']
                q1_a = this_layer['qubits'][q1_id]['a']
                if q0_a == 1 and q1_a == 1:
                    q0_x = prev_layer['qubits'][q0_id]['x']
                    q1_x = prev_layer['qubits'][q1_id]['x']
                    q0_y = prev_layer['qubits'][q0_id]['y']
                    q1_y = prev_layer['qubits'][q1_id]['y']
                    q0_c = this_layer['qubits'][q0_id]['c']
                    q1_c = this_layer['qubits'][q1_id]['c']
                    q0_r = this_layer['qubits'][q0_id]['r']
                    q1_r = this_layer['qubits'][q1_id]['r']
                    # if two qubits are at the same site and
                    # both being picked up in the same row
                    if (q0_x, q0_y, q0_r) == (q1_x, q1_y, q1_r):
                        # if their position and col indeces are in reverse order
                        if (
                            (q0_c > q1_c
                             and qubits[q0_id].x < qubits[q1_id].x
                             )
                            or (q0_c < q1_c
                                and qubits[q0_id].x > qubits[q1_id].x
                                )
                        ):
                            swap_obj.add_swap_pair(cols, rows, qubits,
                                                   q0_id, q1_id)
        program.append_inst(swap_obj)

    def builder_move(
            self,
            s: int,
            cols: Sequence[Col],
            rows: Sequence[Row],
            qubits: Sequence[Qubit],
            program: ComboInst,
    ):
        """to avoid collision, the big moves will end in a slightly adjusted
        location: 1 um to +X direction and AOD_SEP(2) um to +Y direction
        Suppose the O's are the two traps in an interaction site
                O--------O
        The atoms will finish the big move in
                O--------O
             -----A-----
                  -----B-----
        (+X is ->, +Y is down, 1um is --, and AOD_SEP is one line height)
        There can be other cases, e.g., A and B are in the same column
                O--------O
             -----A-----
             -----B-----
        Or, A and B are in the same row
                O--------O
             -----A---B------
        """

        col_idx = []
        col_begin = []
        col_end = []
        for col_id in range(self.c_high):
            if cols[col_id].active:
                col_idx.append(col_id)
                col_begin.append(cols[col_id].x)
                site_x = self.layers[s]['col'][col_id]['x_end']
                offset = self.layers[s]['col'][col_id]['offset_end']
                col_end.append(1 + site_x * X_SITE_SEP + AOD_SEP * offset)

        row_idx = []
        row_begin = []
        row_end = []
        for row_id in range(self.r_high):
            if rows[row_id].active:
                row_idx.append(row_id)
                row_begin.append(rows[row_id].y)
                site_y = self.layers[s]['row'][row_id]['y_end']
                offset = self.layers[s]['row'][row_id]['offset_end']
                row_end.append(site_y * Y_SITE_SEP + AOD_SEP * (1 + offset))

        program.append_inst(
            Move(s=s,
                 col_objs=cols,
                 row_objs=rows,
                 qubit_objs=qubits,
                 col_idx=col_idx,
                 col_begin=col_begin,
                 col_end=col_end,
                 row_idx=row_idx,
                 row_begin=row_begin,
                 row_end=row_end,
                 prefix=f'BigMove_{s}')
        )

    def builder_reload(
            self,
            s: int,
            cols: Sequence[Col],
            rows: Sequence[Row],
            qubits: Sequence[Qubit],
            program: ComboInst,
    ):
        layer = self.layers[s]
        prev_layer = self.layers[s-1]
        reload_obj = Reload(s)
        # reload row by row
        for row_id in range(self.r_high):
            if layer['row'][row_id]['qs']:  # there are qubits to load
                print("\n\nrow_id: ", row_id)
                print("load: ", layer['row'][row_id]['qs'])
                reloadRow_obj = reload_obj.add_row_reload(row_id)
                pickup_qs = []
                cols_to_active = []
                x_to_activate = []
                # consider the movements in a row of sites
                for site_x in range(self.x_high):
                    site_qs = []
                    for q_id in range(len(qubits)):
                        # find out the qubits with site_x and in row_id
                        if (layer['qubits'][q_id]['a'] == 1
                            and prev_layer['qubits'][q_id]['x'] == site_x
                                and layer['qubits'][q_id]['r'] == row_id):
                            site_qs.append(q_id)
                            qubits[q_id].r = row_id
                            qubits[q_id].c = layer['qubits'][q_id]['c']

                    # shift the 1 or 2 cols that are picking up qubits
                    if len(site_qs) == 2:
                        print("case 1663")
                        # which qubit is on the left and which on the right
                        [q_id_left, q_id_right] = site_qs
                        if layer['qubits'][q_id_left]['c'] >\
                                layer['qubits'][q_id_right]['c']:
                            tmp = q_id_left
                            q_id_left = q_id_right
                            q_id_right = tmp

                        # current location of Cols
                        col_id_left = layer['qubits'][q_id_left]['c']
                        col_id_right = layer['qubits'][q_id_right]['c']
                        lower_offset =\
                            layer['col'][col_id_left]['offset_begin']
                        upper_offset =\
                            layer['col'][col_id_right]['offset_begin']

                        # target locations of Cols
                        lower_x = qubits[q_id_left].x
                        upper_x = qubits[q_id_right].x

                        # process the Col on the left
                        if not cols[col_id_left].active:
                            cols_to_active.append(col_id_left)
                            x_to_activate.append(lower_x)
                        else:
                            reloadRow_obj.add_col_shift(
                                id=col_id_left,
                                begin=cols[col_id_left].x,
                                end=lower_x)

                        # process the Col on the right
                        if not cols[col_id_right].active:
                            cols_to_active.append(col_id_right)
                            x_to_activate.append(upper_x)
                        else:
                            reloadRow_obj.add_col_shift(
                                id=col_id_right,
                                begin=cols[col_id_right].x,
                                end=upper_x)

                    elif len(site_qs) == 1:
                        print("case 1705")
                        # for convience later on, we still keep the *_left and
                        # *_right vars even if there is one qubit to pick up
                        q_id = site_qs[0]
                        col_id_left = layer['qubits'][q_id]['c']
                        col_id_right = col_id_left
                        lower_offset =\
                            layer['col'][col_id_left]['offset_begin']
                        upper_offset = lower_offset
                        lower_x = qubits[q_id].x
                        upper_x = lower_x
                        if not cols[col_id_left].active:
                            print("case 1717")
                            cols_to_active.append(col_id_left)
                            x_to_activate.append(lower_x)
                        else:  # col already active, shift it to align with q_id
                            print("case 1721, begin: {}, end: {}".format(cols[col_id_left].x, lower_x))
                            reloadRow_obj.add_col_shift(
                                id=col_id_left,
                                begin=cols[col_id_left].x,
                                end=lower_x)

                    elif len(site_qs) > 2:
                        raise ValueError(
                            f"builder reload {s} row {row_id} site {site_x}:"
                            f" more than 2 qubits"
                        )
                    else:
                        continue

                    # shift other Cols that are already activated. Those Cols
                    # may not be picking up any qubit in this row, but due to
                    # AOD order constraints, we need to shift them to have the
                    # correct order when we shift the Cols that indeed are
                    # picking up some qubit in this row.
                    for col_id in layer['x_cols_begin'][site_x]:
                        # the Cols at site_x in the beginning between stage s-1
                        # and s (since we are processing Reload_s now)

                        if (cols[col_id].active
                            and col_id != col_id_left
                                and col_id != col_id_right):
                            # if this Col is not the one involved in loading

                            # if there is a col on the right of the cols for loading
                            if layer['col'][col_id]['offset_begin'] >\
                                    upper_offset:
                                print("case 1757")
                                if layer['col'][col_id]['offset_begin'] - upper_offset == 1:
                                    reloadRow_obj.add_col_shift(
                                    id=col_id,
                                    begin=cols[col_id].x,
                                    end=upper_x + AOD_SEP)
                                elif layer['col'][col_id]['offset_begin'] - upper_offset == 2:
                                    reloadRow_obj.add_col_shift(
                                    id=col_id,
                                    begin=cols[col_id].x,
                                    end=upper_x + AOD_SEP * 3)
                                

                            # if there is a col on the left of the cols for loading
                            elif layer['col'][col_id]['offset_begin'] <\
                                    lower_offset:
                                print("case 1767")
                                print("col_id: {}, col_id_left: {}, col_id_right: {}", col_id, col_id_left, col_id_right)
                                if layer['col'][col_id]['offset_begin'] - lower_offset == -1:
                                    reloadRow_obj.add_col_shift(
                                    id=col_id,
                                    begin=cols[col_id].x,
                                    end=lower_x - AOD_SEP)
                                elif layer['col'][col_id]['offset_begin'] - lower_offset == -2:
                                    reloadRow_obj.add_col_shift(
                                    id=col_id,
                                    begin=cols[col_id].x,
                                    end=lower_x - 3 * AOD_SEP)
                                
                            # if there is a col in the middle of the cols for loading
                            else:
                                print("case 1777")
                                reloadRow_obj.add_col_shift(
                                    id=col_id,
                                    begin=cols[col_id].x,
                                    end=lower_x + AOD_SEP *
                                    (layer['col'][col_id]['offset_begin'] -
                                     lower_offset))

                    pickup_qs += site_qs

                print("cols_to_active: ", cols_to_active)
                print("x_to_activate: ", x_to_activate)
                print("layer['row'][row_id]['y_begin']*Y_SITE_SEP: ", layer['row'][row_id]['y_begin']*Y_SITE_SEP)
                print("active_xs_id: ", [i for i, col in enumerate(cols) if col.active])
                print("active_xs: ", [col.x for col in cols if col.active])
                print("ori col x x: ", [cols[col].x for col in cols_to_active])
                # apply all the col shifts added previously
                reloadRow_obj.generate_col_shift(cols, rows, qubits)
                print("line 1783")
                print("active_xs_id: ", [i for i, col in enumerate(cols) if col.active])
                print("active_xs: ", [col.x for col in cols if col.active])
                reloadRow_obj.generate_row_activate(
                    cols,
                    rows,
                    qubits,
                    cols_to_active,
                    x_to_activate,
                    layer['row'][row_id]['y_begin']*Y_SITE_SEP,
                    pickup_qs
                )

                # shift down the finished row because later on, some other row
                # may need to adjust the Cols again and if we keep this row
                # as is, some qubits may collid into each other since at each
                # site, the y of two SLM traps are the same.
                num_rows = len(layer['y_rows_begin']
                               [layer['row'][row_id]['y_begin']])
                shift_down = (
                    num_rows - layer['row'][row_id]['offset_begin'])*AOD_SEP
                # Also shift the Cols by 1 to avoid collisions.
                col_idx = [col_id for col_id,
                           col in enumerate(cols) if col.active]
                col_begin = [cols[col_id].x for col_id in col_idx]
                col_end = [1 + layer['col'][col_id]['x_begin'] * X_SITE_SEP +
                           AOD_SEP * layer['col'][col_id]['offset_begin']
                           for col_id in col_idx]
                print("col_begin: ", col_begin)
                print("col_end: ", col_end)
                reloadRow_obj.generate_parking(
                    cols,
                    rows,
                    qubits,
                    shift_down,
                    col_idx,
                    col_begin,
                    col_end
                )

        program.append_inst(reload_obj)

    def builder_offload(
            self,
            s: int,
            cols: Sequence[Col],
            rows: Sequence[Row],
            qubits: Sequence[Qubit],
            program: ComboInst,
    ):
        offload_obj = Offload(s)
        layer = self.layers[s]
        # the row-by-row processing is quite similar to Reload
        for row_id in range(self.r_high):
            if rows[row_id].active:
                dropoff_qs = []
                offloadRow_obj = offload_obj.add_row_offload(row_id)
                for site_x in range(self.x_high):
                    site_q_slm = []
                    site_q_aod = []
                    for q_id, q in enumerate(layer['qubits']):
                        if (
                            q['x'], q['y']
                        ) == (
                            site_x, layer['row'][row_id]['y_end']
                        ):
                            if (qubits[q_id].array == 'AOD'
                                    and q['r'] == row_id):
                                dropoff_qs.append(q_id)
                                site_q_aod.append(q_id)
                            if qubits[q_id].array == 'SLM':
                                site_q_slm.append(q_id)
                    if len(site_q_aod) == 2:
                        [q_id_left, q_id_right] = site_q_aod
                        if layer['qubits'][q_id_left]['c'] >\
                                layer['qubits'][q_id_right]['c']:
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
                            id=col_id_left,
                            begin=qubits[q_id_left].x,
                            end=lower_x
                        )
                        offloadRow_obj.add_col_shift(
                            id=col_id_right,
                            begin=qubits[q_id_right].x,
                            end=upper_x
                        )
                    elif len(site_q_aod) == 1:
                        q_id = site_q_aod[0]
                        col_id_left = layer['qubits'][q_id]['c']
                        col_id_right = col_id_left
                        lower_offset = layer['col'][col_id_left]['offset_end']
                        upper_offset = lower_offset
                        lower_x = X_SITE_SEP*site_x
                        if site_q_slm:
                            lower_x = 2*X_SITE_SEP*site_x + \
                                SITE_WIDTH - qubits[site_q_slm[0]].x
                        upper_x = lower_x
                        offloadRow_obj.add_col_shift(
                            id=col_id_left,
                            begin=qubits[q_id].x,
                            end=lower_x
                        )
                    elif len(site_q_aod) > 2:
                        raise ValueError(
                            f"builder offload {s} row {row_id} site {site_x}:"
                            f" more than 2 qubits"
                        )
                    else:
                        continue

                    for col_id in layer['x_cols_end'][site_x]:
                        if (cols[col_id].active
                            and col_id != col_id_left
                                and col_id != col_id_right):
                            if layer['col'][col_id]['offset_end'] >\
                                    upper_offset:
                                offloadRow_obj.add_col_shift(
                                    id=col_id,
                                    begin=cols[col_id].x,
                                    end=upper_x + AOD_SEP *
                                    (layer['col'][col_id]['offset_end'] -
                                     upper_offset) + 1
                                )
                            elif layer['col'][col_id]['offset_end'] <\
                                    lower_offset:
                                offloadRow_obj.add_col_shift(
                                    id=col_id,
                                    begin=cols[col_id].x,
                                    end=lower_x + AOD_SEP *
                                    (layer['col'][col_id]['offset_end'] -
                                     lower_offset) - 1
                                )
                            else:
                                offloadRow_obj.add_col_shift(
                                    id=col_id,
                                    begin=cols[col_id].x,
                                    end=lower_x + AOD_SEP *
                                    (layer['col'][col_id]['offset_end'] -
                                     lower_offset)
                                )

                # align the Cols to the correct locatios
                offloadRow_obj.generate_col_shift(cols, rows, qubits)
                # the rows are at the parked location finishing a BigMove,
                # so we need to shift them to align with the SLM traps.
                offloadRow_obj.generate_row_shift(
                    cols,
                    rows,
                    qubits,
                    layer['row'][row_id]['y_end']
                )
                offloadRow_obj.generate_row_deactivate(
                    cols, rows, qubits, dropoff_qs)
        offload_obj.all_cols_deactivate(cols, rows, qubits)
        program.append_inst(offload_obj)


class Animator():
    """generate animation movie from code_full files."""

    def __init__(self,
                 code_file_name: str,
                 scaling_factor: int = PT_MICRON,
                 font: int = 10,
                 ffmpeg: str = 'ffmpeg',
                 real_speed: bool = False,
                 show_graph: bool = False,
                 edges: Union[Sequence[Sequence[int]], None] = None,
                 dir: Union[str, None] = None
                 ):
        """
        Args:
            code_file_name (str): file name of code_full generated by CodeGen.
            scaling_factor (int, optional): the unit scaling factor between the
             animation and um. Defaults to PT_MICRON.
            font (int, optional): font size in the animation. Defaults to 10.
            ffmpeg (str, optional): path of ffmpeg. Defaults to 'ffmpeg'.
            real_speed (bool, optional): whether to use real speed in the
                movements in Relaod and Offload. Defaults to False.
            show_graph (bool, optional): whether show the graph on the right
                side of the DPQA in the animation. Defaults to False.
            edges (Sequence[Sequence[int]] | None, optional): the edges of the
                graph. Defaults to None.
            dir (str | None, optional): dir to save output. Defaults to None.
        """
        matplotlib.use('Agg')
        matplotlib.rcParams.update({'font.size': font})
        plt.rcParams['animation.ffmpeg_path'] = ffmpeg
        self.show_graph = show_graph
        self.graph_edges = edges
        self.read_files(code_file_name)

        self.keyframes = []
        self.setup_canvas(scaling_factor)
        self.create_master_schedule(real_speed)

        anim = FuncAnimation(
            self.fig,
            self.update,
            init_func=self.update_init,
            frames=self.keyframes[-1],
            interval=1000/FPS
        )

        animation_file = dir + (code_file_name.replace(
            '_code_full.json', '.mp4')).split('/')[-1]
        anim.save(animation_file, writer=FFMpegWriter(FPS))

    def read_files(self, code_file: str):
        with open(code_file, 'r') as f:
            self.code = json.load(f)

        self.n_q = self.code[0]['n_q']
        self.x_high = self.code[0]['x_high']
        self.y_high = self.code[0]['y_high']
        self.c_high = self.code[0]['c_high']
        self.r_high = self.code[0]['r_high']

    def create_master_schedule(self, real_speed: bool):
        """create a list of keyframes. We need this to calculate how many
        frames are there in total when we call FuncAnimation."""

        frame = 0  # the beginning frame of the Inst currently considered
        for inst in self.code:
            # if not using real speed, make all the movements in Reload
            # and Offload 1 frame
            if not real_speed:
                if inst['type'] == 'Move' and 'BigMove' not in inst['name']:
                    inst['duration'] = 1

            # Rydberg interaction is much shorter compared to the movements,
            # so using real speed, we will never see Rydberg.
            if inst['type'] == 'Rydberg':
                inst['duration'] = MUS_PER_FRM * 8  # i.e., 8 frames

            # Activate and Deactivate is on par with some movements in terms of
            # duration, but we do not have ramping-up or -down animations yet,
            # so we opt for them taking 1 frame.
            if inst['type'] == 'Activate' or inst['type'] == 'Deactivate':
                inst['duration'] = MUS_PER_FRM

            inst['f_begin'] = frame
            # new_frame = frame + ceil(duration / MUS_PER_FRM)
            new_frame = frame +\
                int((inst['duration'] + MUS_PER_FRM - 1)/MUS_PER_FRM)
            # new_frame is the begining of the next Inst, so end of this one is
            inst['f_end'] = new_frame-1
            self.keyframes.append(new_frame)
            frame = new_frame

    """layout of the DPQA plot:
          ____________________________________________________________________
          |                                  ^                               |
          |                        Y_HIGH_PAD|                    X_HIGH_PAD |
        y |                                  V                  |<---------->|
         -|-         o   o              o   o--             o   o            |
        t |                                                 |<->|            |
        i |                   ...                    .   X_SITE_WIDTH        |
        c |                                         .                        |
        k |                                        .                         |
         -|-         o   o              o   o             --o   o            |
          |          |<---X_SITE_SEP--->|                  ^                 |
          |                                        ...     |Y_SITE_SEP       |
          |                                                V                 |
         -|-         o   o              o   o--           --o   o            |
          |          |                       ^                               |
          | X_LOW_PAD|              Y_LOW_PAD|                               |
          |<-------->|                       |                               |
          |____________|__________________|__V________________|______________|
                       |                  |                   |
                          x tick
        """

    def setup_canvas(self, scaling_factor: int):
        """set up various objects before actually drawing."""

        # unit conversion factor from um to pt
        px = 1/plt.rcParams['figure.dpi'] * scaling_factor

        self.X_LOW = -X_LOW_PAD
        self.X_HIGH = SITE_WIDTH + \
            (self.x_high-1)*X_SITE_SEP + X_HIGH_PAD
        self.Y_LOW = -Y_LOW_PAD
        self.Y_HIGH = (self.y_high-1)*Y_SITE_SEP + Y_HIGH_PAD

        if self.show_graph:
            # 1 row 2 col, DPQA on the left, the graph on the right
            self.fig, (self.ax, self.network_ax) = plt.subplots(
                1,
                2,
                gridspec_kw={'width_ratios': [3, 1]},
                figsize=((self.X_HIGH-self.X_LOW)*px*4/3,
                         (self.Y_HIGH-self.Y_LOW)*px)
            )
            self.fig.tight_layout()
            self.network = nx.Graph()
            self.network_ax.set_title('The 3-regular graph')
        else:
            self.fig, self.ax, = plt.subplots(
                figsize=((self.X_HIGH-self.X_LOW)*px,
                         (self.Y_HIGH-self.Y_LOW)*px))

        self.title = self.ax.set_title('')
        self.ax.set_xlim([self.X_LOW, self.X_HIGH])
        self.ax.set_xticks(
            [SITE_WIDTH/2+X_SITE_SEP*i for i in range(self.x_high)])
        self.ax.set_xticklabels([i for i in range(self.x_high)])
        self.ax.set_ylim(
            [self.Y_LOW, self.Y_HIGH])
        self.ax.set_yticks(
            [Y_SITE_SEP*i for i in range(self.y_high)])
        self.ax.set_yticklabels([i for i in range(self.y_high)])

        # draw all the SLMs used throught out the computation
        # slm_xs = [slm[0] for slm in self.code[0]['all_slms']]
        # slm_ys = [slm[1] for slm in self.code[0]['all_slms']]
        # self.ax.scatter(slm_xs, slm_ys, marker='o', s=50,
        #                 facecolor='none', edgecolor=(0, 0, 1, 0.5))

    def update_init(self):
        # init the Qubits, Cols, and Rows. Later just update their attributes
        self.qubit_scat = self.ax.scatter([0 for _ in range(self.n_q)],
                                          [0 for _ in range(self.n_q)])
        self.qubit_scat.set_color('b')
        self.col_plots = [self.ax.axvline(0, self.Y_LOW, self.Y_HIGH, c=(
            1, 0, 0, 0), ls='--') for _ in range(self.c_high)]
        self.row_plots = [self.ax.axhline(0, self.X_LOW, self.X_HIGH, c=(
            1, 0, 0, 0), ls='--') for _ in range(self.c_high)]

        # draw the graph with black thin edges
        if self.show_graph:
            self.network.add_edges_from(self.graph_edges)
            self.pos = nx.spring_layout(self.network)
            nx.draw(
                self.network,
                pos=self.pos,
                ax=self.network_ax,
                with_labels=True,
                edgecolors='black',
                node_color='lightgray'
            )
        return

    def update(self, f: int):  # f is the frame
        if f < self.keyframes[0]:
            return
        for i, inst in enumerate(self.code):
            if f >= self.keyframes[i-1] and f < self.keyframes[i]:
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

    def update_rydberg(self, f: int, inst: dict):
        edges = [(g['q0'], g['q1']) for g in inst['gates']]
        if f == inst['f_begin']:
            self.title.set_text(inst['name'])

            # find the qubits involved in 2Q gates and annotate their ids
            active_qubits = []
            for g in inst['gates']:
                active_qubits += [g['q0'], g['q1']]
            self.texts = [
                self.ax.text(inst['state']['qubits'][q_id]['x'] + 1,
                             inst['state']['qubits'][q_id]['y'] + 1,
                             q_id) for q_id in active_qubits
            ]

            # whole plane lights up (alpha=.2 blue)
            self.ax.set_facecolor((0, 0, 1, 0.2))

            # draw SLM qubits in blue and at the correct locations
            self.qubit_scat.set_offsets(
                [(q['x'], q['y']) for q in inst['state']['qubits']])

        if self.show_graph and f == int((inst['f_begin'] + inst['f_end'])/2):
            # draw some edges in the graph (each corresponding to a 2Q gate
            # executed at this Rydberg stage) in thicker blue
            nx.draw_networkx_edges(
                self.network,
                pos=self.pos,
                edgelist=edges,
                edge_color='blue',
                ax=self.network_ax,
                width=4
            )

        if f == inst['f_end']:
            # clean up the annotations and blue background at finishing frame
            self.ax.set_facecolor('w')
            for text in self.texts:
                text.remove()

    def interpolate(self, progress: int, duration: int, begin: int, end: int):
        """implement cubic interpolation per Bluvstein et al.
        Suppose we want to move from x=begin at t=0 to x=end at t=duration
        if we are at t=progress*duration, return our x. (0<= progress <=1)
        """

        D = end - begin
        if D == 0:
            return begin
        r = (1+progress)/duration
        return begin + 3*D*(r**2) - 2*D*(r**3)

    def update_move(self, f: int, inst: dict, prev_state: dict):
        if f == inst['f_begin']:
            self.title.set_text(inst['name'])

        # needs to change location every frame, as below
        progress = f - inst['f_begin']
        duration = inst['f_end'] - inst['f_begin'] + 1

        col_xs = [col['x'] for col in prev_state['cols']]
        for col in inst['cols']:
            curr_x = self.interpolate(
                progress, duration, col['begin'], col['end'])
            col_xs[col['id']] = curr_x
            self.col_plots[col['id']].set_xdata((col_xs[col['id']], ))

        row_ys = [row['y'] for row in prev_state['rows']]
        for row in inst['rows']:
            curr_y = self.interpolate(
                progress, duration, row['begin'], row['end'])
            row_ys[row['id']] = curr_y
            self.row_plots[row['id']].set_ydata((row_ys[row['id']], ))

        q_xs = [q['x'] for q in prev_state['qubits']]
        q_ys = [q['y'] for q in prev_state['qubits']]
        for q_id in range(self.n_q):
            if prev_state['qubits'][q_id]['array'] == 'AOD':
                q_xs[q_id] = col_xs[prev_state['qubits'][q_id]['c']]
                q_ys[q_id] = row_ys[prev_state['qubits'][q_id]['r']]
        self.qubit_scat.set_offsets([(q_xs[i], q_ys[i])
                                    for i in range(self.n_q)])
        return

    def update_activate(self, f: int, inst: dict):
        if f == inst['f_begin']:
            self.title.set_text(inst['name'])

            # set Cols/Rows to activate to the correct location and turn red.2
            for id, col in enumerate(inst['col_idx']):
                self.col_plots[col].set_xdata((inst['col_xs'][id], ))
                self.col_plots[col].set_color((1, 0, 0, 0.2))

            for id, row in enumerate(inst['row_idx']):
                self.row_plots[row].set_ydata((inst['row_ys'][id], ))
                self.row_plots[row].set_color((1, 0, 0, 0.2))

            # SLM qubits remain blue while qubits being picked up turns red
            self.qubit_scat.set_color(
                ['b' if q['array'] == 'SLM' else 'r'
                 for q in inst['state']['qubits']])

    def update_deactivate(self, f: int, inst: dict):
        if f == inst['f_end']:
            self.title.set_text(inst['name'])

            self.qubit_scat.set_color(
                ['b' if q['array'] == 'SLM' else 'r'
                 for q in inst['state']['qubits']])

            # deactivate Cols/Rows by setting alpha=0
            for col in inst['col_idx']:
                self.col_plots[col].set_color((1, 0, 0, 0))
            for row in inst['row_idx']:
                self.row_plots[row].set_color((1, 0, 0, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument(
        '--scaling', help='scaling factor of the animation', type=int)
    parser.add_argument('--font', help='font size in the animation', type=int)
    parser.add_argument('--ffmpeg', help='custom ffmpeg path', type=str)
    parser.add_argument(
        '--realSpeed',
        help='real speed in the animation of reload and offload procedures',
        action='store_true')
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
