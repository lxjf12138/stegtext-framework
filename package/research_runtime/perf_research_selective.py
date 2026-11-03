"""Avoid full-vocabulary suffix-table construction for sparse/cold accesses."""
from collections import OrderedDict

import numpy as np

from perf_research_planner import make_jump_planner


class AccessTable:
    def __init__(self, values, cut):
        self.values = values; self.cut = cut
        self.full = None
        self.accessed = 0

    def __getitem__(self, positions):
        positions = np.asarray(positions, dtype=np.int64)
        if self.full is not None:
            return self.full[positions]
        self.accessed += len(positions)
        # Eager materialization pays off only after enough actual accesses.
        # This is a STORAGE policy, not posterior/support pruning.
        if self.accessed >= len(self.values) or len(positions)*3 >= len(self.values):
            self.full = np.fromiter((v[self.cut:] for v in self.values), dtype=object, count=len(self.values))
            return self.full[positions]
        return np.fromiter((self.values[int(i)][self.cut:] for i in positions), dtype=object, count=len(positions))


class SelectiveTables:
    def __init__(self, capacity=12):
        self.capacity = capacity; self.tables = OrderedDict()

    def get(self, values, cut):
        key = (id(values), cut)
        table = self.tables.get(key)
        if table is None or table.values is not values:
            table = AccessTable(values, cut)
            self.tables[key] = table
            while len(self.tables) > self.capacity:
                self.tables.popitem(last=False)
        else:
            self.tables.move_to_end(key)
        return table


def make_selective_planner(index):
    planner = make_jump_planner(index)
    planner.tables = SelectiveTables()
    return planner
