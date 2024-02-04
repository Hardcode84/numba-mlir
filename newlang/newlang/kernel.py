
from itertools import product
import numpy as np
from numpy import ma
from collections.abc import Iterable

DEF_GROUP_SHAPE = (64,1,1)
DEF_SUBGROUP_SIZE = 16

def _get_uninit_value(dtype):
    return 0

def _divup(a, b):
    return (a + b - 1) // b

class Group:
    def __init__(self, work_shape, group_shape_hint = DEF_GROUP_SHAPE, subgroup_size_hint = DEF_SUBGROUP_SIZE):
        self.work_shape = work_shape
        self.group_shape = group_shape_hint
        self.subgroup_size = subgroup_size_hint

    def get_num_groups(self):
        return tuple(_divup(a, b) for a, b in zip(self.work_shape, self.group_shape))

class CurrentSubGroup:
    def __init__(self, size):
        self._size = size
        self._subgroup_id = None

    def size(self):
        return self._size

    def subgroup_id(self):
        return self._subgroup_id


class CurrentGroup:
    def __init__(self, group_shape, subgroup_size):
        self._group_shape = group_shape
        self._subgroup_size = subgroup_size
        self._subgroup_ranges = (range(group_shape[0]), range(group_shape[1]), range(_divup(group_shape[2], subgroup_size)))
        self._group_id = None
        self._subgroup = CurrentSubGroup(subgroup_size)

    def group_id(self):
        return self._group_id

    def group_shape(self):
        return self._group_shape

    def work_offset(self):
        return tuple(a * b for a,b in zip(self._group_shape, self._group_id))

    def load(self, array, shape):
        if not isinstance(shape, Iterable):
            shape = (shape,)

        dtype = array.dtype
        init = _get_uninit_value(dtype)
        res = ma.masked_array(np.full(shape, fill_value=init, dtype=dtype), mask=True)
        s = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(array.shape, shape))
        res[s] = array[s]
        return res

    def store(self, dst, src):
        s = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(src.shape, dst.shape))
        dst[s] = np.where(src.mask[s], dst[s], src[s])

    def empty(self, shape, dtype):
        init = _get_uninit_value(dtype)
        return ma.masked_array(np.full(shape, fill_value=init, dtype=dtype), mask=False)

    def subgroups(self, func):
        def _wrapper():
            for sgid in product(*self._subgroup_ranges):
                sg = self._subgroup
                sg._subgroup_id = sgid
                func(sg)

        return _wrapper

def kernel(func):
    def wrapper(group, *args, **kwargs):
        n_groups = group.get_num_groups()
        cg = CurrentGroup(group.group_shape, group.subgroup_size)
        for gid in product(*(range(g) for g in n_groups)):
            cg._group_id = gid
            func(cg, *args, **kwargs)


    return wrapper
