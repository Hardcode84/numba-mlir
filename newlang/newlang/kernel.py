
from itertools import product
import numpy as np
from numpy import ma
from collections.abc import Iterable

DEF_GROUP_SHAPE = (64,1,1)
DEF_SUBGROUP_SIZE = 16

def _get_uninit_value(dtype):
    return 0

class Group:
    def __init__(self, work_shape, group_shape_hint = DEF_GROUP_SHAPE, subgroup_size_hint = DEF_SUBGROUP_SIZE):
        self.work_shape = work_shape
        self.group_shape = group_shape_hint
        self.subgroup_size = subgroup_size_hint

    def get_num_groups(self):
        return tuple((a + b - 1) // b for a, b in zip(self.work_shape, self.group_shape))

class CurrentGroup:
    def __init__(self, group_shape):
        self._group_shape = group_shape
        self._group_id = None

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

def kernel(func):
    def wrapper(group, *args, **kwargs):
        n_groups = group.get_num_groups()
        cg = CurrentGroup(group.group_shape)
        for gid in product(*(range(g) for g in n_groups)):
            cg._group_id = gid
            func(cg, *args, **kwargs)


    return wrapper
