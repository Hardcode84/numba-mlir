
from collections.abc import Iterable
from functools import partial
from greenlet import greenlet
from itertools import product
from numpy import ma
import numpy as np

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
    def __init__(self, size, subgroup_id):
        self._size = size
        self._subgroup_id = subgroup_id

    def size(self):
        return self._size

    def subgroup_id(self):
        return self._subgroup_id


class CurrentWorkitem:
    def __init__(self, lid, gid):
        self._local_id = lid
        self._global_id = gid

    def local_id(self):
        return self._local_id

    def global_id(self):
        return self._global_id


class CurrentGroup:
    def __init__(self, group_shape, subgroup_size):
        self._group_shape = group_shape
        self._subgroup_size = subgroup_size
        self._subgroup_ranges = (range(group_shape[0]), range(group_shape[1]), range(_divup(group_shape[2], subgroup_size)))
        self._workitem_ranges = (range(group_shape[0]), range(group_shape[1]), range(group_shape[2]))
        self._group_id = None
        self._tasks = []
        self._current_task = 0

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
        return self._alloc_impl(shape, dtype, _get_uninit_value(dtype))

    def zeros(self, shape, dtype):
        return self._alloc_impl(shape, dtype, 0)

    def ones(self, shape, dtype):
        return self._alloc_impl(shape, dtype, 1)

    def full(self, shape, dtype, fill_value):
        return self._alloc_impl(shape, dtype, fill_value)

    def vempty(self, shape, dtype):
        return self._alloc_impl(shape, dtype, _get_uninit_value(dtype))

    def vzeros(self, shape, dtype):
        return self._alloc_impl(shape, dtype, 0)

    def vones(self, shape, dtype):
        return self._alloc_impl(shape, dtype, 1)

    def vfull(self, shape, dtype, fill_value):
        return self._alloc_impl(shape, dtype, fill_value)

    def vec(self, src, shape=None):
        if shape is None:
            shape = src.shape

        s = tuple(slice(0, s) for s in shape)
        return src.copy()[s]

    def subgroups(self, func):
        def _body_wrapper(sgid):
            func(CurrentSubGroup(self._subgroup_size, sgid))

        def _func():
            tasks = self._tasks
            assert(len(tasks) == 0)
            assert(self._current_task == 0)
            for sgid in product(*self._subgroup_ranges):
                tasks.append(greenlet(partial(_body_wrapper,sgid)))

            for t in tasks:
                t.switch()

            self._current_task = 0
            tasks.clear()


        return _func

    def workitems(self, func):
        def _body_wrapper(lid):
            gid = tuple(gi * gs + li for gi, gs, li in zip(self._group_id, self._group_shape, lid))
            func(CurrentWorkitem(lid, gid))

        def _func():
            tasks = self._tasks
            assert(len(tasks) == 0)
            assert(self._current_task == 0)
            for lid in product(*self._workitem_ranges):
                tasks.append(greenlet(partial(_body_wrapper,lid)))

            for t in tasks:
                t.switch()

            self._current_task = 0
            tasks.clear()


        return _func

    def barrier(self):
        tasks = self._tasks
        num_tasks = len(tasks)
        if num_tasks <= 1:
            return

        next_task = self._current_task + 1
        if next_task >= num_tasks:
            next_task = 0

        self._current_task = next_task
        tasks[next_task].switch()

    def _alloc_impl(self, shape, dtype, init):
        return ma.masked_array(np.full(shape, fill_value=init, dtype=dtype), mask=False)


def kernel():
    def _kernel_impl(func):
        def wrapper(group, *args, **kwargs):
            n_groups = group.get_num_groups()
            cg = CurrentGroup(group.group_shape, group.subgroup_size)
            for gid in product(*(range(g) for g in n_groups)):
                cg._group_id = gid
                func(cg, *args, **kwargs)

        return wrapper

    return _kernel_impl
