
from collections.abc import Iterable
from functools import partial
from greenlet import greenlet
from itertools import product
from numpy import ma
from sympy import Symbol
import inspect
import numpy as np
import types
import typing

from .indexing import sym

DEF_GROUP_SHAPE = (64,1,1)
DEF_SUBGROUP_SIZE = 16

def _get_uninit_value(dtype):
    return 0

def _divup(a, b):
    return (a + b - 1) // b

def _get_num_groups(work_shape, group_shape):
    return tuple(_divup(a, b) for a, b in zip(work_shape, group_shape))

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


def _get_dims(arg):
    if not isinstance(arg, Iterable):
        return (arg, 1, 1)

    assert len(arg) == 3
    return arg


def _add_sub(subs, sym, val):
    if sym in subs:
        old_val = subs[sym]
        assert old_val == val, f"Symbol value conflict for {sym}: {old_val} and {val}"
    else:
        subs[sym] = val

def _visit_arg_annotation(idx, ann, prev_handler):
    if isinstance(ann, Symbol):
        def handler(subs, args):
            val = args[idx]
            _add_sub(subs, ann, val)

    elif isinstance(ann(), tuple):
        def handler(subs, args):
            val = args[idx]
            assert isinstance(val, Iterable)
            ann_args = typing.get_args(ann)
            assert len(val) == len(ann_args)
            for s, v in zip(ann_args, val):
                if not isinstance(s, Symbol):
                    continue

                _add_sub(subs, s, v)

    else:
        assert False, f"Unsupported annotation: {ann}"

    if prev_handler:
        def chained(subs, args):
            prev_handler(subs, args)
            handler(subs, args)

        return chained
    else:
        return handler


def _visit_func_annotations(func):
    ann = inspect.get_annotations(func)
    handler = None
    for i, arg in enumerate(inspect.signature(func).parameters):
        arg_ann = ann.get(arg, None);
        if arg_ann is None:
            continue;

        handler = _visit_arg_annotation(i - 1, arg_ann, handler)

    return handler

def _handle_dim(src, subs):
    if isinstance(src, Iterable):
        return tuple(_handle_dim(v, subs) for v in src)

    if isinstance(src, Symbol):
        src = src.subs(subs)

    return int(src)

def kernel(work_shape, group_shape=DEF_GROUP_SHAPE, subgroup_size=DEF_SUBGROUP_SIZE, literals=()):
    assert isinstance(subgroup_size, int) or subgroup_size in literals, "Subgroup size must be const or literal"
    work_shape = _get_dims(work_shape)
    group_shape = _get_dims(group_shape)
    def _kernel_impl(func):
        handler = _visit_func_annotations(func)
        def wrapper(*args, **kwargs):
            if handler:
                subs_map = {}
                handler(subs_map, args)
                subs_args = list(subs_map.items())
            else:
                subs_args = []


            ws = _handle_dim(work_shape, subs_args)
            gs = _handle_dim(group_shape, subs_args)
            ss = _handle_dim(subgroup_size, subs_args)

            n_groups = _get_num_groups(ws, gs)
            cg = CurrentGroup(gs, ss)
            for gid in product(*(range(g) for g in n_groups)):
                cg._group_id = gid
                func(cg, *args, **kwargs)

        return wrapper

    return _kernel_impl
