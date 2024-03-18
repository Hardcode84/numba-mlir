from collections.abc import Iterable
from functools import partial
from greenlet import greenlet
from itertools import product
from numpy import ma
from sympy import Symbol
from sympy.core.expr import Expr
from math import prod
import inspect
import numpy as np
import types
import typing
import functools
import operator

from .indexing import sym

DEF_GROUP_SHAPE = 64
DEF_SUBGROUP_SIZE = 16


class TunableParam:
    def __init__(self, sym, default, vals, strategy=None):
        self.sym = sym
        self.default = default
        self.vals = vals
        self.strategy = strategy


def _get_default_tunables(tunables):
    if isinstance(tunables, TunableParam):
        return {tunables.sym: tunables.default}

    if isinstance(tunables, Iterable):
        return {t.sym: t.default for t in tunables}

    assert False, "Unsupported tunables"


def resolve_symbols(func, symbols):
    old_closure = func.__closure__
    new_closure = None

    if old_closure is not None:
        cell_cls = type(old_closure[0])

        def resolve_cell(cell):
            if isinstance(cell.cell_contents, Symbol):
                if cell.cell_contents in symbols.keys():
                    return cell_cls(symbols[cell.cell_contents])
            elif isinstance(cell.cell_contents, Expr):
                return cell_cls(cell.cell_contents.subs(symbols))

            return cell

        new_closure = tuple([resolve_cell(cell) for cell in old_closure])

    def resolve_global(val):
        if isinstance(val, Symbol):
            if val in symbols.keys():
                return symbols[val]
        elif isinstance(val, Expr):
            return val.subs(symbols)

        return val

    new_globals = {key: resolve_global(val) for key, val in func.__globals__.items()}

    g = types.FunctionType(
        func.__code__,
        new_globals,
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=new_closure,
    )
    g = functools.update_wrapper(g, func)
    g.__kwdefaults__ = func.__kwdefaults__
    return g


class _kernel:
    def __init__(
        self, func, work_shape, group_shape, subgroup_size, literals, tunables, tuned={}
    ):
        self.handler = _visit_func_annotations(func)
        self.orig_func = func
        self.work_shape = work_shape
        self.group_shape = group_shape
        self.subgroup_size = subgroup_size
        self.literals = literals
        self.tunables = tunables
        self.tuned = tuned
        self.default_tunbles = _get_default_tunables(tunables)

    def __call__(self, *args, **kwargs):
        if self.handler:
            subs_map = {}
            self.handler(subs_map, args)
            subs_map = subs_map | self.default_tunbles | self.tuned
            subs_args = list(subs_map.items())
        else:
            subs_args = []

        ws = _handle_dim(self.work_shape, subs_args)
        gs = _handle_dim(self.group_shape, subs_args)
        ss = _handle_dim(self.subgroup_size, subs_args)

        resolved_func = resolve_symbols(self.orig_func, subs_map)

        n_groups = _get_num_groups(ws, gs)
        cg = CurrentGroup(gs, ss)
        for gid in product(*(range(g) for g in n_groups)):
            cg._group_id = gid
            resolved_func(cg, *args, **kwargs)

    def parametrize(self, tuned):
        return _kernel(
            self.orig_func,
            self.work_shape,
            self.group_shape,
            self.subgroup_size,
            self.literals,
            self.tunables,
            self.tuned | tuned,
        )


def _not(val):
    if isinstance(val, (bool, int)):
        return val == 0

    if isinstance(val, np.ndarray):
        return ~val

    if isinstance(val, (tuple, list)):
        return type(val)([_not(v) for v in val])

    assert False


class _masked_array:
    def __init__(self, data, mask=True):
        if isinstance(data, ma.masked_array):
            assert mask
            self.__ma = data
        else:
            self.__ma = ma.masked_array(data, mask=_not(mask))

    def __set_mask(self, key, value):
        self.__ma.mask[key] = _not(value)

    @property
    def mask(self):
        return _not(self.__ma.mask)

    @property
    def data(self):
        return self.__ma.data

    def __setitem__(self, key, value):
        if isinstance(value, _masked_array):
            self.data[key] = np.where(value.mask, value.data, self.data[key])
            self.__set_mask(key, self.mask | value.mask)
        else:
            self.data[key] = value
            self.__set_mask(key, True)

    def __getitem__(self, key):
        return _masked_array(self.__ma[key])

    def _bin_op_implementation(self, other, op):
        if isinstance(other, _masked_array):
            return _masked_array(op(self.__ma, other.__ma))
        else:
            return _masked_array(op(self.__ma, other))

    def _bin_iop_implementation(self, other, op):
        self[:] = op(self, other)

        return self

    def _un_op_implemenation(self, op):
        return _masked_array(op(self.__ma))

    def __getattr__(self, name):
        return getattr(self.__ma, name)

    def __repr__(self):
        return f"_masked_array(data={self.__ma.data},\n mask={self.mask})"


def set_arithm_methods():
    bin_op_list = [
        operator.add,
        operator.sub,
        operator.mul,
        operator.matmul,
        operator.truediv,
        operator.floordiv,
        operator.pow,
        operator.mod,
        operator.lt,
        operator.le,
        operator.eq,
        operator.ne,
        operator.ge,
        operator.gt,
        operator.and_,
        operator.or_,
        operator.xor,
        operator.rshift,
        operator.lshift,
    ]

    un_op_list = [operator.abs, operator.neg, operator.not_, operator.inv]

    def _name(op):
        return op.__name__.replace("_", "")

    def set_bin_attr(op):
        setattr(
            _masked_array,
            f"__{_name(op)}__",
            lambda self, other: self._bin_op_implementation(other, op),
        )
        setattr(
            _masked_array,
            f"__i{_name(op)}__",
            lambda self, other: self._bin_iop_implementation(other, op),
        )

    def set_un_attr(op):
        setattr(
            _masked_array,
            f"__{_name(op)}__",
            lambda self: self._un_op_implemenation(op),
        )

    for op in bin_op_list:
        set_bin_attr(op)

    for op in un_op_list:
        set_un_attr(op)


set_arithm_methods()


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
        self._dims = len(group_shape)
        self._group_shape = group_shape
        self._subgroup_size = subgroup_size
        self._subgroup_ranges = self.__get_subgroup_ranges()
        self._workitem_ranges = self.__get_workitems_ranges()
        self._group_id = None
        self._tasks = []
        self._current_task = 0

    def __get_subgroup_ranges(self):
        rngs = []
        if len(self._group_shape) > 1:
            rngs += [range(v) for v in self._group_shape[0:-1]]

        rngs += [range(_divup(self._group_shape[-1], self._subgroup_size))]

        return tuple(rngs)

    def __get_workitems_ranges(self):
        return tuple([range(v) for v in self._group_shape])

    @property
    def id(self):
        return self._group_id

    @property
    def shape(self):
        return self._group_shape

    @property
    def size(self):
        if isinstance(self._group_shape, int):
            return self._group_shape

        return prod(self._group_shape)

    @property
    def work_offset(self):
        return tuple(a * b for a, b in zip(self._group_shape, self._group_id))

    def load(self, array, shape):
        if not isinstance(shape, Iterable):
            shape = (shape,)

        dtype = array.dtype
        init = _get_uninit_value(dtype)
        res = _masked_array(np.full(shape, fill_value=init, dtype=dtype), mask=False)
        s = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(array.shape, shape))

        res[s] = array[s]
        return res

    def store(self, dst, src):
        s = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(src.shape, dst.shape))
        if isinstance(src, _masked_array):
            dst[s] = np.where(src.mask[s], src[s], dst[s])
        else:
            dst[s] = src[s]

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
            assert len(tasks) == 0
            assert self._current_task == 0
            for sgid in product(*self._subgroup_ranges):
                tasks.append(greenlet(partial(_body_wrapper, sgid)))

            for t in tasks:
                t.switch()

            self._current_task = 0
            tasks.clear()

        return _func

    def workitems(self, func):
        def _body_wrapper(lid):
            gid = tuple(
                gi * gs + li
                for gi, gs, li in zip(self._group_id, self._group_shape, lid)
            )
            func(CurrentWorkitem(lid, gid))

        def _func():
            tasks = self._tasks
            assert len(tasks) == 0
            assert self._current_task == 0
            for lid in product(*self._workitem_ranges):
                tasks.append(greenlet(partial(_body_wrapper, lid)))

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
        return _masked_array(np.full(shape, fill_value=init, dtype=dtype), mask=True)


class Buffer(typing.Generic[typing.ParamSpec("Args")]):
    pass


def _get_dims(arg):
    if not isinstance(arg, Iterable):
        return (arg,)

    return arg


def _add_sub(subs, sym, val):
    if sym in subs:
        old_val = subs[sym]
        assert old_val == val, f"Symbol value conflict for {sym}: {old_val} and {val}"
    else:
        subs[sym] = val


def _visit_arg_annotation(idx, ann, prev_handler):
    def istypingtype(a, typ):
        return typing.get_origin(ann) == typ or isinstance(ann, typ)

    def get_typing_args(ann):
        if isinstance(ann, (types.GenericAlias, typing._GenericAlias)):
            return typing.get_args(ann)

        if isinstance(ann, Iterable):
            return ann

        assert False

    if ann in (CurrentGroup, CurrentSubGroup, CurrentWorkitem):
        # nothing
        return

    if isinstance(ann, Symbol):

        def handler(subs, args):
            val = args[idx]
            _add_sub(subs, ann, val)

    elif istypingtype(ann, tuple):

        def handler(subs, args):
            val = args[idx]
            assert isinstance(val, Iterable)
            ann_args = get_typing_args(ann)
            assert len(val) == len(ann_args)
            for s, v in zip(ann_args, val):
                if not isinstance(s, Symbol):
                    continue

                _add_sub(subs, s, v)

    elif istypingtype(ann, Buffer):

        def handler(subs, args):
            val = args[idx]
            ann_args = typing.get_args(ann)[0]
            assert len(val.shape) == len(ann_args)
            for s, v in zip(ann_args, val.shape):
                if not isinstance(s, Symbol):
                    continue

                _add_sub(subs, s, v)

    else:
        assert False, f"Unsupported annotation: {type(ann)} {ann}"

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
        arg_ann = ann.get(arg, None)
        if arg_ann is None:
            continue

        handler = _visit_arg_annotation(i - 1, arg_ann, handler)

    return handler


def _handle_dim(src, subs):
    if isinstance(src, Iterable):
        return tuple(_handle_dim(v, subs) for v in src)

    if isinstance(src, Expr):
        src = src.subs(subs)

    return int(src)


def kernel(
    work_shape,
    group_shape=None,
    subgroup_size=None,
    literals=(),
    tunables=(),
):
    if subgroup_size is None:
        subgroup_size = DEF_SUBGROUP_SIZE

    assert (
        isinstance(subgroup_size, int) or subgroup_size in literals
    ), "Subgroup size must be const or literal"
    work_shape = _get_dims(work_shape)

    if group_shape is None:
        group_shape = (DEF_GROUP_SHAPE,)
        while len(group_shape) < len(work_shape):
            group_shape = (1,) + group_shape

    group_shape = _get_dims(group_shape)

    assert len(work_shape) <= 3
    assert len(work_shape) == len(group_shape)

    def _kernel_impl(func):
        return _kernel(func, work_shape, group_shape, subgroup_size, literals, tunables)

    return _kernel_impl


def atomic_ref(a):
    return a
