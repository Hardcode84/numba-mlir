
import pytest
import numpy as np
from itertools import product
from numpy.testing import assert_equal

from newlang.kernel import kernel, Group

@pytest.mark.parametrize("gsize", [(512,1,1),(511,1,1),(1,16,1),(1,1,16),(1,1,1)])
@pytest.mark.parametrize("lsize", [(64,1,1),(1,1,1)])
def test_group_iteration(gsize, lsize):
    def get_range(i):
        return range((gsize[i] + lsize[i] - 1) // lsize[i])

    def get_group_ranges():
        return product(get_range(0),get_range(1),get_range(2))

    res_ids = []
    res_offsets = []

    @kernel
    def test(gr):
        res_ids.append(gr.group_id())
        res_offsets.append(gr.work_offset())

    test(Group(gsize, lsize))
    exp_res_ids = [(i, j, k) for i, j, k in get_group_ranges()]
    assert(res_ids == exp_res_ids)
    exp_res_offsets = [(i * lsize[0], j * lsize[1], k * lsize[2]) for i, j, k in get_group_ranges()]
    assert_equal(res_offsets, exp_res_offsets)


def test_group_load_small():
    gsize = (8,1,1)
    lsize = (8,1,1)

    res = []

    @kernel
    def test(gr, arr):
        a = gr.load(arr[gr.work_offset()[0]:], shape=gr.group_shape()[0])
        res.append(a.compressed())

    src = np.arange(12)
    test(Group(gsize, lsize), src)
    expected = [np.array([0,1,2,3,4,5,6,7])]
    assert_equal(res, expected)

def test_group_load():
    gsize = (16,1,1)
    lsize = (8,1,1)

    res = []

    @kernel
    def test(gr, arr):
        a = gr.load(arr[gr.work_offset()[0]:], shape=gr.group_shape()[0])
        res.append(a.compressed())

    src = np.arange(12)
    test(Group(gsize, lsize), src)
    expected = [np.array([0,1,2,3,4,5,6,7]),np.array([8,9,10,11])]
    assert_equal(res, expected)


def test_group_store():
    gsize = (16,1,1)
    lsize = (8,1,1)

    @kernel
    def test(gr, arr1, arr2):
        gid = gr.work_offset()
        a = gr.load(arr1[gid[0]:], shape=gr.group_shape()[0])
        gr.store(arr2[gid[0]:], a)

    src = np.arange(12)
    dst = np.full(16, fill_value=-1)
    test(Group(gsize, lsize), src, dst)
    assert_equal(dst, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1, -1])


def test_group_empty():
    gsize = (8,1,1)
    lsize = (8,1,1)

    @kernel
    def test(gr):
        a = gr.empty((3,7), dtype=np.int32)
        assert_equal(a.shape, (3,7))

    test(Group(gsize, lsize))


def test_group_zeros():
    gsize = (8,1,1)
    lsize = (8,1,1)

    @kernel
    def test(gr):
        a = gr.zeros((2,3), dtype=np.int32)
        assert_equal(a, [[0, 0, 0], [0, 0, 0]])

    test(Group(gsize, lsize))


def test_group_ones():
    gsize = (8,1,1)
    lsize = (8,1,1)

    @kernel
    def test(gr):
        a = gr.ones((2,3), dtype=np.int32)
        assert_equal(a, [[1, 1, 1], [1, 1, 1]])

    test(Group(gsize, lsize))

def test_group_full():
    gsize = (8,1,1)
    lsize = (8,1,1)

    @kernel
    def test(gr):
        a = gr.full((2,3), dtype=np.int32, fill_value=42)
        assert_equal(a, [[42, 42, 42], [42, 42, 42]])

    test(Group(gsize, lsize))


def test_group_vempty():
    gsize = (8,1,1)
    lsize = (8,1,1)

    @kernel
    def test(gr):
        a = gr.vempty((3,7), dtype=np.int32)
        assert_equal(a.shape, (3,7))

    test(Group(gsize, lsize))


def test_group_vzeros():
    gsize = (8,1,1)
    lsize = (8,1,1)

    @kernel
    def test(gr):
        a = gr.vzeros((2,3), dtype=np.int32)
        assert_equal(a, [[0, 0, 0], [0, 0, 0]])

    test(Group(gsize, lsize))


def test_group_vones():
    gsize = (8,1,1)
    lsize = (8,1,1)

    @kernel
    def test(gr):
        a = gr.vones((2,3), dtype=np.int32)
        assert_equal(a, [[1, 1, 1], [1, 1, 1]])

    test(Group(gsize, lsize))

def test_group_vfull():
    gsize = (8,1,1)
    lsize = (8,1,1)

    @kernel
    def test(gr):
        a = gr.vfull((2,3), dtype=np.int32, fill_value=42)
        assert_equal(a, [[42, 42, 42], [42, 42, 42]])

    test(Group(gsize, lsize))

def test_group_vec1():
    gsize = (8,1,1)
    lsize = (8,1,1)

    @kernel
    def test(gr):
        a = gr.full((2,3), dtype=np.int32, fill_value=42)
        v = gr.vec(a)
        assert_equal(v, [[42, 42, 42], [42, 42, 42]])

    test(Group(gsize, lsize))

def test_group_vec2():
    gsize = (8,1,1)
    lsize = (8,1,1)

    @kernel
    def test(gr):
        a = gr.full((2,3), dtype=np.int32, fill_value=42)
        v = gr.vec(a, shape=(1,2))
        assert_equal(v, [[42, 42]])

    test(Group(gsize, lsize))

def test_subgroup_iteration():
    gsize = (1,1,16)
    lsize = (1,1,8)
    sgsize = 4

    res_ids = []
    res_sizes = []

    @kernel
    def test(gr):
        @gr.subgroups
        def inner(sg):
            res_ids.append(sg.subgroup_id())
            res_sizes.append(sg.size())

        inner()

    test(Group(gsize, lsize, sgsize))
    assert_equal(res_ids, [(0, 0, 0), (0, 0, 1), (0, 0, 0), (0, 0, 1)])
    assert_equal(res_sizes, [4, 4, 4, 4])

def test_subgroup_barrier():
    gsize = (1,1,16)
    lsize = (1,1,8)
    sgsize = 4

    res = []

    @kernel
    def test(gr):
        @gr.subgroups
        def inner(sg):
            res.append((1, sg.subgroup_id()))
            gr.barrier()
            res.append((2, sg.subgroup_id()))

        inner()

    test(Group(gsize, lsize, sgsize))
    assert_equal(res, [(1, (0, 0, 0)), (1, (0, 0, 1)), (2, (0, 0, 0)), (2, (0, 0, 1)), (1, (0, 0, 0)), (1, (0, 0, 1)), (2, (0, 0, 0)), (2, (0, 0, 1))])


def test_workitems_iteration():
    gsize = (1,1,4)
    lsize = (1,1,2)

    res_gids = []
    res_lids = []

    @kernel
    def test(gr):
        @gr.workitems
        def inner(wi):
            res_lids.append(wi.local_id())
            res_gids.append(wi.global_id())

        inner()

    test(Group(gsize, lsize))
    assert_equal(res_lids, [(0, 0, 0), (0, 0, 1), (0, 0, 0), (0, 0, 1)])
    assert_equal(res_gids, [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)])


def test_workitem_barrier():
    gsize = (1,1,4)
    lsize = (1,1,2)

    res = []

    @kernel
    def test(gr):
        @gr.workitems
        def inner(wi):
            res.append((1, wi.global_id()))
            gr.barrier()
            res.append((2, wi.global_id()))

        inner()

    test(Group(gsize, lsize))
    assert_equal(res, [(1, (0, 0, 0)), (1, (0, 0, 1)), (2, (0, 0, 0)), (2, (0, 0, 1)), (1, (0, 0, 2)), (1, (0, 0, 3)), (2, (0, 0, 2)), (2, (0, 0, 3))])
