import ast
import inspect
from newlang.kernel import (
    kernel,
    sym,
    CurrentGroup,
    CurrentSubGroup,
    CurrentWorkitem,
    Buffer,
)

G1 = sym.G1
G2 = sym.G2


@kernel(work_shape=(G1, G2, 1))
def foo(gr: CurrentGroup, arr: Buffer[G1, G2]):
    @gr.workitems
    def inner(wi: CurrentWorkitem):
        pass
        # gid = wi.global_id()[:2]
        # if gid[0] < arr.shape[0] and gid[1] < arr.shape[1]:
        #    arr[gid] = gid[0]*arr.shape[1] + gid[1]

    inner()


print(inspect.getsource(foo.orig_func))
root = ast.parse(inspect.getsource(foo.orig_func))
print(ast.dump(root, indent=1))
