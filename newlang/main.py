import newlang.kernel as kernel

print("ASASDASDAS")

@kernel.kernel
def foo(gr):
    print(gr.group_id())

g = kernel.Group((512,1,1))
foo(g)
