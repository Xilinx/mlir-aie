from aie.ir import Context, Location, Module, InsertionPoint


# Create and print ModuleOp.
def construct_and_print_module(f):
    print("\nTEST:", f.__name__)
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        assert module.operation.verify()
        print(module)
