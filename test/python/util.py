import inspect
from aie.ir import Context, Location, Module, InsertionPoint


# Run test
def construct_test(f):
    print("\nTEST:", f.__name__)
    f()


# Create and print ModuleOp.
def construct_and_print_module(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            args = inspect.getfullargspec(f).args
            if args:
                if args == ["module"]:
                    module = f(module)
                else:
                    raise Exception(f"only `module` arg supported {args=}")
            else:
                f()
        if module is not None:
            assert module.operation.verify()
            print(module)
