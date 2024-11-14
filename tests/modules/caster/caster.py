import numpy as np
import blade as bl

@bl.runner
class Pipeline:
    def __init__(self, shape, config):
        self.input.buf = bl.array_tensor(shape, dtype=bl.i8)
        self.output.buf = bl.array_tensor(shape, dtype=bl.f32)

        self.module.caster = bl.module(bl.caster, config, self.input.buf, it=bl.i8, ot=bl.f32)

    def transfer_in(self, buf):
        self.copy(self.input.buf, buf)

    def transfer_out(self, buf):
        self.copy(self.output.buf, self.module.caster.get_output())
        self.copy(buf, self.output.buf)


if __name__ == "__main__":
    shape = (2, 192, 8750, 2)

    config = {}

    host_input = bl.array_tensor(shape, dtype=bl.i8, device=bl.cpu)
    host_output = bl.array_tensor(shape, dtype=bl.f32, device=bl.cpu)

    bl_input = host_input.as_numpy()
    bl_output = host_output.as_numpy()

    np.copyto(bl_input, np.random.uniform(-int(2**8/2), int(2**8/2), shape).astype(np.int8))

    #
    # Blade Implementation
    #

    pipeline = Pipeline(shape, config)
    pipeline(host_input, host_output)

    #
    # Python Implementation
    #

    py_output = bl_input.astype(np.float32)

    #
    # Compare Results
    #

    assert np.allclose(bl_output, py_output, rtol=0.1)
    print("Test successfully completed!")
