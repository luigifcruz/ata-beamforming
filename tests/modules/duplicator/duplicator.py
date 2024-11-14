import numpy as np
import blade as bl

@bl.runner
class Pipeline:
    def __init__(self, shape, config):
        self.input.buf = bl.array_tensor(shape, dtype=bl.cf32)
        self.output.buf = bl.array_tensor(shape, dtype=bl.cf32)

        self.module.duplicator = bl.module(bl.duplicator, config, self.input.buf)

    def transfer_in(self, buf):
        self.copy(self.input.buf, buf)

    def transfer_out(self, buf):
        self.copy(self.output.buf, self.module.duplicator.get_output())
        self.copy(buf, self.output.buf)


if __name__ == "__main__":
    shape = (2, 192, 8750, 2)

    config = {}

    host_input = bl.array_tensor(shape, dtype=bl.cf32, device=bl.cpu)
    host_output = bl.array_tensor(shape, dtype=bl.cf32, device=bl.cpu)

    bl_input = host_input.as_numpy()
    bl_output = host_output.as_numpy()

    np.copyto(bl_input, np.random.random(size=shape) + 1j*np.random.random(size=shape))

    #
    # Blade Implementation
    #

    pipeline = Pipeline(shape, config)
    pipeline(host_input, host_output)

    #
    # Python Implementation
    #

    py_output = np.zeros(shape, dtype=np.complex64)
    np.copyto(py_output, bl_input)

    #
    # Compare Results
    #

    assert np.allclose(bl_output, py_output, rtol=0.01)
    print("Test successfully completed!")
