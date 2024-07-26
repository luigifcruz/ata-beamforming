import sys
import numpy as np
import blade as bl

@bl.runner
class Pipeline:
    def __init__(self, in_shape, out_shape, config):
        self.input.buf = bl.array_tensor(in_shape, dtype=bl.cf32)
        self.output.buf = bl.array_tensor(out_shape, dtype=bl.cf32)

        self.module.integrate = bl.module(bl.integrate, config, self.input.buf)

    def transfer_in(self, buf):
        self.copy(self.input.buf, buf)

    def transfer_out(self, buf):
        self.copy(self.output.buf, self.module.integrate.get_output())
        self.copy(buf, self.output.buf)


def test(A, F, T, P, Size):
    in_shape = (A, F, T, P)
    out_shape = (A, F, T, P)

    config = {
        "size": Size,
    }

    host_input = bl.array_tensor(in_shape, dtype=bl.cf32, device=bl.cpu)
    host_output = bl.array_tensor(out_shape, dtype=bl.cf32, device=bl.cpu)

    bl_input = host_input.as_numpy()
    bl_output = host_output.as_numpy()

    np.copyto(bl_input, np.random.random(size=in_shape) + 1j*np.random.random(size=in_shape))

    #
    # Blade Implementation
    #

    pipeline = Pipeline(in_shape, out_shape, config)
    while True:
        if pipeline(host_input, host_output):
            break

    #
    # Python Implementation
    #

    py_output = np.zeros(out_shape, dtype=np.complex64)
    for _ in range(Size):
        py_output += bl_input

    #
    # Compare Results
    #

    assert np.allclose(bl_output, py_output)
    print("Test successfully completed!")


if __name__ == "__main__":
    test(int(sys.argv[1]),
         int(sys.argv[2]),
         int(sys.argv[3]),
         int(sys.argv[4]),
         int(sys.argv[5]))
