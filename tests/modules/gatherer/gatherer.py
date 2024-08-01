import sys
import numpy as np
import blade as bl

@bl.runner
class Pipeline:
    def __init__(self, in_shape, out_shape, config):
        self.input.buf = bl.array_tensor(in_shape, dtype=bl.cf32)
        self.output.buf = bl.array_tensor(out_shape, dtype=bl.cf32)

        self.module.gatherer = bl.module(bl.gatherer, config, self.input.buf)

    def transfer_in(self, buf):
        self.copy(self.input.buf, buf)

    def transfer_out(self, buf):
        self.copy(self.output.buf, self.module.gatherer.get_output())
        self.copy(buf, self.output.buf)


def test(A, F, T, P, Axis, Multiplier):
    in_shape = (A, F, T, P)

    config = {
        "axis": Axis,
        "multiplier": Multiplier,
    }

    out_shape = list(in_shape)
    out_shape[config["axis"]] *= config["multiplier"]
    out_shape = tuple(out_shape)

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
    np.concatenate([bl_input for _ in range(config["multiplier"])], axis=config["axis"], out=py_output)

    #
    # Compare Results
    #

    assert np.allclose(bl_output, py_output, rtol=0.01)
    print("Test successfully completed!")


if __name__ == "__main__":
    test(int(sys.argv[1]),
         int(sys.argv[2]),
         int(sys.argv[3]), 
         int(sys.argv[4]), 
         int(sys.argv[5]),
         int(sys.argv[6]))