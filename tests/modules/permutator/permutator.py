import sys
import numpy as np
import blade as bl

@bl.runner
class Pipeline:
    def __init__(self, in_shape, out_shape, config):
        self.input.buf = bl.array_tensor(in_shape, dtype=bl.cf32)
        self.output.buf = bl.array_tensor(out_shape, dtype=bl.cf32)

        self.module.permutator = bl.module(bl.permutator, config, self.input.buf)

    def transfer_in(self, buf):
        self.copy(self.input.buf, buf)

    def transfer_out(self, buf):
        self.copy(self.output.buf, self.module.permutator.get_output())
        self.copy(buf, self.output.buf)


def test(A, B, C, D):
    in_shape = (2, 192, 8192, 2)

    config = {
        "indexes": [A, B, C, D],
    }

    out_shape = (in_shape[A], in_shape[B], in_shape[C], in_shape[D])

    host_input = bl.array_tensor(in_shape, dtype=bl.cf32, device=bl.cpu)
    host_output = bl.array_tensor(out_shape, dtype=bl.cf32, device=bl.cpu)

    bl_input = host_input.as_numpy()
    bl_output = host_output.as_numpy()

    np.copyto(bl_input, np.random.random(size=in_shape) + 1j*np.random.random(size=in_shape))

    #
    # Blade Implementation
    #

    pipeline = Pipeline(in_shape, out_shape, config)
    pipeline(host_input, host_output)

    #
    # Python Implementation
    #

    py_output = np.transpose(bl_input, axes=config["indexes"])

    #
    # Compare Results
    #

    assert np.allclose(bl_output, py_output, rtol=0.01)
    print("Test successfully completed!")


if __name__ == "__main__":
    test(int(sys.argv[1]),
         int(sys.argv[2]),
         int(sys.argv[3]), 
         int(sys.argv[4]))