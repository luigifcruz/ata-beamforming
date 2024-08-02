import math
import numpy as np
import blade as bl

@bl.runner
class Pipeline:
    def __init__(self, in_shape, out_shape, config):
        self.input.buffer = bl.array_tensor(in_shape, dtype=bl.cf32)
        self.output.buffer = bl.array_tensor(out_shape, dtype=bl.cf32)

        self.module.mode_x = bl.module(bl.modex, config, self.input.buffer)

    def transfer_in(self, buffer):
        self.copy(self.input.buffer, buffer)

    def transfer_out(self, buffer):
        self.copy(self.output.buffer, self.module.mode_x.get_output())
        self.copy(buffer, self.output.buffer)


if __name__ == "__main__":
    in_shape = (4, 1, 32768, 2)
    out_shape = (10, 65536, 1, 4)

    config = {
        'input_shape': in_shape,
        'output_shape': out_shape,

        'pre_correlator_gatherer_rate': 2,

        'correlator_integration_rate': 8192
    }

    host_input_buffer = bl.array_tensor(in_shape, dtype=bl.cf32, device=bl.cpu)
    host_output_buffer = bl.array_tensor(out_shape, dtype=bl.cf32, device=bl.cpu)

    bl_input_buffer = host_input_buffer.as_numpy()
    bl_output_buffer = host_output_buffer.as_numpy()

    np.copyto(bl_input_buffer, np.random.random(size=in_shape) + 1j*np.random.random(size=in_shape))

    #
    # Blade Implementation
    #

    pipeline = Pipeline(in_shape, out_shape, config)

    while True:
        if pipeline(host_input_buffer, host_output_buffer):
            break

    #
    # Python Implementation
    #

    py_output = np.zeros(out_shape, dtype=np.complex64)

    # TODO: Implement this.
    exit(0)

    #
    # Compare Results
    #

    print("Top 10 differences:")
    diff = np.abs(bl_output_buffer - py_output_buffer)
    diff = diff.flatten()
    diff.sort()
    print(diff[-10:])
    print("")
    print("Average difference: ", np.mean(diff))
    print("Maximum difference: ", np.max(diff))
    print("Minimum difference: ", np.min(diff))

    assert np.allclose(bl_output_buffer, py_output_buffer, rtol=0.5, atol=0.5)
    print("Test successfully completed!")
