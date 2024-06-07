import numpy as np
import blade as bl

@bl.runner
class Pipeline:
    def __init__(self, input_shape, output_shape, config):
        self.input.buf = bl.array_tensor(input_shape, dtype=bl.cf32)
        self.output.buf = bl.array_tensor(output_shape, dtype=bl.cf32)

        self.module.correlator = bl.module(bl.correlator, config, self.input.buf)

    def transfer_in(self, buf):
        self.copy(self.input.buf, buf)

    def transfer_out(self, buf):
        self.copy(self.output.buf, self.module.correlator.get_output())
        self.copy(buf, self.output.buf)


if __name__ == "__main__":
    # This assumes that the input data was already transferred to the frequency domain.
    number_of_antennas = 2
    number_of_channels = 4
    number_of_samples = 2
    number_of_polarizations = 2

    number_of_baselines = int((number_of_antennas * (number_of_antennas + 1)) / 2)

    input_shape = (number_of_antennas, number_of_channels, number_of_samples, number_of_polarizations)
    output_shape = (number_of_baselines, number_of_channels, 1, 4)

    config = {
        'integration_size': number_of_samples,
        'block_size': 4
    }

    host_input = bl.array_tensor(input_shape, dtype=bl.cf32, device=bl.cpu)
    host_output = bl.array_tensor(output_shape, dtype=bl.cf32, device=bl.cpu)

    bl_input = host_input.as_numpy()
    bl_output = host_output.as_numpy()

    np.copyto(bl_input, np.random.random(size=input_shape) + 1j*np.random.random(size=input_shape))

    #
    # Blade Implementation
    #

    pipeline = Pipeline(input_shape, output_shape, config)
    pipeline(host_input, host_output)

    #
    # Python Implementation
    #

    py_output = np.zeros(output_shape, dtype=np.complex64)
    
    ibline = 0
    for iant1 in range(number_of_antennas):
        for iant2 in range(iant1, number_of_antennas):
            ant1 = bl_input[iant1, ...]
            ant2 = bl_input[iant2, ...]

            py_output[ibline, :, 0, 0] = np.sum(ant1[:, :, 0] * np.conj(ant2[:, :, 0]), axis=1)
            py_output[ibline, :, 0, 1] = np.sum(ant1[:, :, 0] * np.conj(ant2[:, :, 1]), axis=1)
            py_output[ibline, :, 0, 2] = np.sum(ant1[:, :, 1] * np.conj(ant2[:, :, 0]), axis=1)
            py_output[ibline, :, 0, 3] = np.sum(ant1[:, :, 1] * np.conj(ant2[:, :, 1]), axis=1)
            
            ibline += 1

    #
    # Compare Results
    #

    assert np.allclose(bl_output, py_output, rtol=0.1)
    print("Test successfully completed!")
