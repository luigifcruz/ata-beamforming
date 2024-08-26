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
    number_of_antennas = 28
    number_of_channels = 2**16
    number_of_samples = 1
    number_of_polarizations = 2
    integration_rate = 8

    number_of_baselines = int((number_of_antennas * (number_of_antennas + 1)) / 2)

    input_shape = (number_of_antennas, number_of_channels, number_of_samples, number_of_polarizations)
    output_shape = (number_of_baselines, number_of_channels, number_of_samples, 4)

    config = {
        'integration_rate': integration_rate,
        'block_size': 32
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
    while True:
        if pipeline(host_input, host_output):
            break

    #
    # Python Implementation
    #

    py_output = np.zeros(output_shape, dtype=np.complex64)
    
    ibline = 0
    for _ in range(integration_rate):
        for iant1 in range(number_of_antennas):
            for iant2 in range(iant1, number_of_antennas):
                ant1 = bl_input[iant1, ...]
                ant2 = bl_input[iant2, ...]

                py_output[ibline, :, :, 0] += ant1[:, :, 0] * np.conj(ant2[:, :, 0])
                py_output[ibline, :, :, 1] += ant1[:, :, 0] * np.conj(ant2[:, :, 1])
                py_output[ibline, :, :, 2] += ant1[:, :, 1] * np.conj(ant2[:, :, 0])
                py_output[ibline, :, :, 3] += ant1[:, :, 1] * np.conj(ant2[:, :, 1])
                
                ibline += 1
        ibline = 0

    #
    # Compare Results
    #

    assert np.allclose(bl_output, py_output, rtol=0.1)
    print("Test successfully completed!")
