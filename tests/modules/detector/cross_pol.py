import numpy as np
import blade as bl

@bl.runner
class Pipeline:
    def __init__(self, input_shape, output_shape, config):
        self.input.buf = bl.array_tensor(input_shape, dtype=bl.cf32)
        self.output.buf = bl.array_tensor(output_shape, dtype=bl.f32)

        self.module.detector = bl.module(bl.detector, config, self.input.buf, ot=bl.f32)

    def transfer_in(self, buf):
        self.copy(self.input.buf, buf)

    def transfer_out(self, buf):
        self.copy(self.output.buf, self.module.detector.get_output())
        self.copy(buf, self.output.buf)


if __name__ == "__main__":
    number_of_beams = 2
    number_of_channels = 192
    number_of_samples = 8750
    number_of_polarizations = 2

    integration_rate = 10
    output_polarizations = 4

    input_shape = (number_of_beams, number_of_channels, number_of_samples, number_of_polarizations)
    output_shape = (number_of_beams, number_of_channels, number_of_samples // integration_rate, output_polarizations)

    config = {
        'integration_rate': integration_rate,
        'number_of_output_polarizations': output_polarizations,
    }

    host_input = bl.array_tensor(input_shape, dtype=bl.cf32, device=bl.cpu)
    host_output = bl.array_tensor(output_shape, dtype=bl.f32, device=bl.cpu)

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

    py_output = np.zeros(output_shape, dtype=np.float32)
    
    for ibeam in range(number_of_beams):
        for ichan in range(number_of_channels):
            for isamp in range(number_of_samples//integration_rate):
                x = bl_input[ibeam, ichan, isamp*integration_rate:isamp*integration_rate+integration_rate, 0]
                y = bl_input[ibeam, ichan, isamp*integration_rate:isamp*integration_rate+integration_rate, 1]

                auto_x = x*np.conj(x)
                auto_y = y*np.conj(y)
                cross  = x*np.conj(y)

                py_output[ibeam, ichan, isamp, 0] = np.sum(auto_x.real)
                py_output[ibeam, ichan, isamp, 1] = np.sum(auto_y.real)
                py_output[ibeam, ichan, isamp, 2] = np.sum(cross.real)
                py_output[ibeam, ichan, isamp, 3] = np.sum(cross.imag)

    #
    # Compare Results
    #

    assert np.allclose(bl_output, py_output, rtol=0.1)
    print("Test successfully completed!")
