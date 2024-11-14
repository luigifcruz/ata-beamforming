import math
import numpy as np
import blade as bl

@bl.runner
class Pipeline:
    def __init__(self, in_shape, out_shape, config):
        self.input.buffer = bl.array_tensor(in_shape, dtype=bl.cf32)
        self.output.buffer = bl.array_tensor(out_shape, dtype=bl.f32)

        self.module.mode_h = bl.module(bl.modeh, config, self.input.buffer, ot=bl.f32)

    def transfer_in(self, buffer):
        self.copy(self.input.buffer, buffer)

    def transfer_out(self, buffer):
        self.copy(self.output.buffer, self.module.mode_h.get_output())
        self.copy(buffer, self.output.buffer)


if __name__ == "__main__":
    in_shape = (2, 192, 8192, 2)
    int_shape = (2, 192*8192, 1, 2)
    out_shape = (2, 192*8192, 1, 1)

    config = {
        'input_shape': in_shape,
        'output_shape': out_shape,

        'polarizer_convert_to_circular': True,

        'detector_integration_rate': 1,
        'detector_number_of_output_polarizations': 1,
    }

    host_input_buffer = bl.array_tensor(in_shape, dtype=bl.cf32, device=bl.cpu)
    host_output_buffer = bl.array_tensor(out_shape, dtype=bl.f32, device=bl.cpu)

    bl_input_buffer = host_input_buffer.as_numpy()
    bl_output_buffer = host_output_buffer.as_numpy()

    np.copyto(bl_input_buffer, np.random.random(size=in_shape) + 1j*np.random.random(size=in_shape))

    #
    # Blade Implementation
    #

    pipeline = Pipeline(in_shape, out_shape, config)

    pipeline(host_input_buffer, host_output_buffer)

    #
    # Python Implementation
    #

    py_output_buffer = np.zeros(out_shape, dtype=np.float32)

    fft_result = np.fft.fftshift(np.fft.fft(bl_input_buffer, axis=2), axes=2)
    fft_result = fft_result.reshape(int_shape)

    polarized = np.zeros(int_shape, dtype=np.complex64)
    polarized[..., 0] = fft_result[..., 0] + 1j * fft_result[..., 1]
    polarized[..., 1] = fft_result[..., 0] - 1j * fft_result[..., 1]

    for ibeam in range(int_shape[0]):
        for ichan in range(int_shape[1]):
            for isamp in range(int_shape[2]):
                x = polarized[ibeam, ichan, isamp, 0]
                y = polarized[ibeam, ichan, isamp, 1]

                auto_x = x.real * x.real + x.imag * x.imag
                auto_y = y.real * y.real + y.imag * y.imag

                py_output_buffer[ibeam, ichan, isamp, 0] = np.sum(auto_x) + np.sum(auto_y)

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