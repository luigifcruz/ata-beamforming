import time
import blade as bl
import numpy as np

class Test(bl.Pipeline):
    beamformer: bl.Beamformer
    input = bl.vector.cuda.cf32()
    phasors = bl.vector.cuda.cf32()

    def __init__(self,
                 number_of_beams,
                 number_of_antennas,
                 number_of_frequency_channels,
                 number_of_time_samples,
                 number_of_polarizations):
        bl.Pipeline.__init__(self)
        _config = bl.Beamformer.Config(
            number_of_beams,
            number_of_antennas,
            number_of_frequency_channels,
            number_of_time_samples,
            number_of_polarizations,
            True, 
            True,
            512
        )
        _input = bl.Beamformer.Input(self.input, self.phasors)
        self.beamformer = self.connect(_config, _input)

    def input_size(self):
        return self.beamformer.input_size()

    def phasors_size(self):
        return self.beamformer.phasors_size()

    def output_size(self):
        return self.beamformer.output_size()

    def run(self, input: bl.vector.cpu.cf32,
                  phasors: bl.vector.cpu.cf32,
                  output: bl.vector.cpu.cf32):
        self.copy(self.beamformer.input(), input)
        self.copy(self.beamformer.phasors(), phasors)
        self.compute()
        self.copy(output, self.beamformer.output())
        self.synchronize()


if __name__ == "__main__":
    # Specify dimension of array.
    number_of_beams = 16
    number_of_antennas = 20
    number_of_frequency_channels = 192
    number_of_time_samples = 8192
    number_of_polarizations = 2

    # Initialize Blade pipeline.
    mod = Test(
        number_of_beams,
        number_of_antennas,
        number_of_frequency_channels,
        number_of_time_samples, 
        number_of_polarizations
    )

    # Generate test data with Python.
    _a = np.random.uniform(-int(2**8/2), int(2**8/2), mod.input_size())
    _b = np.random.uniform(-int(2**8/2), int(2**8/2), mod.input_size())
    _c = np.array(_a + _b * 1j).astype(np.complex64)
    input = _c.reshape((
            number_of_antennas, 
            number_of_frequency_channels, 
            number_of_time_samples,
            number_of_polarizations
        ))

    _a = np.zeros((
            number_of_beams,
            number_of_antennas,
            number_of_frequency_channels,
            number_of_polarizations
        ), dtype=np.complex64)
    phasors = np.random.random(size=_a.shape) + 1j*np.random.random(size=_a.shape)

    output = np.zeros((
            number_of_beams + 1,
            number_of_frequency_channels,
            number_of_time_samples,
            number_of_polarizations
        ), dtype=np.complex64)

    # Import test data from Python to Blade.
    bl_input = bl.vector.cpu.cf32(mod.input_size())
    bl_phasors = bl.vector.cpu.cf32(mod.phasors_size())
    bl_output = bl.vector.cpu.cf32(mod.output_size())

    np.copyto(np.array(bl_input, copy=False), input.flatten())
    np.copyto(np.array(bl_phasors, copy=False), phasors.flatten())
    np.copyto(np.array(bl_output, copy=False), output.flatten())

    # Beamform with Blade.
    start = time.time()
    mod.run(bl_input, bl_phasors, bl_output)
    print(f"Beamform with Blade took {time.time()-start:.2f} s.")

    # Beamform with Numpy.
    start = time.time()
    for ibeam in range(number_of_beams):
        phased = input * phasors[ibeam][..., np.newaxis, :]
        output[ibeam] = phased.sum(axis=0)
    phased = input * phasors[-1][..., np.newaxis, :]
    phased = (phased.real * phased.real) + (phased.imag * phased.imag)
    output[-1] = np.sqrt(phased.sum(axis=0))
    print(f"Beamform with Numpy took {time.time()-start:.2f} s.")

    # Check both answers.
    bl_out = np.array(bl_output, copy=False).reshape(output.shape)
    py_out = output

    print(bl_out[0, 0, :16, 0])
    print(py_out[0, 0, :16, 0])

    assert np.allclose(bl_out[:-1, :, :, :], py_out[:-1, :, :, :], rtol=0.01)
    assert np.allclose(bl_out[-1, :, :, :], py_out[-1, :, :, :], atol=250)
    print("Test successfully completed!")
