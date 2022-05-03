import time
from random import random
import blade as bl
import numpy as np

class Test(bl.Pipeline):
    detector: bl.Detector
    input = bl.vector.cuda.cf32()

    def __init__(self, detector_config: bl.Detector.Config):
        bl.Pipeline.__init__(self)
        _config = detector_config
        _input = bl.Detector.Input(self.input)
        self.detector = self.connect(_config, _input)

    def input_size(self):
        return self.detector.input_size()

    def output_size(self):
        return self.detector.output_size()

    def run(self, input: bl.vector.cpu.cf32,
                  output: bl.vector.cpu.f32):
        self.copy(self.detector.input(), input)
        self.compute()
        self.copy(output, self.detector.output())
        self.synchronize()


if __name__ == "__main__":
    NTIME = 8750
    TFACT = 10
    NCHANS = 192
    OUTPOLS = 1
    NBEAMS = 2
    NPOLS = 2

    #
    # Blade Implementation
    #

    detector_config = bl.Detector.Config(
        number_of_beams = NBEAMS,
        number_of_frequency_channels = NCHANS,
        number_of_time_samples = NTIME,
        number_of_polarizations = NPOLS,

        integration_size = TFACT,
        number_of_output_polarizations = OUTPOLS,
        
        block_size = 512
    )

    mod = Test(detector_config)

    bl_input_raw = bl.vector.cpu.cf32(mod.input_size())
    bl_output_raw = bl.vector.cpu.f32(mod.output_size())

    bl_input = np.array(bl_input_raw, copy=False).reshape((NBEAMS, NCHANS, NTIME, NPOLS))
    bl_output = np.array(bl_output_raw, copy=False).reshape((NBEAMS, NCHANS, NTIME//TFACT, OUTPOLS))

    np.copyto(bl_input, np.random.random(size=bl_input.shape) + 1j*np.random.random(size=bl_input.shape))

    start = time.time()
    mod.run(bl_input_raw, bl_output_raw)
    print(f"Detection with Blade took {time.time()-start:.2f} s.")

    #
    # Python Implementation
    #

    py_output = np.zeros((NBEAMS, NCHANS, NTIME//TFACT, OUTPOLS), dtype=np.float32)
    
    start = time.time()
    for ibeam in range(NBEAMS):
        for ichan in range(NCHANS):
            for isamp in range(NTIME//TFACT):
                x = bl_input[ibeam, ichan, isamp*TFACT:isamp*TFACT+TFACT, 0] #just to make code more visible
                y = bl_input[ibeam, ichan, isamp*TFACT:isamp*TFACT+TFACT, 1] #just to make code more visible

                auto_x = x*np.conj(x) # or: x.real*x.real + x.imag*x.imag... this will definitely be a real value, no .imag part
                auto_y = y*np.conj(y) # or: y.real*y.real + y.imag*y.imag... this will definitely be a real value, no .imag part

                py_output[ibeam, ichan, isamp, 0] = np.sum(auto_x.real) + np.sum(auto_y.real)
    print(f"Detection with Python took {time.time()-start:.2f} s.")

    #
    # Compare Results
    #

    assert np.allclose(bl_output, py_output, rtol=0.1)
    print("Test successfully completed!")
