import time
import blade as bl
import numpy as np

class Test(bl.Pipeline):
    beamformer: bl.Beamformer
    input = bl.vector.cuda.cf32()
    phasors = bl.vector.cuda.cf32()

    def __init__(self, dims: bl.ArrayDims):
        bl.Pipeline.__init__(self)
        _config = bl.Beamformer.Config(dims, 512)
        _input = bl.Beamformer.Input(self.input, self.phasors)
        self.beamformer = self.connect(_config, _input)

    def inputSize(self):
        return self.beamformer.inputSize()

    def phasorSize(self):
        return self.beamformer.phasorSize()

    def outputSize(self):
        return self.beamformer.outputSize()

    def run(self, input: bl.vector.cpu.cf32,
                  phasor: bl.vector.cpu.cf32,
                  output: bl.vector.cpu.cf32):
        self.copy(self.beamformer.input(), input)
        self.copy(self.beamformer.phasor(), phasor)
        self.compute()
        self.copy(output, self.beamformer.output())
        self.synchronize()


if __name__ == "__main__":
    # Specify dimension of array.
    d = bl.ArrayDims(NBEAMS=16, NANTS=20, NCHANS=192, NTIME=8192, NPOLS=2)

    # Initialize Blade pipeline.
    mod = Test(d)

    # Generate test data with Python.
    _a = np.random.uniform(-int(2**16/2), int(2**16/2), mod.inputSize())
    _b = np.random.uniform(-int(2**16/2), int(2**16/2), mod.inputSize())
    _c = np.array(_a + _b * 1j).astype(np.complex64)
    input = _c.reshape((d.NANTS, d.NCHANS, d.NTIME, d.NPOLS))

    _a = np.zeros((d.NBEAMS, d.NANTS, d.NCHANS, d.NPOLS), dtype=np.complex64)
    phasors = np.random.random(size=_a.shape) + 1j*np.random.random(size=_a.shape)

    output = np.zeros((d.NBEAMS, d.NCHANS, d.NTIME, d.NPOLS), dtype=np.complex64)

    # Import test data from Python to Blade.
    bl_input = bl.vector.cpu.cf32(mod.inputSize())
    bl_phasors = bl.vector.cpu.cf32(mod.phasorSize())
    bl_output = bl.vector.cpu.cf32(mod.outputSize())

    np.copyto(np.array(bl_input, copy=False), input.flatten())
    np.copyto(np.array(bl_phasors, copy=False), phasors.flatten())
    np.copyto(np.array(bl_output, copy=False), output.flatten())

    # Beamform with Blade.
    start = time.time()
    mod.run(bl_input, bl_phasors, bl_output)
    print(f"Beamform with Blade took {time.time()-start:.2f} s.")

    # Beamform with Numpy.
    start = time.time()
    for ibeam in range(d.NBEAMS):
        phased = input * phasors[ibeam][..., np.newaxis, :]
        output[ibeam] = phased.sum(axis=0)
    print(f"Beamform with Numpy took {time.time()-start:.2f} s.")

    # Check both answers.
    assert np.allclose(np.array(bl_output, copy=False), output.flatten(), rtol=0.01)
    print("Test successfully completed!")
