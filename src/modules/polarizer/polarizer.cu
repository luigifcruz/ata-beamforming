#include "blade/memory/base.hh"

using namespace Blade;

template<typename IT, typename OT>
__global__ void polarizer_xy_lr(const ArrayTensor<Device::CUDA, IT> input,
                                ArrayTensor<Device::CUDA, OT> output) {
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    assert(input.size() == output.size());

    if (tid < input.size()) {
        // The complex multiplication below can be simplified because
        // the real part of the phasor is 0.0. Boring implementation:
        // const IT yPol90 = cuCmulf(yPol, make_cuFloatComplex(0.0, 1.0));

        const IT xPol = input[tid + 0];
        const IT yPol = input[tid + 1];

        const IT yPol90(-yPol.imag(), +yPol.real());
        
        output[tid + 0] = xPol + yPol90;
        output[tid + 1] = xPol - yPol90;
    }
}

template<typename IT, typename OT>
__global__ void polarizer_xy_x(const ArrayTensor<Device::CUDA, IT> input,
                               ArrayTensor<Device::CUDA, OT> output) {
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x);

    if (tid < output.size()) {
        output[tid] = input[(tid * 2) + 0];
    }
}

template<typename IT, typename OT>
__global__ void polarizer_xy_y(const ArrayTensor<Device::CUDA, IT> input,
                               ArrayTensor<Device::CUDA, OT> output) {
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x);

    if (tid < output.size()) {
        output[tid] = input[(tid * 2) + 1];
    }
}
