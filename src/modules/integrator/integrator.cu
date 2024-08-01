#include "blade/memory/base.hh"

using namespace Blade;

template<typename IT, typename OT>
__global__ void integrator(const ArrayTensor<Device::CUDA, IT> input,
                                 ArrayTensor<Device::CUDA, OT> output) {
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x);

    if (tid < input.size()) {
        output[tid] += input[tid];
    }
}
