#include "blade/memory/base.hh"

using namespace Blade;

template<typename IT, typename OT, uint64_t N, uint64_t INT_SIZE>
__global__ void correlator(const ArrayTensor<Device::CUDA, IT> input, 
                                 ArrayTensor<Device::CUDA, OT> output) {
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) {
        return;
    }
    
    // TODO: Add correlation block.
}