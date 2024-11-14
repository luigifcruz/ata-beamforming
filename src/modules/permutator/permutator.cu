#include "blade/memory/base.hh"

using namespace Blade;

// TODO: Several improvements can be made to this kernel.
template<typename T>
__global__ void permutator(const ArrayTensor<Device::CUDA, T> input,
                                 ArrayTensor<Device::CUDA, T> output,
                                 ArrayShape permutatorIndex) {
    const U64 tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < input.size()) {
        ArrayShape::Type permutedCoords = {};
        ArrayShape::Type originalCoords = input.shape().offsetToShape(tid);
        for (U64 dim = 0; dim < permutatorIndex.dimensions(); dim++) {
            permutedCoords[dim] = originalCoords[permutatorIndex[dim]];
        }
        output[permutedCoords] = input[tid];
    }
}