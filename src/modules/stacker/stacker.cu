#include "blade/memory/base.hh"

using namespace Blade;

// TODO: Several improvements can be made to this kernel.
template<typename T>
__global__ void stacker(const ArrayTensor<Device::CUDA, T> input,
                              ArrayTensor<Device::CUDA, T> output,
                        const U64 axis,
                        const U64 offset) {
    const U64 tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < input.size()) {
        const auto inputShape = input.shape().offsetToShape(tid);
        const U64 oid = output.shape().shapeToOffset(inputShape, axis, offset);
        output[oid] = input[tid];
    }
}