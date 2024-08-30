#include "blade/memory/base.hh"

using namespace Blade;

template<typename IT, typename OT, U64 integrationSize, U64 numberOfPolarizations, U64 numberOfElements>
__global__ void integrator(const ArrayTensor<Device::CUDA, IT> input,
                                 ArrayTensor<Device::CUDA, OT> output) {
    const U64 tid = (blockIdx.x * blockDim.x + threadIdx.x);

    if (tid < numberOfElements) {
        OT accumulator[numberOfPolarizations] = {};

        for (U64 i = 0; i < integrationSize; i++) {
            for (U64 j = 0; j < numberOfPolarizations; j++) {
                accumulator[j] += input[(tid * integrationSize * numberOfPolarizations) + (i * numberOfPolarizations) + j];
            }
        }

        for (U64 j = 0; j < numberOfPolarizations; j++) {
            output[(tid * numberOfPolarizations) + j] += accumulator[j];
        }
    }
}
