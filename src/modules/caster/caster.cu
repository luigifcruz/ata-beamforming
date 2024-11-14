#include <cuComplex.h>
#include <cuda_fp16.h>
#include <stdint.h>

template<typename IT, typename OT, uint64_t N>
__global__ void caster(const IT* input, OT* output) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N){
        output[tid] = static_cast<OT>(input[tid]);
    }
}
