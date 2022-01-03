#ifndef BLADE_MEMORY_CUDA_COPY_HH
#define BLADE_MEMORY_CUDA_COPY_HH

#include "blade/memory/types.hh"
#include "blade/memory/vector.hh"

namespace Blade::Memory {

template<typename T>
static Result Copy(VectorImpl<T>& dst,
                   const VectorImpl<T>& src,
                   const cudaMemcpyKind& kind,
                   const cudaStream_t& stream = 0) {
    if (dst.size() != src.size()) {
        BL_FATAL("Size mismatch between source and destination ({}, {}).",
                src.size(), dst.size());
        return Result::ASSERTION_ERROR;
    }

    BL_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.data(), src.size_bytes(),
                kind, stream), [&]{
        BL_FATAL("Can't copy data: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

template<typename T>
static Result Copy(Vector<Device::CUDA, T>& dst,
                   const Vector<Device::CUDA, T>& src,
                   const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyDeviceToDevice, stream);
}

template<typename T>
static Result Copy(Vector<Device::CUDA, T>& dst,
                   const Vector<Device::CPU, T>& src,
                   const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyHostToDevice, stream);
}

template<typename T>
static Result Copy(Vector<Device::CPU, T>& dst,
                   const Vector<Device::CUDA, T>& src,
                   const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyDeviceToHost, stream);
}

}  // namespace Blade::Memory

#endif
