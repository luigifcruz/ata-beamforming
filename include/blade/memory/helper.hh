#ifndef BLADE_MEMORY_HELPER_HH
#define BLADE_MEMORY_HELPER_HH

#include <cuda_runtime.h>

#include "blade/memory/types.hh"
#include "blade/memory/vector.hh"

namespace Blade::Memory {

template<typename Type, typename Dims>
static inline Result PageLock(const Vector<Device::CPU, Type, Dims>& vec,
                              const bool& readOnly = false) {
    cudaPointerAttributes attr;
    BL_CUDA_CHECK(cudaPointerGetAttributes(&attr, vec.data()), [&]{
        BL_FATAL("Failed to get pointer attributes: {}", err);
    });

    if (attr.type != cudaMemoryTypeUnregistered) {
        BL_WARN("Memory already registered.");
        return Result::SUCCESS;
    }

    unsigned int kind = cudaHostRegisterDefault;
    if (readOnly) {
        kind = cudaHostRegisterReadOnly;
    }

    BL_CUDA_CHECK(cudaHostRegister(vec.data(), vec.size_bytes(), kind), [&]{
        BL_FATAL("Failed to register CPU memory: {}", err);
    });

    return Result::SUCCESS;
}

template<Device DeviceId, typename Type, typename Shape>
static inline Result Link(Vector<DeviceId, Type, Shape>& dst,
                          const Vector<DeviceId, Type, Shape>& src) {
    dst = src;
    return Result::SUCCESS;
}

template<Device DeviceId, typename Type, typename Shape>
static inline Result Link(Vector<DeviceId, Type, Shape>& dst,
                          const Vector<DeviceId, Type, Shape>& src,
                          const Shape dstShape) {
    dst = src;
    return dst.reshape(dstShape);
    return Result::SUCCESS;
}

}  // namespace Blade::Memory

#endif
