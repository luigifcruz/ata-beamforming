#ifndef BLADE_MEMORY_CUDA_VECTOR_HH
#define BLADE_MEMORY_CUDA_VECTOR_HH

#include <memory>

#include "blade/memory/types.hh"
#include "blade/memory/vector.hh"

namespace Blade {

template<typename T>
class BLADE_API Vector<Device::CUDA, T> : public VectorImpl<T> {
 public:
    using VectorImpl<T>::VectorImpl;

    explicit Vector(const U64& size) {
        BL_CHECK_THROW(this->resize(size));
    }

    ~Vector() {
        if (this->container.empty() || !this->managed) {
            return;
        }

        if (cudaFree(this->container.data()) != cudaSuccess) {
            BL_FATAL("Failed to deallocate CUDA memory.");
        }
    }

    // TODO: Implement resize.
    Result resize(const U64& size) override {
        if (!this->container.empty() && !this->managed) {
            return Result::ERROR;
        }

        T* ptr;
        auto size_bytes = size * sizeof(T);

        BL_CUDA_CHECK(cudaMalloc(&ptr, size_bytes), [&]{
            BL_FATAL("Failed to allocate CUDA memory: {}", err);
        });

        this->container = std::span<T>(ptr, size);
        this->managed = true;

        return Result::SUCCESS;
    }
};

template<typename T>
class BLADE_API Vector<Device::CUDA | Device::CPU, T> : public VectorImpl<T> {
 public:
    using VectorImpl<T>::VectorImpl;

    explicit Vector(const U64& size) {
        BL_CHECK_THROW(this->resize(size));
    }

    ~Vector() {
        if (this->container.empty() || !this->managed) {
            return;
        }

        if (cudaFree(this->container.data()) != cudaSuccess) {
            BL_FATAL("Failed to deallocate CUDA memory.");
        }
    }

    // TODO: Implement resize.
    Result resize(const U64& size) override {
        if (!this->container.empty() && !this->managed) {
            return Result::ERROR;
        }

        T* ptr;
        auto size_bytes = size * sizeof(T);

        BL_CUDA_CHECK(cudaMallocManaged(&ptr, size_bytes), [&]{
            BL_FATAL("Failed to allocate CUDA memory: {}", err);
        });

        this->container = std::span<T>(ptr, size);
        this->managed = true;

        this->cpuVector.release();
        this->cudaVector.release();

        return Result::SUCCESS;
    }

    operator Vector<Device::CPU, T>&() {
        if (!this->cpuVector) {
            this->cpuVector = std::make_unique
                    <Vector<Device::CPU, T>>(this->container);
        }

        return *this->cpuVector;
    }

    operator Vector<Device::CUDA, T>&() {
        if (!this->cudaVector) {
            this->cudaVector = std::make_unique
                    <Vector<Device::CUDA, T>>(this->container);
        }

        return *this->cudaVector;
    }

 protected:
    std::unique_ptr<Vector<Device::CPU, T>> cpuVector;
    std::unique_ptr<Vector<Device::CUDA, T>> cudaVector;
};

}  // namespace Blade

#endif
