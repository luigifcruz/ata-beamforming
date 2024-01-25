#ifndef BLADE_MEMORY_VECTOR_HH
#define BLADE_MEMORY_VECTOR_HH

#include <cuda_runtime.h>

#include "blade/macros.hh"
#include "blade/memory/types.hh"
#include "blade/memory/shape.hh"
#ifndef __CUDA_ARCH__
#include "blade/memory/profiler.hh"
#endif

namespace Blade {

template<Device DeviceId,
         typename DataType,
         typename Shape>
struct Vector {
 public:
    Vector()
             : _shape(),
               _data(nullptr),
               _refs(nullptr),
               _unified(false) {
        BL_TRACE("Empty vector created.");
    }

    explicit Vector(void* ptr, const typename Shape::Type& shape, const bool& unified = false)
             : _shape(shape), 
               _data(static_cast<DataType*>(ptr)),
               _refs(nullptr),
               _unified(unified) {
    }

    explicit Vector(const typename Shape::Type& shape, const bool& unified = false)
             : _shape(shape),
               _data(nullptr),
               _refs(nullptr),
               _unified(unified) {
        BL_TRACE("Vector allocated and created: {}", shape);

        _refs = new U64;
        *_refs = 1;

#ifndef __CUDA_ARCH__
        if (Profiler::IsCapturing()) {
            if (_unified) {
                Profiler::RegisterUnifiedAllocation(size_bytes());
            } else if (DeviceId == Device::CPU) {
                Profiler::RegisterCpuAllocation(size_bytes());
            } else if (DeviceId == Device::CUDA) {
                Profiler::RegisterCudaAllocation(size_bytes());
            }
            return;
        }
#endif

        if constexpr (DeviceId == Device::CPU) {
            BL_CUDA_CHECK_THROW(cudaMallocHost(&_data, size_bytes()), [&]{
                BL_FATAL("Failed to allocate pinned host memory: {}", err);
            });
        }

        if constexpr (DeviceId == Device::CUDA) {
            if (unified) {
                BL_CUDA_CHECK_THROW(cudaMallocManaged(&_data, size_bytes()), [&]{
                    BL_FATAL("Failed to allocate managed CUDA memory: {}", err);
                });
            } else {
                BL_CUDA_CHECK_THROW(cudaMalloc(&_data, size_bytes()), [&]{
                    BL_FATAL("Failed to allocate CUDA memory: {}", err);
                });
            }
        }
    }

    __host__ __device__ Vector(const Vector& other)
             : _shape(other._shape),
               _data(other._data),
               _refs(other._refs),
               _unified(other._unified) {
        BL_TRACE("Vector created by copy.");

        increaseRefCount();
    }

    __host__ __device__ Vector(Vector&& other)
             : _shape(),
               _data(nullptr),
               _refs(nullptr),
               _unified(false) { 
        BL_TRACE("Vector created by move.");

        std::swap(_data, other._data);
        std::swap(_refs, other._refs);
        std::swap(_shape, other._shape);
        std::swap(_unified, other._unified);
    }

    __host__ __device__ Vector& operator=(const Vector& other) {
        BL_TRACE("Vector copied to existing.");

        decreaseRefCount();
        _data = other._data;
        _refs = other._refs;
        _shape = other._shape;
        _unified = other._unified;
        increaseRefCount();

        return *this;
    }

    __host__ __device__ Vector& operator=(Vector&& other) {
        BL_TRACE("Vector moved to existing.");

        decreaseRefCount();
        reset();
        std::swap(_data, other._data);
        std::swap(_refs, other._refs);
        std::swap(_shape, other._shape);
        std::swap(_unified, other._unified);

        return *this;
    }

    __host__ __device__ ~Vector() {
        decreaseRefCount();
    }

    __host__ __device__ constexpr DataType* data() const noexcept {
        return _data;
    }

    __host__ __device__ constexpr U64 refs() const noexcept {
        if (!_refs) {
            return 0;
        }
        return *_refs;
    }

    __host__ __device__ constexpr const char* type() const noexcept {
        return TypeInfo<DataType>::name;
    }

    __host__ __device__ constexpr const char* device() const noexcept {
        return DeviceInfo<DeviceId>::name;
    }

    __host__ __device__ constexpr U64 hash() const noexcept {
        return std::hash<void*>{}(_data);
    }

    __host__ __device__ constexpr U64 size() const noexcept {
        return _shape.size();
    }

    __host__ __device__ constexpr U64 size_bytes() const noexcept {
        return size() * sizeof(DataType);
    }

    __host__ __device__ constexpr DataType& operator[](const typename Shape::Type& shape) {
        return _data[_shape.shapeToOffset(shape)];
    }

    __host__ __device__ constexpr const DataType& operator[](const typename Shape::Type& shape) const {
        return _data[_shape.shapeToOffset(shape)];
    }

    __host__ __device__ constexpr DataType& operator[](const U64& idx) {
        return _data[idx];
    }

    __host__ __device__ constexpr const DataType& operator[](const U64& idx) const {
        return _data[idx];
    }

    __host__ __device__ [[nodiscard]] constexpr bool empty() const noexcept {
        return (_data == nullptr);
    }

    __host__ __device__ constexpr auto begin() {
        return _data;
    }

    __host__ __device__ constexpr auto end() {
        return _data + size();
    }

    __host__ __device__ constexpr auto begin() const {
        return _data;
    }

    __host__ __device__ constexpr auto end() const {
        return _data + size();
    }

    __host__ __device__ constexpr const Shape& shape() const {
        return _shape;
    }

    __host__ __device__ constexpr bool unified() const {
            return _unified;
    }

    __host__ __device__ Result reshape(const Shape& shape) {
        if (shape.size() != _shape.size()) {
            return Result::ERROR;
        }
        _shape = shape;

        return Result::SUCCESS;
    }

 private:
    Shape _shape;
    DataType* _data;
    U64* _refs;
    bool _unified;

    __host__ __device__ void decreaseRefCount() {
        if (!_refs) {
            return;
        }

        BL_TRACE("Decreasing reference counter ({}).", *_refs);

        if (--(*_refs) == 0) {
            BL_TRACE("Deleting vector.");

            delete _refs;

#ifndef __CUDA_ARCH__
            if (Profiler::IsCapturing()) {
                if (_unified) {
                    Profiler::RegisterUnifiedDeallocation(size_bytes());
                } else if (DeviceId == Device::CPU) {
                    Profiler::RegisterCpuDeallocation(size_bytes());
                } else if (DeviceId == Device::CUDA) {
                    Profiler::RegisterCudaDeallocation(size_bytes());
                }

                reset();
                return;
            }
#endif

            if constexpr (DeviceId == Device::CPU) {
                if (cudaFreeHost(_data) != cudaSuccess) {
                    BL_FATAL("Failed to deallocate host memory.");
                }
            }

            if constexpr (DeviceId == Device::CUDA) {
                if (cudaFree(_data) != cudaSuccess) {
                    BL_FATAL("Failed to deallocate CUDA memory.");
                }
            }

            reset();
        }
    }

    __host__ __device__ void increaseRefCount() {
        if (!_refs) {
            return;
        }

        BL_TRACE("Increasing reference counter ({}).", *_refs);
        *_refs += 1;
    }

    __host__ __device__ void reset() {
        _data = nullptr;
        _refs = nullptr;
        _shape = Shape();
        _unified = false;
    }
};

}  // namespace Blade

#endif
