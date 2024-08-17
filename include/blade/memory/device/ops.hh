#ifndef BLADE_MEMORY_DEVICE_OPS_HH
#define BLADE_MEMORY_DEVICE_OPS_HH

#include <cstdint>

#include <cuda_fp16.h>

#if !defined(__CUDA_ARCH__) && !defined(BL_OPS_HOST_SIDE_KEY)
// This is not meant to be a fully featured complex library.
// It's meant to be a replacement for cuComplex.h that supports
// half-precision operations. It ignores multiple std::complex
// standards for the sake of computational efficiency.
#error "This header should only be included in device code."
#endif

// Don't forget to add tests and benchmarks for every Op.

namespace Blade::ops {

template<typename T>
class alignas(2 * sizeof(T)) complex {
 public:
    __host__ __device__ complex() : _real(0), _imag(0) {}
    __host__ __device__ complex(T r) : _real(r), _imag(0) {}
    __host__ __device__ complex(T r, T i) : _real(r), _imag(i) {}

    // CF32 -> CF64
    __host__ __device__ explicit complex(const complex<float>& rhs) : _real(static_cast<T>(rhs.real())), _imag(static_cast<T>(rhs.imag())) {}

    // CF64 -> CF32
    __host__ __device__ operator complex<float>() const {
        return complex<float>(static_cast<float>(_real), static_cast<float>(_imag));
    }

    __host__ __device__ complex<T> operator+(const complex<T>& rhs) const {
        return complex<T>(_real + rhs._real, _imag + rhs._imag);
    }

    __host__ __device__ complex<T> operator-(const complex<T>& rhs) const {
        return complex<T>(_real - rhs._real, _imag - rhs._imag);
    }

    __host__ __device__ complex<T> operator*(const complex<T>& rhs) const {
        return complex<T>(_real * rhs._real - _imag * rhs._imag,
                          _real * rhs._imag + _imag * rhs._real);
    }

    __host__ __device__ complex<T> operator/(const complex<T>& rhs) const {
        T denom = rhs._real * rhs._real + rhs._imag * rhs._imag;
        T real = (_real * rhs._real + _imag * rhs._imag) / denom;
        T imag = (_imag * rhs._real - _real * rhs._imag) / denom;
        return complex<T>(real, imag);
    }

    __host__ __device__ complex<T>& operator+=(const complex<T>& rhs) {
        _real += rhs._real;
        _imag += rhs._imag;
        return *this;
    }

    __host__ __device__ complex<T>& operator-=(const complex<T>& rhs) {
        _real -= rhs._real;
        _imag -= rhs._imag;
        return *this;
    }

    __host__ __device__ complex<T>& operator*=(const complex<T>& rhs) {
        T real = _real * rhs._real - _imag * rhs._imag;
        T imag = _real * rhs._imag + _imag * rhs._real;
        _real = real;
        _imag = imag;
        return *this;
    }

    __host__ __device__ complex<T>& operator/=(const complex<T>& rhs) {
        T denom = rhs._real * rhs._real + rhs._imag * rhs._imag;
        T real = (_real * rhs._real + _imag * rhs._imag) / denom;
        T imag = (_imag * rhs._real - _real * rhs._imag) / denom;
        _real = real;
        _imag = imag;
        return *this;
    }

    __host__ __device__ bool operator==(const complex<T>& rhs) const {
        if constexpr (std::is_same<T, __half>::value) {
            return __heq(_real, rhs._real) && __heq(_imag, rhs._imag);
        } else {
            return _real == rhs._real && _imag == rhs._imag;
        }
    }

    __host__ __device__ bool operator!=(const complex<T>& rhs) const {
        if constexpr (std::is_same<T, __half>::value) {
            return __hne(_real, rhs._real) || __hne(_imag, rhs._imag);
        } else {
            return _real != rhs._real || _imag != rhs._imag;
        }
    }

    __host__ __device__ bool operator<(const complex<T>& rhs) const {
        if constexpr (std::is_same<T, __half>::value) {
            return __hlt(_real, rhs._real) || (__heq(_real, rhs._real) && __hlt(_imag, rhs._imag));
        } else {
            return (_real < rhs._real) || ((_real == rhs._real) && (_imag < rhs._imag));
        }
    }

    __host__ __device__ bool operator>(const complex<T>& rhs) const {
        if constexpr (std::is_same<T, __half>::value) {
            return __hgt(_real > rhs._real) || (__heq(_real, rhs._real) && __hgt(_imag, rhs._imag));
        } else {
            return (_real > rhs._real) || ((_real == rhs._real) && (_imag > rhs._imag));
        }
    }

    __host__ __device__ constexpr T real() const {
        return _real;
    }

    __host__ __device__ constexpr T imag() const {
        return _imag;
    }

    __host__ __device__ constexpr complex<T> conj() const {
        return complex<T>(_real, -_imag);
    }

    __host__ __device__ void atomic_add(const complex<T>& rhs) {
        atomicAdd(&_real, rhs._real);
        atomicAdd(&_imag, rhs._imag);
    }

    __host__ __device__ void atomic_sub(const complex<T>& rhs) {
        atomicSub(&_real, rhs._real);
        atomicSub(&_imag, rhs._imag);
    }

 private:
    T _real;
    T _imag;
};

}  // namespace Blade::ops

#endif
