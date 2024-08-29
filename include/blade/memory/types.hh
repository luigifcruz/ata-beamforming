#ifndef BLADE_MEMORY_TYPES_HH
#define BLADE_MEMORY_TYPES_HH

#include <cstdint>
#include <cstddef>
#include <complex>
#include <cstdlib>

#include <cuda.h>
#include <cuda_fp16.h>

#include "blade/logger.hh"

#ifdef __CUDA_ARCH__
#include "blade/memory/device/ops.hh"
#endif

#ifndef BL_MEM_ALIGN
#define BL_MEM_ALIGN(n) alignas(n)
#endif

#ifndef BL_PHYSICAL_CONSTANT_C
#define BL_PHYSICAL_CONSTANT_C (double)299792458.0  // Speed of Light (m/s)
#endif

#ifndef BL_PHYSICAL_CONSTANT_PI
#define BL_PHYSICAL_CONSTANT_PI M_PI
#endif

#ifndef BL_DEG_TO_RAD
#define BL_DEG_TO_RAD(DEG) (DEG * M_PI / 180.0)
#endif

#ifndef BL_RAD_TO_DEG
#define BL_RAD_TO_DEG(RAD) (RAD * 180.0 / M_PI)
#endif

namespace Blade {

struct Event {
    void* handle = nullptr;

    operator CUevent_st*() const {
        return reinterpret_cast<CUevent_st*>(handle);
    }

    operator CUevent_st**() {
        return reinterpret_cast<CUevent_st**>(&handle);
    }
};

struct Stream {
    void* handle = nullptr;

    operator CUstream_st*() const {
        return reinterpret_cast<CUstream_st*>(handle);
    }

    operator CUstream_st**() {
        return reinterpret_cast<CUstream_st**>(&handle);
    }
};

typedef __half   F16;
typedef float    F32;
typedef double   F64;
typedef int8_t   I8;
typedef int16_t  I16;
typedef int32_t  I32;
typedef int64_t  I64;
typedef uint8_t  U8;
typedef uint16_t U16;
typedef uint32_t U32;
typedef uint64_t U64;
typedef bool     BOOL;

#ifndef __CUDA_ARCH__

typedef std::complex<F16> CF16;
typedef std::complex<F32> CF32;
typedef std::complex<F64> CF64;
typedef std::complex<I8>  CI8;
typedef std::complex<I16> CI16;
typedef std::complex<I32> CI32;
typedef std::complex<I64> CI64;
typedef std::complex<U8>  CU8;
typedef std::complex<U16> CU16;
typedef std::complex<U32> CU32;
typedef std::complex<U64> CU64;

#define BLADE_API __attribute__((visibility("default")))
#define BLADE_HIDDEN __attribute__((visibility("hidden")))

#else

typedef ops::complex<F16> CF16;
typedef ops::complex<F32> CF32;
typedef ops::complex<F64> CF64;
typedef ops::complex<I8>  CI8;
typedef ops::complex<I16> CI16;
typedef ops::complex<I32> CI32;
typedef ops::complex<I64> CI64;
typedef ops::complex<U8>  CU8;
typedef ops::complex<U16> CU16;
typedef ops::complex<U32> CU32;
typedef ops::complex<U64> CU64;

#define BLADE_API
#define BLADE_HIDDEN

#endif

enum class BLADE_API Device : uint8_t {
    NONE    = 0 << 0,
    CPU     = 1 << 0,
    CUDA    = 1 << 1,
};

template <Device D = Device::NONE>
struct BLADE_API DeviceInfo;

template<>
struct BLADE_API DeviceInfo<Device::CUDA> {
    inline static const char* name = "CUDA";
};

template<>
struct BLADE_API DeviceInfo<Device::CPU> {
    inline static const char* name = "CPU";
};

template <typename T = void>
struct BLADE_API TypeInfo;

template<>
struct BLADE_API TypeInfo<F16> {
    using type = F16;
    using subtype = F16;
    using surtype = CF16;
    inline static bool is_complex = false;
    inline static const char* name = "F16";
    inline static const U64 cudaSize = 1;           // TODO: Remove all cudaSize after port to Ops is complete.
    inline static const char* cudaName = "__half";  // TODO: Remove all cudaName after port to Ops is complete.
};

template<>
struct BLADE_API TypeInfo<F32> {
    using type = F32;
    using subtype = F32;
    using surtype = CF32;
    inline static bool is_complex = false;
    inline static const char* name = "F32";
    inline static const U64 cudaSize = 1;
    inline static const char* cudaName = "float";
};

template<>
struct BLADE_API TypeInfo<F64> {
    using type = F64;
    using subtype = F64;
    using surtype = CF64;
    inline static bool is_complex = false;
    inline static const char* name = "F64";
    inline static const U64 cudaSize = 1;
    inline static const char* cudaName = "double";
};

template<>
struct BLADE_API TypeInfo<I8> {
    using type = I8;
    using subtype = I8;
    using surtype = CI8;
    inline static bool is_complex = false;
    inline static const char* name = "I8";
    inline static const U64 cudaSize = 1;
    inline static const char* cudaName = "signed char";
};

template<>
struct BLADE_API TypeInfo<I16> {
    using type = I16;
    using subtype = I16;
    using surtype = CI16;
    inline static bool is_complex = false;
    inline static const char* name = "I16";
    inline static const U64 cudaSize = 1;
    inline static const char* cudaName = "short";
};

template<>
struct BLADE_API TypeInfo<I32> {
    using type = I32;
    using subtype = I32;
    using surtype = CI32;
    inline static bool is_complex = false;
    inline static const char* name = "I32";
    inline static const U64 cudaSize = 1;
    inline static const char* cudaName = "long";
};

template<>
struct BLADE_API TypeInfo<I64> {
    using type = I64;
    using subtype = I64;
    using surtype = CI64;
    inline static bool is_complex = false;
    inline static const char* name = "I64";
    inline static const U64 cudaSize = 1;
    inline static const char* cudaName = "long long";
};

template<>
struct BLADE_API TypeInfo<U8> {
    using type = U8;
    using subtype = U8;
    using surtype = CU8;
    inline static bool is_complex = false;
    inline static const char* name = "U8";
    inline static const U64 cudaSize = 1;
    inline static const char* cudaName = "unsigned char";
};

template<>
struct BLADE_API TypeInfo<U16> {
    using type = U16;
    using subtype = U16;
    using surtype = CU16;
    inline static bool is_complex = false;
    inline static const char* name = "U16";
    inline static const U64 cudaSize = 1;
    inline static const char* cudaName = "unsigned short";
};

template<>
struct BLADE_API TypeInfo<U32> {
    using type = U32;
    using subtype = U32;
    using surtype = CU32;
    inline static bool is_complex = false;
    inline static const char* name = "U32";
    inline static const U64 cudaSize = 1;
    inline static const char* cudaName = "unsigned long";
};

template<>
struct BLADE_API TypeInfo<U64> {
    using type = U64;
    using subtype = U64;
    using surtype = CU64;
    inline static bool is_complex = false;
    inline static const char* name = "U64";
    inline static const U64 cudaSize = 1;
    inline static const char* cudaName = "unsigned long long";
};

template<>
struct BLADE_API TypeInfo<BOOL> {
    using type = BOOL;
    using subtype = BOOL;
    using surtype = BOOL;
    inline static bool is_complex = false;
    inline static const char* name = "BOOL";
    inline static const U64 cudaSize = 1;
    inline static const char* cudaName = "bool";
};

template<>
struct BLADE_API TypeInfo<CF16> {
    using type = CF16;
    using subtype = F16;
    using surtype = F16;
    inline static bool is_complex = true;
    inline static const char* name = "CF16";
    inline static const U64 cudaSize = 2;
    inline static const char* cudaName = "ops::complex<F16>";
};

template<>
struct BLADE_API TypeInfo<CF32> {
    using type = CF32;
    using subtype = F32;
    using surtype = F32;
    inline static bool is_complex = true;
    inline static const char* name = "CF32";
    inline static const U64 cudaSize = 2;
    inline static const char* cudaName = "ops::complex<F32>";
};

template<>
struct BLADE_API TypeInfo<CF64> {
    using type = CF64;
    using subtype = F64;
    using surtype = F64;
    inline static bool is_complex = true;
    inline static const char* name = "CF64";
    inline static const U64 cudaSize = 2;
    inline static const char* cudaName = "ops::complex<F64>";
};

template<>
struct BLADE_API TypeInfo<CI8> {
    using type = CI8;
    using subtype = I8;
    using surtype = I8;
    inline static bool is_complex = true;
    inline static const char* name = "CI8";
    inline static const U64 cudaSize = 2;
    inline static const char* cudaName = "NonSupported";
};

template<>
struct BLADE_API TypeInfo<CI16> {
    using type = CI16;
    using subtype = I16;
    using surtype = I16;
    inline static bool is_complex = true;
    inline static const char* name = "CI16";
    inline static const U64 cudaSize = 2;
    inline static const char* cudaName = "NonSupported";
};

template<>
struct BLADE_API TypeInfo<CI32> {
    using type = CI32;
    using subtype = I32;
    using surtype = I32;
    inline static bool is_complex = true;
    inline static const char* name = "CI32";
    inline static const U64 cudaSize = 2;
    inline static const char* cudaName = "NonSupported";
};

template<>
struct BLADE_API TypeInfo<CI64> {
    using type = CI64;
    using subtype = I64;
    using surtype = I64;
    inline static bool is_complex = true;
    inline static const char* name = "CI64";
    inline static const U64 cudaSize = 2;
    inline static const char* cudaName = "NonSupported";
};

template<>
struct BLADE_API TypeInfo<CU8> {
    using type = CU8;
    using subtype = U8;
    using surtype = U8;
    inline static bool is_complex = true;
    inline static const char* name = "CU8";
    inline static const U64 cudaSize = 2;
    inline static const char* cudaName = "NonSupported";
};

template<>
struct BLADE_API TypeInfo<CU16> {
    using type = CU16;
    using subtype = U16;
    using surtype = U16;
    inline static bool is_complex = true;
    inline static const char* name = "CU16";
    inline static const U64 cudaSize = 2;
    inline static const char* cudaName = "NonSupported";
};

template<>
struct BLADE_API TypeInfo<CU32> {
    using type = CU32;
    using subtype = U32;
    using surtype = U32;
    inline static bool is_complex = true;
    inline static const char* name = "CU32";
    inline static const U64 cudaSize = 2;
    inline static const char* cudaName = "NonSupported";
};

template<>
struct BLADE_API TypeInfo<CU64> {
    using type = CU64;
    using subtype = U64;
    using surtype = U64;
    inline static bool is_complex = true;
    inline static const char* name = "CU64";
    inline static const U64 cudaSize = 2;
    inline static const char* cudaName = "NonSupported";
};

}  // namespace Blade

#endif
