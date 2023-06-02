#ifndef BLADE_MEMORY_CUSTOM_HH
#define BLADE_MEMORY_CUSTOM_HH

#include "blade/logger.hh"
#include "blade/memory/vector.hh"
#include "blade/memory/shape.hh"

namespace Blade {

struct ArrayShape : public Shape<4> {
 public:
    inline static const char* name = "ArrayShape";

    using Shape::Shape;

    constexpr const U64& numberOfAspects() const {
        return (*this)[0];
    }

    constexpr const U64& numberOfFrequencyChannels() const {
        return (*this)[1];
    }

    constexpr const U64& numberOfTimeSamples() const {
        return (*this)[2];
    }

    constexpr const U64& numberOfPolarizations() const {
        return (*this)[3];
    }

    ArrayShape operator*(const ArrayShape& other) const {
        return ArrayShape(Shape::operator*(other));
    }

    ArrayShape operator/(const ArrayShape& other) const {
        return ArrayShape(Shape::operator/(other));
    }

 private:
#ifndef __CUDA_ARCH__
    friend std::ostream& operator<<(std::ostream& os, const ArrayShape& shape) {
        return os << fmt::format("[A: {}, F: {}, T: {}, P: {}]", 
                                   shape.numberOfAspects(),
                                   shape.numberOfFrequencyChannels(),
                                   shape.numberOfTimeSamples(),
                                   shape.numberOfPolarizations()); 
    }
#endif
};

template<Device DeviceId, typename DataType>
using ArrayTensor = Vector<DeviceId, DataType, ArrayShape>;

struct PhasorShape : public Shape<5> {
 public:
    inline static const char* name = "PhasorShape";

    using Shape::Shape;

    constexpr const U64& numberOfBeams() const {
        return (*this)[0];
    }

    constexpr const U64& numberOfAntennas() const {
        return (*this)[1];
    }

    constexpr const U64& numberOfFrequencyChannels() const {
        return (*this)[2];
    }

    constexpr const U64& numberOfTimeSamples() const {
        return (*this)[3];
    }

    constexpr const U64& numberOfPolarizations() const {
        return (*this)[4];
    }

    PhasorShape operator*(const PhasorShape& other) const {
        return PhasorShape(Shape::operator*(other));
    }

    PhasorShape operator/(const PhasorShape& other) const {
        return PhasorShape(Shape::operator/(other));
    }

 private:
#ifndef __CUDA_ARCH__
    friend std::ostream& operator<<(std::ostream& os, const PhasorShape& shape) {
        return os << fmt::format("[B: {}, A: {}, F: {}, T: {}, P: {}]", 
                                   shape.numberOfBeams(), 
                                   shape.numberOfAntennas(),
                                   shape.numberOfFrequencyChannels(),
                                   shape.numberOfTimeSamples(),
                                   shape.numberOfPolarizations()); 
    }
#endif
};

template<Device DeviceId, typename DataType>
using PhasorTensor = Vector<DeviceId, DataType, PhasorShape>;

struct DelayShape : public Shape<2> {
 public:
    inline static const char* name = "DelayShape";

    using Shape::Shape;

    constexpr const U64& numberOfBeams() const {
        return (*this)[0];
    }

    constexpr const U64& numberOfAntennas() const {
        return (*this)[1];
    }

    DelayShape operator*(const DelayShape& other) const {
        return DelayShape(Shape::operator*(other));
    }

    DelayShape operator/(const DelayShape& other) const {
        return DelayShape(Shape::operator/(other));
    }

 private:
#ifndef __CUDA_ARCH__
    friend std::ostream& operator<<(std::ostream& os, const DelayShape& shape) {
        return os << fmt::format("[B: {}, A: {}]", 
                                   shape.numberOfBeams(), 
                                   shape.numberOfAntennas()); 
    }
#endif
};

template<Device DeviceId, typename DataType>
using DelayTensor = Vector<DeviceId, DataType, DelayShape>;

struct VectorShape : public Shape<1> {
 public:
    inline static const char* name = "VectorShape";

    using Shape::Shape;

    VectorShape operator*(const VectorShape& other) const {
        return VectorShape(Shape::operator*(other));
    }

    VectorShape operator/(const VectorShape& other) const {
        return VectorShape(Shape::operator/(other));
    }

 private:
#ifndef __CUDA_ARCH__
    friend std::ostream& operator<<(std::ostream& os, const VectorShape& shape) {
        return os << fmt::format("[{}]", shape[0]);
    }
#endif
};

template<Device DeviceId, typename DataType>
using Tensor = Vector<DeviceId, DataType, VectorShape>;

}  // namespace Blade

#ifndef __CUDA_ARCH__
template <> struct fmt::formatter<Blade::ArrayShape> : ostream_formatter {};
template <> struct fmt::formatter<Blade::PhasorShape> : ostream_formatter {};
template <> struct fmt::formatter<Blade::DelayShape> : ostream_formatter {};
template <> struct fmt::formatter<Blade::VectorShape> : ostream_formatter {};
#endif 

#endif
