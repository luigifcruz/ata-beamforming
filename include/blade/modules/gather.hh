#ifndef BLADE_MODULES_GATHER_GENERIC_HH
#define BLADE_MODULES_GATHER_GENERIC_HH

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

// TODO: Add support for input with fixed axis larger than one.
// MAYDO: Add built-in casting, if necessary.
// MAYDO: Add support for types different than ArrayTensor, if necessary.
template<Device ID, typename IT, Device OD, typename OT>
class BLADE_API Gather : public Module {
 public:
    // Configuration

    struct Config {
        U64 axis = 0;
        U64 multiplier = 1;

        U64 copySizeThreshold = 512;
        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return this->config;
    }

    // Input

    struct Input {
        const ArrayTensor<ID, IT>& buf;
    };

    constexpr const ArrayTensor<ID, IT>& getInputBuffer() const {
        return this->input.buf;
    }

    // Output 

    struct Output {
        ArrayTensor<OD, OT> buf;
    };

    constexpr const ArrayTensor<OD, OT>& getOutputBuffer() const {
        return this->output.buf;
    }

    // Taint Registers

    constexpr Taint getTaint() const {
        return Taint::CONSUMER |
               Taint::PRODUCER |
               Taint::CHRONOUS;
    }

    constexpr U64 getComputeRatio() const {
        return computeRatio;
    }

    std::string name() const {
        return "Gather";
    }

    // Constructor & Processing

    explicit Gather(const Config& config, const Input& input, const Stream& stream = {});
    Result process(const U64& currentStepCount, const Stream& stream = {}) final;

 private:
    // Variables
        
    enum class Strategy {
        Kernel = 0,
        Copy   = 1,
    };

    const Config config;
    const Input input;
    Output output;

    U64 computeRatio;
    U64 widthSize;
    U64 widthByteSize;
    U64 heightSize;
    Strategy strategy;

    // Expected Shape

    const ArrayShape getOutputBufferShape() {
        ArrayShape::Type shape = getInputBuffer().shape();
        shape[config.axis] *= config.multiplier;
        return shape;
    }
};

}  // namespace Blade::Modules

#endif
