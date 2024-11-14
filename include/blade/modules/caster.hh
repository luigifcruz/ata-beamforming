#ifndef BLADE_MODULES_CASTER_GENERIC_HH
#define BLADE_MODULES_CASTER_GENERIC_HH

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
class BLADE_API Caster : public Module {
 public:
    // Configuration

    struct Config {
        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return this->config;
    }

    // Input

    struct Input {
        const ArrayTensor<Device::CUDA, IT>& buf;
    };

    constexpr const ArrayTensor<Device::CUDA, IT>& getInputBuffer() const {
        return this->input.buf;
    }

    // Output 

    struct Output {
        ArrayTensor<Device::CUDA, OT> buf;
    };

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutputBuffer() const {
        return this->output.buf;
    }

    // Taint Registers

    constexpr Taint getTaint() const {
        return Taint::CONSUMER |
               Taint::PRODUCER;
    }

    std::string name() const {
        return "Caster";
    }

    // Constructor & Processing

    explicit Caster(const Config& config, const Input& input, const Stream& stream = {});
    Result process(const U64& currentStepCount, const Stream& stream = {}) final;

 private:
    // Variables

    const Config config;
    const Input input;
    Output output;

    // Expected Shape

    const ArrayShape getOutputBufferShape() const {
        return getInputBuffer().shape();
    }
};

}  // namespace Blade::Modules

#endif
