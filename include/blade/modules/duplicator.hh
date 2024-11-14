#ifndef BLADE_MODULES_DUPLICATOR_GENERIC_HH
#define BLADE_MODULES_DUPLICATOR_GENERIC_HH

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

// MAYDO: Add built-in casting, if necessary.
// MAYDO: Add support for types different than ArrayTensor, if necessary.
template<typename IT, typename OT>
class BLADE_API Duplicator : public Module {
 public:
    // Configuration

    struct Config {};

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
        return "Duplicator";
    }

    // Constructor & Processing

    explicit Duplicator(const Config& config, const Input& input, const Stream& stream = {});
    Result process(const U64& currentStepCount, const Stream& stream = {}) final;

 private:
    // Variables
        
    const Config config;
    const Input input;
    Output output;
};

}  // namespace Blade::Modules

#endif
