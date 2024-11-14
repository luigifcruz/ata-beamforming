#ifndef BLADE_MODULES_INTEGRATOR_GENERIC_HH
#define BLADE_MODULES_INTEGRATOR_GENERIC_HH

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
class BLADE_API Integrator : public Module {
 public:
    // Configuration

    struct Config {
        U64 size = 1;  // Number of time samples to integrate within one block.
        U64 rate = 1;  // Number of blocks to integrate together.

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
               Taint::PRODUCER |
               Taint::CHRONOUS;
    }

    constexpr U64 getComputeRatio() const {
        return computeRatio;
    }

    std::string name() const {
        return "Integrator";
    }

    // Constructor & Processing

    explicit Integrator(const Config& config, const Input& input, const Stream& stream = {});
    Result process(const U64& currentStepCount, const Stream& stream = {}) final;

 private:
    // Variables

    const Config config;
    const Input input;
    Output output;

    U64 computeRatio;

    const ArrayShape getOutputBufferShape() const {
        const auto& in = getInputBuffer().shape();

        return ArrayShape({
            static_cast<U64>(in.numberOfAspects()),
            static_cast<U64>(in.numberOfFrequencyChannels()),
            static_cast<U64>(in.numberOfTimeSamples() / config.size),
            static_cast<U64>(in.numberOfPolarizations()),
        });
    }
};

}  // namespace Blade::Modules

#endif
