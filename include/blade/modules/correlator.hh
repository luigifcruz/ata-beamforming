#ifndef BLADE_MODULES_CORRELATOR_HH
#define BLADE_MODULES_CORRELATOR_HH

#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
class BLADE_API Correlator : public Module {
 public:
    // Configuration

    struct Config {
        U64 integrationRate = 1;

        U64 blockSize = 32;
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
        return "Correlator";
    }

    // Constructor & Processing

    explicit Correlator(const Config& config, const Input& input, const Stream& stream = {});
    Result process(const U64& currentStepCount, const Stream& stream = {}) final;

 private:
    // Variables

    const Config config;
    const Input input;
    Output output;

    U64 computeRatio;

    // Expected Shape

    const ArrayShape getOutputBufferShape() const {
        const auto& in = getInputBuffer().shape();

        return ArrayShape({
            static_cast<U64>((in.numberOfAspects() * (in.numberOfAspects() + 1)) / 2),
            static_cast<U64>(in.numberOfFrequencyChannels()),
            static_cast<U64>(1),
            static_cast<U64>((in.numberOfPolarizations() == 2) ? 4 : 0),
        });
    }
};

}  // namespace Blade::Modules

#endif
