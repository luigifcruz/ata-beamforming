#ifndef BLADE_MODULES_DETECTOR_HH
#define BLADE_MODULES_DETECTOR_HH

#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

// TODO: Add support for stepped integration.
template<typename IT, typename OT>
class BLADE_API Detector : public Module {
 public:
    // Configuration

    struct Config {
        U64 integrationRate;
        U64 numberOfOutputPolarizations = 4;

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
        return "Detector";
    }

    // Constructor & Processing

    explicit Detector(const Config& config, const Input& input, const Stream& stream = {});
    Result process(const U64& currentStepCount, const Stream& stream = {}) final;

 private:
    // Variables

    const Config config;
    const Input input;
    Output output;

    // Expected Shape

    const ArrayShape getOutputBufferShape() const {
        return ArrayShape({
            getInputBuffer().shape().numberOfAspects(),
            getInputBuffer().shape().numberOfFrequencyChannels(),
            getInputBuffer().shape().numberOfTimeSamples() / config.integrationRate,
            config.numberOfOutputPolarizations,
        });
    }
};

}  // namespace Blade::Modules

#endif
