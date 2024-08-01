#ifndef BLADE_MODULES_PERMUTATOR_GENERIC_HH
#define BLADE_MODULES_PERMUTATOR_GENERIC_HH

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

// MAYDO: Add built-in casting, if necessary.
// MAYDO: Add support for types different than ArrayTensor, if necessary.
template<typename IT, typename OT>
class BLADE_API Permutator : public Module {
 public:
    // Configuration

    struct Config {
        ArrayShape indexes;

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
        return "Permutator";
    }

    // Constructor & Processing

    explicit Permutator(const Config& config, const Input& input, const Stream& stream = {});
    Result process(const U64& currentStepCount, const Stream& stream = {}) final;

 private:
    // Variables

    const Config config;
    const Input input;
    Output output;

    // Expected Shape

    const ArrayShape getOutputBufferShape() {
        ArrayShape::Type outputShape;
        for (U64 i = 0; i < config.indexes.dimensions(); i++) {
            outputShape[i] = input.buf.shape()[config.indexes[i]];
        }
        return outputShape;
    }
};

}  // namespace Blade::Modules

#endif
