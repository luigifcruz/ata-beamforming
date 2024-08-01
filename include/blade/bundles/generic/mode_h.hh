#ifndef BLADE_BUNDLES_GENERIC_MODE_H_HH
#define BLADE_BUNDLES_GENERIC_MODE_H_HH

#include "blade/bundle.hh"

#include "blade/modules/channelizer/base.hh"
#include "blade/modules/detector.hh"
#include "blade/modules/caster.hh"
#include "blade/modules/polarizer.hh"

namespace Blade::Bundles::Generic {

template<typename IT, typename OT>
class BLADE_API ModeH : public Bundle {
 public:
    // Configuration

    struct Config {
        ArrayShape inputShape;
        ArrayShape outputShape;

        BOOL polarizerConvertToCircular = false;

        U64 detectorIntegrationRate;
        U64 detectorNumberOfOutputPolarizations;

        U64 casterBlockSize = 512;
        U64 polarizerBlockSize = 512;
        U64 channelizerBlockSize = 512;
        U64 detectorBlockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return this->config;
    }

    // Input

    struct Input {
        ArrayTensor<Device::CUDA, IT> buffer;
    };

    constexpr const ArrayTensor<Device::CUDA, IT>& getInputBuffer() const {
        return this->input.buffer;
    }

    // Output

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutputBuffer() {
        return outputCaster->getOutputBuffer();
    }

    // Constructor

    explicit ModeH(const Config& config, const Input& input, const Stream& stream)
         : Bundle(stream), config(config), input(input) {
        BL_DEBUG("Initializing Mode-H Bundle.");

        BL_DEBUG("Instantiating input caster from {} to CF32.", TypeInfo<IT>::name);
        this->connect(inputCaster, {
            .blockSize = config.casterBlockSize,
        }, {
            .buf = input.buffer,
        });

        BL_DEBUG("Instantiating channelizer with rate {}.", config.inputShape.numberOfTimeSamples());
        this->connect(channelizer, {
            .rate = config.inputShape.numberOfTimeSamples(),

            .blockSize = config.channelizerBlockSize,
        }, {
            .buf = inputCaster->getOutputBuffer(),
        });

        BL_DEBUG("Instatiating polarizer module.")
        this->connect(polarizer, {
            .inputPolarization = POL::XY,
            .outputPolarization = (config.polarizerConvertToCircular) ? POL::LR : POL::XY,
            .blockSize = config.polarizerBlockSize,
        }, {
            .buf = channelizer->getOutputBuffer(),
        });

        BL_DEBUG("Instantiating detector module.");
        this->connect(detector, {
            .integrationRate = config.detectorIntegrationRate,
            .numberOfOutputPolarizations = config.detectorNumberOfOutputPolarizations,

            .blockSize = config.detectorBlockSize,
        }, {
            .buf = polarizer->getOutputBuffer(),
        });

        BL_DEBUG("Instantiating output caster from F32 to {}.", TypeInfo<OT>::name);
        this->connect(outputCaster, {
            .blockSize = config.casterBlockSize,
        }, {
            .buf = detector->getOutputBuffer(),
        });

        if (getOutputBuffer().shape() != config.outputShape) {
            BL_FATAL("Expected output buffer size ({}) mismatch with actual size ({}).",
                     config.outputShape, getOutputBuffer().shape());
            BL_CHECK_THROW(Result::ERROR);
        }
    }

 private:
    const Config config;
    Input input;

    using InputCaster = typename Modules::Caster<IT, CF32>;
    std::shared_ptr<InputCaster> inputCaster;

    using Channelizer = typename Modules::Channelizer<CF32, CF32>;
    std::shared_ptr<Channelizer> channelizer;

    using Polarizer = typename Modules::Polarizer<CF32, CF32>;
    std::shared_ptr<Polarizer> polarizer;

    using Detector = typename Modules::Detector<CF32, F32>;
    std::shared_ptr<Detector> detector;

    using OutputCaster = typename Modules::Caster<F32, OT>;
    std::shared_ptr<OutputCaster> outputCaster;
};

}  // namespace Blade::Bundles::Generic

#endif
