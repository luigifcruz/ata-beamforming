#ifndef BLADE_BUNDLES_GENERIC_MODE_X_HH
#define BLADE_BUNDLES_GENERIC_MODE_X_HH

#include <vector>

#include "blade/bundle.hh"

#include "blade/modules/gatherer.hh"
#include "blade/modules/cast.hh"
#include "blade/modules/channelizer/base.hh"
#include "blade/modules/correlator.hh"
#include "blade/modules/integrator.hh"

namespace Blade::Bundles::Generic {

template<typename IT, typename OT>
class BLADE_API ModeX : public Bundle {
 public:
    // Configuration

    struct Config {
        ArrayShape inputShape;
        ArrayShape outputShape;

        U64 preCorrelationGathererRate;

        U64 postCorrelationIntegrationRate;

        U64 gathererBlockSize = 512;
        U64 castBlockSize = 512;
        U64 channelizerBlockSize = 512;
        U64 correlatorBlockSize = 512;
        U64 integratorBlockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return this->config;
    }

    // Input

    struct Input {
        const ArrayTensor<Device::CUDA, IT>& buffer;
    };

    constexpr const Tensor<Device::CPU, F64>& getInputJulianDate() const {
        return this->input.julianDate;
    }

    // Output

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutputBuffer() {
        return integrator->getOutputBuffer();
    }

    // Constructor

    explicit ModeX(const Config& config, const Input& input, const Stream& stream)
         : Bundle(stream), config(config), input(input) {
        BL_DEBUG("Initializing Mode-X Bundle.");

        BL_DEBUG("Instantiating gatherer module.");
        this->connect(gatherer, {
            .axis = 2,
            .multiplier = config.preCorrelationGathererRate,

            .blockSize = config.gathererBlockSize,
        }, {
            .buf = input.buffer,
        });

        BL_DEBUG("Instantiating input cast module.");
        this->connect(inputCast, {
            .blockSize = config.castBlockSize,
        }, {
            .buf = gatherer->getOutputBuffer(),
        });

        BL_DEBUG("Instantiating channelizer module.");
        this->connect(channelizer, {
            .rate = config.inputShape.numberOfTimeSamples() *
                    config.preCorrelationGathererRate,

            .blockSize = config.channelizerBlockSize,
        }, {
            .buf = inputCast->getOutputBuffer(),
        });

        BL_DEBUG("Instantiating correlator module.");
        this->connect(correlator, {
            .integrationRate = 1,

            .blockSize = config.correlatorBlockSize,
        }, {
            .buf = channelizer->getOutputBuffer(),
        });

        BL_DEBUG("Instantiating integrator module.");
        this->connect(integrator, {
            .rate = config.postCorrelationIntegrationRate,

            .blockSize = config.integratorBlockSize,
        }, {
            .buf = correlator->getOutputBuffer(),
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

    using Gatherer = typename Modules::Gatherer<IT, IT>;
    std::shared_ptr<Gatherer> gatherer;

    using InputCast = typename Modules::Cast<IT, OT>;
    std::shared_ptr<InputCast> inputCast;

    using PreChannelizer = typename Modules::Channelizer<CF32, CF32>;
    std::shared_ptr<PreChannelizer> channelizer;

    using Correlator = typename Modules::Correlator<CF32, CF32>;
    std::shared_ptr<Correlator> correlator;

    using Integrator = typename Modules::Integrator<CF32, CF32>;
    std::shared_ptr<Integrator> integrator;
};

}  // namespace Blade::Bundles::Generic

#endif
