#ifndef BLADE_BUNDLES_ATA_MODE_B_HH
#define BLADE_BUNDLES_ATA_MODE_B_HH

#include <vector>

#include "blade/bundle.hh"

#include "blade/modules/cast.hh"
#include "blade/modules/channelizer/base.hh"
#include "blade/modules/beamformer/ata.hh"
#include "blade/modules/phasor/ata.hh"
#include "blade/modules/polarizer.hh"
#include "blade/modules/detector.hh"

namespace Blade::Bundles::ATA {

template<typename IT, typename OT>
class BLADE_API ModeB : public Bundle {
 public:
    // Configuration

    struct Config {
        ArrayShape inputShape;
        ArrayShape outputShape;

        U64 preBeamformerChannelizerRate;

        BOOL preBeamformerPolarizerConvertToCircular = false;

        F64 phasorObservationFrequencyHz;
        F64 phasorChannelBandwidthHz;
        F64 phasorTotalBandwidthHz;
        U64 phasorFrequencyStartIndex;
        U64 phasorReferenceAntennaIndex;
        LLA phasorArrayReferencePosition;
        RA_DEC phasorBoresightCoordinate;
        std::vector<XYZ> phasorAntennaPositions;
        ArrayTensor<Device::CPU, CF64> phasorAntennaCalibrations;
        std::vector<RA_DEC> phasorBeamCoordinates;

        BOOL beamformerIncoherentBeam = false;

        BOOL detectorEnable = false;
        U64 detectorIntegrationRate;
        U64 detectorNumberOfOutputPolarizations;

        U64 castBlockSize = 512;
        U64 channelizerBlockSize = 512;
        U64 phasorBlockSize = 512;
        U64 beamformerBlockSize = 512;
        U64 polarizerBlockSize = 512;
        U64 detectorBlockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return this->config;
    }

    // Input

    struct Input {
        const Tensor<Device::CPU, F64>& dut;
        const Tensor<Device::CPU, F64>& julianDate;
        const ArrayTensor<Device::CUDA, IT>& buffer;
    };

    constexpr const ArrayTensor<Device::CUDA, IT>& getInputBuffer() const {
        return this->input.buffer;
    }

    constexpr const Tensor<Device::CPU, F64>& getInputDut() const {
        return this->input.dut;
    }

    constexpr const Tensor<Device::CPU, F64>& getInputJulianDate() const {
        return this->input.julianDate;
    }

    // Output

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutputBuffer() {
        if (config.detectorEnable) {
            if constexpr (!std::is_same<OT, F32>::value) {
                return outputCast->getOutputBuffer();
            } else {
                return detector->getOutputBuffer();
            }
        } else {
            if constexpr (!std::is_same<OT, CF32>::value) {
                return complexOutputCast->getOutputBuffer();
            } else {
                return beamformer->getOutputBuffer();
            }
        }
    }

    // Constructor

    explicit ModeB(const Config& config, const Input& input, const Stream& stream)
         : Bundle(stream), config(config), input(input) {
        BL_DEBUG("Initializing Mode-B Bundle for ATA.");

        BL_DEBUG("Instantiating input cast from {} to CF32.", TypeInfo<IT>::name);
        this->connect(inputCast, {
            .blockSize = config.castBlockSize,
        }, {
            .buf = input.buffer,
        });

        BL_DEBUG("Instantiating pre-beamformer channelizer with rate {}.",
                config.preBeamformerChannelizerRate);
        this->connect(channelizer, {
            .rate = config.preBeamformerChannelizerRate,

            .blockSize = config.channelizerBlockSize,
        }, {
            .buf = inputCast->getOutputBuffer(),
        });

        BL_DEBUG("Instatiating polarizer module.")
        this->connect(polarizer, {
            .inputPolarization = POL::XY,
            .outputPolarization = (config.preBeamformerPolarizerConvertToCircular) ? POL::LR : POL::XY,
            .blockSize = config.polarizerBlockSize,
        }, {
            .buf = channelizer->getOutputBuffer(),
        });

        BL_DEBUG("Instantiating phasor module.");
        this->connect(phasor, {
            .numberOfAntennas = channelizer->getOutputBuffer().shape().numberOfAspects(),
            .numberOfFrequencyChannels = channelizer->getOutputBuffer().shape().numberOfFrequencyChannels(),
            .numberOfPolarizations = channelizer->getOutputBuffer().shape().numberOfPolarizations(),

            .observationFrequencyHz = config.phasorObservationFrequencyHz,
            .channelBandwidthHz = config.phasorChannelBandwidthHz,
            .totalBandwidthHz = config.phasorTotalBandwidthHz,
            .frequencyStartIndex = config.phasorFrequencyStartIndex,

            .referenceAntennaIndex = config.phasorReferenceAntennaIndex,
            .arrayReferencePosition = config.phasorArrayReferencePosition,
            .boresightCoordinate = config.phasorBoresightCoordinate,
            .antennaPositions = config.phasorAntennaPositions,
            .antennaCalibrations = config.phasorAntennaCalibrations,
            .beamCoordinates = config.phasorBeamCoordinates,

            .blockSize = config.phasorBlockSize,
        }, {
            .blockJulianDate = input.julianDate,
            .blockDut1 = input.dut,
        });

        BL_DEBUG("Instantiating beamformer module.");
        this->connect(beamformer, {
            .enableIncoherentBeam = config.beamformerIncoherentBeam,
            .enableIncoherentBeamSqrt = (config.detectorEnable) ? true : false,

            .blockSize = config.beamformerBlockSize,
        }, {
            .buf = polarizer->getOutputBuffer(),
            .phasors = phasor->getOutputPhasors(),
        });

        if (config.detectorEnable) {
            BL_DEBUG("Instantiating detector module.");
            this->connect(detector, {
                .integrationRate = config.detectorIntegrationRate,
                .numberOfOutputPolarizations = config.detectorNumberOfOutputPolarizations,

                .blockSize = config.detectorBlockSize,
            }, {
                .buf = beamformer->getOutputBuffer(),
            });

            if constexpr (!std::is_same<OT, F32>::value) {
                BL_DEBUG("Instantiating output cast from F32 to {}.", TypeInfo<OT>::name);
                this->connect(outputCast, {
                    .blockSize = config.castBlockSize,
                }, {
                    .buf = detector->getOutputBuffer(),
                });
            }
        } else {
            if constexpr (!std::is_same<OT, CF32>::value) {
                BL_DEBUG("Instantiating output cast from CF32 to {}.", TypeInfo<OT>::name);
                this->connect(complexOutputCast, {
                    .blockSize = config.castBlockSize,
                }, {
                    .buf = beamformer->getOutputBuffer(),
                });
            }
        }

        if (getOutputBuffer().shape() != config.outputShape) {
            BL_FATAL("Expected output buffer size ({}) mismatch with actual size ({}).",
                     config.outputShape, getOutputBuffer().shape());
            BL_CHECK_THROW(Result::ERROR);
        }
    }

 private:
    const Config config;
    Input input;

    using InputCast = typename Modules::Cast<IT, CF32>;
    std::shared_ptr<InputCast> inputCast;

    using PreChannelizer = typename Modules::Channelizer<CF32, CF32>;
    std::shared_ptr<PreChannelizer> channelizer;

    using Phasor = typename Modules::Phasor::ATA<CF32>;
    std::shared_ptr<Phasor> phasor;

    using Beamformer = typename Modules::Beamformer::ATA<CF32, CF32>;
    std::shared_ptr<Beamformer> beamformer;

    using Polarizer = typename Modules::Polarizer<CF32, CF32>;
    std::shared_ptr<Polarizer> polarizer;

    using Detector = typename Modules::Detector<CF32, F32>;
    std::shared_ptr<Detector> detector;

    std::shared_ptr<Modules::Cast<CF32, OT>> complexOutputCast;
    std::shared_ptr<Modules::Cast<F32, OT>> outputCast;
};

}  // namespace Blade::Bundles::ATA

#endif
