#ifndef BLADE_PIPELINES_ATA_MODE_A_HH
#define BLADE_PIPELINES_ATA_MODE_A_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/cast.hh"
#include "blade/modules/channelizer.hh"
#include "blade/modules/beamformer/ata.hh"
#include "blade/modules/phasor/ata.hh"
#include "blade/modules/detector.hh"

namespace Blade::Pipelines::ATA {

template<typename OT = F32>
class BLADE_API ModeA : public Pipeline {
 public:
    struct Config {
        U64 preBeamformerChannelizerRate;

        F64 phasorObservationFrequencyHz;
        F64 phasorChannelBandwidthHz;
        F64 phasorTotalBandwidthHz;
        U64 phasorFrequencyStartIndex;
        U64 phasorReferenceAntennaIndex;
        LLA phasorArrayReferencePosition; 
        RA_DEC phasorBoresightCoordinate;
        std::vector<XYZ> phasorAntennaPositions;
        std::vector<CF64> phasorAntennaCalibrations; 
        std::vector<RA_DEC> phasorBeamCoordinates;

        U64 beamformerNumberOfAntennas;
        U64 beamformerNumberOfFrequencyChannels;
        U64 beamformerNumberOfTimeSamples;
        U64 beamformerNumberOfPolarizations;
        U64 beamformerNumberOfBeams;
        BOOL beamformerIncoherentBeam = false;

        U64 detectorIntegrationSize;
        U64 detectorNumberOfOutputPolarizations;

        U64 outputMemWidth;
        U64 outputMemPad;

        U64 castBlockSize = 512;
        U64 channelizerBlockSize = 512;
        U64 phasorBlockSize = 512;
        U64 beamformerBlockSize = 512;
        U64 detectorBlockSize = 512;
    };

    explicit ModeA(const Config& config);

    constexpr const U64 getInputSize() const {
        return channelizer->getBufferSize();
    }

    constexpr const U64 getOutputSize() const {
        return detector->getOutputSize();
    }

    Result run(const F64& blockJulianDate,
               const F64& blockDut1,
               const Vector<Device::CPU, CI8>& input,
                     Vector<Device::CPU, OT>& output);

 private:
    const Config config;

    U64 outputMemPitch;

    Vector<Device::CUDA, CI8> input;
    Vector<Device::CPU, F64> blockJulianDate;
    Vector<Device::CPU, F64> blockDut1;

    std::shared_ptr<Modules::Cast<CI8, CF32>> inputCast;
    std::shared_ptr<Modules::Channelizer<CF32, CF32>> channelizer;
    std::shared_ptr<Modules::Phasor::ATA<CF32>> phasor;
    std::shared_ptr<Modules::Beamformer::ATA<CF32, CF32>> beamformer;
    std::shared_ptr<Modules::Detector<CF32, F32>> detector;

    constexpr const Vector<Device::CUDA, OT>& getOutput() {
        return detector->getOutput();
    }
};

}  // namespace Blade::Pipelines::ATA

#endif
