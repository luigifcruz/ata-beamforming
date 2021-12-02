#ifndef BLADE_PIPELINES_ATA_MODE_B_HHH
#define BLADE_PIPELINES_ATA_MODE_B_HHH

#include <deque>

#include "blade/pipeline.hh"
#include "blade/manager.hh"

#include "blade/modules/cast/base.hh"
#include "blade/modules/checker/base.hh"
#include "blade/modules/channelizer/base.hh"
#include "blade/modules/beamformer/ata.hh"

extern "C" {
    #include "blade/pipelines/ata/mode_b_config.h"
}

using namespace std::chrono;

namespace Blade::Pipelines::ATA {

class ModeB : public Pipeline {
 public:
    struct Config {
        ArrayDims inputDims;
        std::size_t channelizerRate = 4;
        std::size_t beamformerBeams = 16;
        std::size_t castBlockSize = 512;
        std::size_t channelizerBlockSize = 512;
        std::size_t beamformerBlockSize = 512;
    };

    explicit ModeB(const Config& configuration);

    std::size_t getInputSize() const {
        return channelizer->getBufferSize();
    }

    std::size_t getOutputSize() const {
        return beamformer->getOutputSize();
    }

    Result run(const std::span<CI8>& in, std::span<uint8_t>& out);

 protected:
    Result setupModules() final;
    Result setupMemory() final;

    Result loopUpload() final;
    Result loopProcess(cudaStream_t& cudaStream) final;
    Result loopDownload() final;

 private:
    const Config& config;

    std::span<CI8> input;
    std::span<uint8_t> output;

    std::span<CF32> phasors;
    std::span<CI8> bufferA;
    std::span<CF32> bufferB;
    std::span<CF32> bufferC;
    std::span<CF32> bufferD;
    std::span<BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T> bufferE;

    std::unique_ptr<Modules::Cast> cast;
    std::unique_ptr<Modules::Beamformer::ATA> beamformer;
    std::unique_ptr<Modules::Channelizer> channelizer;
};

}  // namespace Blade::Pipelines::ATA

#endif
