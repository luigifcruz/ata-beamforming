#define BL_LOG_DOMAIN "P::FILE_WRITER"

#include "blade/pipelines/generic/file_writer.hh"

namespace Blade::Pipelines::Generic {

template<typename IT>
FileWriter<IT>::FileWriter(const Config& config) 
     : Accumulator(config.totalNumberOfFrequencyChannels /
                   config.stepNumberOfFrequencyChannels),
       config(config) {
    BL_DEBUG("Initializing CLI File Writer Pipeline.");

    BL_DEBUG("Instantiating GUPPI RAW file writer.");
    this->connect(guppi, {
        .filepath = config.outputGuppiFile,
        .directio = config.directio,

        .stepNumberOfBeams = config.stepNumberOfBeams,
        .stepNumberOfAntennas = config.stepNumberOfAntennas,
        .stepNumberOfFrequencyChannels = config.stepNumberOfFrequencyChannels,
        .stepNumberOfTimeSamples = config.stepNumberOfTimeSamples,
        .stepNumberOfPolarizations = config.stepNumberOfPolarizations,

        .totalNumberOfFrequencyChannels = config.totalNumberOfFrequencyChannels,

        .blockSize = config.writerBlockSize,
    }, {
        .totalBuffer = writerBuffer,
    });
}

template<typename IT>
const Result FileWriter<IT>::accumulate(const Vector<Device::CUDA, IT>& data,
                                        const cudaStream_t& stream) {
    if (guppi->getStepInputBufferSize() != data.size()) {
        BL_FATAL("Accumulate input size ({}) mismatches writer step input buffer size ({}).",
            data.size(), guppi->getStepInputBufferSize());
        return Result::ASSERTION_ERROR;
    }

    const auto offset = this->getCurrentAccumulatorStep() * guppi->getStepInputBufferSize();
    auto input = Vector<Device::CPU, IT>(writerBuffer.data() + offset, data.size());
    BL_CHECK(Memory::Copy(input, data, stream));

    return Result::SUCCESS;
}

template class BLADE_API FileWriter<CF16>;
template class BLADE_API FileWriter<CF32>;

}  // namespace Blade::Pipelines::Generic