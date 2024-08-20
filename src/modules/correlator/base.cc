#define BL_LOG_DOMAIN "M::CORRELATOR"

#include "blade/modules/correlator.hh"

#include "correlator.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Correlator<IT, OT>::Correlator(const Config& config,
                           const Input& input,
                           const Stream& stream)
        : Module(correlator_program),
          config(config),
          input(input),
          computeRatio(config.integrationRate) {

    // Check configuration values.

    if (getInputBuffer().shape().numberOfAspects() <= 0) {
        BL_FATAL("Number of aspects ({}) should be more than zero.",
                 getInputBuffer().shape().numberOfAspects());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (getInputBuffer().shape().numberOfFrequencyChannels() <= 0) {
        BL_FATAL("Number of frequency channels ({}) should be more than zero.",
                 getInputBuffer().shape().numberOfFrequencyChannels());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (getInputBuffer().shape().numberOfTimeSamples() != 1) {
        BL_FATAL("Number of time samples ({}) should be exactly one.",
                 getInputBuffer().shape().numberOfTimeSamples());
        BL_CHECK_THROW(Result::ERROR);
    }

    if ((getInputBuffer().shape().numberOfFrequencyChannels() % config.blockSize) != 0) {
        BL_FATAL("Number of frequency channels ({}) should be divisible by block size ({}).",
                 getInputBuffer().shape().numberOfFrequencyChannels(), config.blockSize);
        BL_CHECK_THROW(Result::ERROR);
    }

    if (config.integrationRate < 1) {
        BL_FATAL("Integration size ({}) should be one (1) or more.", config.integrationRate);
        BL_CHECK_THROW(Result::ERROR);
    }

    if (config.conjugateAntennaIndex > 1) {
        BL_FATAL("Conjugate antenna index ({}) should be zero (0) for Antenna A or one (1) for Antenna B.",
                 config.conjugateAntennaIndex);
        BL_CHECK_THROW(Result::ERROR);
    }

    // TODO: Implement other polarizations.
    if (getInputBuffer().shape().numberOfPolarizations() != 2) {
        BL_FATAL("Number of polarizations ({}) should be two. Feature not implemented.",
                 getInputBuffer().shape().numberOfPolarizations());
        BL_CHECK_THROW(Result::ERROR);
    }

    // Choose best kernel based on input buffer size.

    std::string kernel_key = "correlator";
    std::string pretty_kernel_key = "Global Memory";

    if (kernel_key.empty()) {
        BL_FATAL("Can't find any compatible kernel.");
        BL_CHECK_THROW(Result::ERROR);
    }

    // Configure kernel instantiation.

    BL_CHECK_THROW(
        createKernel(
            // Kernel name.
            "main",
            // Kernel function key.
            kernel_key,
            // Kernel grid & block size.
            dim3(
                getInputBuffer().shape().numberOfAspects(),
                getInputBuffer().shape().numberOfFrequencyChannels() / config.blockSize
            ),
            config.blockSize,
            // Kernel templates.
            TypeInfo<IT>::name,
            TypeInfo<OT>::name,
            getInputBuffer().shape().numberOfAspects(),
            getInputBuffer().shape().numberOfFrequencyChannels(),
            getInputBuffer().shape().numberOfTimeSamples(),
            getInputBuffer().shape().numberOfPolarizations(),
            config.blockSize,
            config.conjugateAntennaIndex
        )
    );

    // Allocate output buffers.
    output.buf = ArrayTensor<Device::CUDA, OT>(getOutputBufferShape());

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(),
                               getOutputBuffer().shape());
    BL_INFO("Integration Rate: {}", config.integrationRate);
    BL_INFO("Correlator Kernel: {}", pretty_kernel_key);
    BL_INFO("Antenna Conjugation: {}", (config.conjugateAntennaIndex) ? "Antenna B" : "Antenna A");
}

template<typename IT, typename OT>
Result Correlator<IT, OT>::process(const U64& currentStepCount, const Stream& stream) {
    if (currentStepCount == 0) {
        cudaMemsetAsync(output.buf.data(), 0, output.buf.size_bytes(), stream);
    }
    return runKernel("main", stream, input.buf, output.buf);
}

template class BLADE_API Correlator<CF32, CF32>;

}  // namespace Blade::Modules
