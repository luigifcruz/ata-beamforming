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
          input(input) {

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

    if (getInputBuffer().shape().numberOfTimeSamples() <= 0) {
        BL_FATAL("Number of time samples ({}) should be more than zero.", 
                 getInputBuffer().shape().numberOfTimeSamples());
        BL_CHECK_THROW(Result::ERROR);
    }

    // TODO: Implement other polarizations.
    if (getInputBuffer().shape().numberOfPolarizations() != 2) {
        BL_FATAL("Number of polarizations ({}) should be two. Feature not implemented.",
                 getInputBuffer().shape().numberOfPolarizations());
        BL_CHECK_THROW(Result::ERROR);
    }

    // TODO: Implement integration.
    if (config.integrationSize <= 0) {
        BL_FATAL("Integration size ({}) should be one. Feature not implemented.", 
                 config.integrationSize);
        BL_CHECK_THROW(Result::ERROR);
    }

    // TODO: The number of frequency channels has to be divisible by block size.

    // Choose best kernel based on input buffer size.

    std::string kernel_key;

    // Enable Shared Memory correlator if the size if less than 100 KB.
    if (((getInputBuffer().size() / getInputBuffer().shape().numberOfAspects()) * sizeof(IT)) < 100000) {
        kernel_key = "correlator_sm";
    } 

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
            getInputBuffer().size(),
            config.blockSize,
            config.integrationSize
        )
    );

    // Allocate output buffers.
    output.buf = ArrayTensor<Device::CUDA, OT>(getOutputBufferShape());

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(), 
                               getOutputBuffer().shape());
    BL_INFO("Integration Size: {}", config.integrationSize);
}

template<typename IT, typename OT>
Result Correlator<IT, OT>::process(const U64&, const Stream& stream) {
    return runKernel("main", stream, input.buf, output.buf);
}

template class BLADE_API Correlator<CF32, CF32>;

}  // namespace Blade::Modules
