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

    // Configure kernel instantiation.

    BL_CHECK_THROW(
        createKernel(
            // Kernel name.
            "main",
            // Kernel function key.
            "correlator",
            // Kernel grid & block size.
            PadGridSize(
                // TODO: Replace with right values.
                1024,
                config.blockSize
            ),
            config.blockSize,
            // Kernel templates.
            TypeInfo<IT>::name,
            TypeInfo<OT>::name,
            getInputBuffer().size(),
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
    return runKernel("main", stream, input.buf.data(), output.buf.data());
}

template class BLADE_API Correlator<CF32, CF32>;

}  // namespace Blade::Modules
