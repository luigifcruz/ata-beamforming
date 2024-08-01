#define BL_LOG_DOMAIN "M::DETECTOR"

#include "blade/modules/detector.hh"

#include "detector.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Detector<IT, OT>::Detector(const Config& config,
                           const Input& input,
                           const Stream& stream)
        : Module(detector_program),
          config(config),
          input(input) {
    // Check configuration values.
    if (config.integrationRate <= 0) {
        BL_WARN("Integration rate ({}) should be more than zero.", config.integrationRate);
        BL_CHECK_THROW(Result::ERROR);
    }

    if ((getInputBuffer().shape().numberOfTimeSamples() % config.integrationRate) != 0) {
        BL_FATAL("The number of time samples ({}) should be divisable "
                 "by the integration size ({}).",
                 getInputBuffer().shape().numberOfTimeSamples(),
                 config.integrationRate);
        BL_CHECK_THROW(Result::ERROR);
    }

    if (getInputBuffer().shape().numberOfPolarizations() != 2) {
        BL_FATAL("Number of polarizations ({}) should be two (2).",
                 getInputBuffer().shape().numberOfPolarizations());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (getInputBuffer().shape().numberOfAspects() <= 0) {
        BL_FATAL("Number of aspects ({}) should be more than zero.",
                 getInputBuffer().shape().numberOfAspects());
        BL_CHECK_THROW(Result::ERROR);
    }

    // Configure kernel instantiation.
    std::string kernel_key;
    switch (config.numberOfOutputPolarizations) {
        case 4: kernel_key = "detector_4pol"; break;
        case 1: kernel_key = "detector_1pol"; break;
        default:
            BL_FATAL("Number of output polarizations ({}) not supported.",
                     config.numberOfOutputPolarizations);
            BL_CHECK_THROW(Result::ERROR);
    }

    BL_CHECK_THROW(
        createKernel(
            // Kernel name.
            "main",
            // Kernel function key.
            kernel_key,
            // Kernel grid & block size.
            PadGridSize(
                getInputBuffer().size() / getInputBuffer().shape().numberOfPolarizations(),
                config.blockSize
            ),
            config.blockSize,
            // Kernel templates.
            getInputBuffer().size() / getInputBuffer().shape().numberOfPolarizations(),
            config.integrationRate
        )
    );

    // Allocate output buffers.
    output.buf = ArrayTensor<Device::CUDA, OT>(getOutputBufferShape());

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(),
                               getOutputBuffer().shape());
    BL_INFO("Integration Rate: {}", config.integrationRate);
    BL_INFO("Number of Output Polarizations: {}", config.numberOfOutputPolarizations);
}

template<typename IT, typename OT>
Result Detector<IT, OT>::process(const U64&, const Stream& stream) {
    return runKernel("main", stream, input.buf.data(), output.buf.data());
}

template class BLADE_API Detector<CF32, F32>;

}  // namespace Blade::Modules
