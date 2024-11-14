#define BL_LOG_DOMAIN "M::POLARIZER"

#include <type_traits>
#include <typeindex>

#include "blade/modules/polarizer.hh"

#include "polarizer.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Polarizer<IT, OT>::Polarizer(const Config& config, 
                             const Input& input, 
                             const Stream& stream)
        : Module(polarizer_program),
          config(config),
          input(input) {
    if constexpr (!std::is_same<IT, OT>::value) {
        BL_FATAL("This module requires the type of the input "
                 "({}) and output ({}) to be the same.",
                 TypeInfo<IT>::name, TypeInfo<OT>::name); 
        BL_INFO("Contact the maintainer if this "
                "functionality is required.");
        BL_CHECK_THROW(Result::ERROR);
    }

    if (config.inputPolarization == config.outputPolarization) {
        // Link output buffers
        BL_INFO("Bypass: Enabled");

        // Link output buffer or link input with output.
        BL_CHECK_THROW(Link(output.buf, input.buf));
    } else {
        // Configure kernel.

        U64 inputPolarizationSize = 0;
        U64 outputPolarizationSize = 0;
        std::string kernelName = "none";

        if ((config.inputPolarization == POL::XY) and 
            (config.outputPolarization == POL::LR)) {
                kernelName = "polarizer_xy_lr";
                inputPolarizationSize = 2;
                outputPolarizationSize = 2;
        }

        if ((config.inputPolarization == POL::XY) and
            (config.outputPolarization == POL::X)) {
                kernelName = "polarizer_xy_x";
                inputPolarizationSize = 2;
                outputPolarizationSize = 1;
        }

        if ((config.inputPolarization == POL::XY) and
            (config.outputPolarization == POL::Y)) {
                kernelName = "polarizer_xy_y";
                inputPolarizationSize = 2;
                outputPolarizationSize = 1;
        }

        if (kernelName == "none") {
            BL_FATAL("Invalid polarization configuration.");
            BL_CHECK_THROW(Result::ERROR);
        }

        BL_DEBUG("Kernel: {}", kernelName);

        // Configure kernel instantiation.
        BL_CHECK_THROW(
            this->createKernel(
                // Kernel name.
                "main",
                // Kernel function key.
                kernelName,
                // Kernel grid & block size.
                PadGridSize(
                    getInputBuffer().size(), 
                    config.blockSize
                ),
                config.blockSize,
                // Kernel templates.
                TypeInfo<IT>::name,
                TypeInfo<OT>::name
            )
        );

        // Allocate or link output buffer.

        if (inputPolarizationSize == outputPolarizationSize) {
            // Link output buffer or link input with output.
            BL_CHECK_THROW(Link(output.buf, input.buf));
            BL_DEBUG("Linking input to output.");
        } else {
            // Allocate output buffer.
            output.buf = ArrayTensor<Device::CUDA, OT>({
                getInputBuffer().shape().numberOfAspects(),
                getInputBuffer().shape().numberOfFrequencyChannels(),
                getInputBuffer().shape().numberOfTimeSamples(),
                outputPolarizationSize,
            });
            BL_DEBUG("Allocating new output buffer.");
        }
    }

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(), 
                               getOutputBuffer().shape());
}

template<typename IT, typename OT>
Result Polarizer<IT, OT>::process(const U64& currentStepCount, const Stream& stream) {
    if (config.inputPolarization == config.outputPolarization) {
        return Result::SUCCESS;
    }

    return this->runKernel("main", stream, input.buf, output.buf);
}

template class BLADE_API Polarizer<CF32, CF32>;
template class BLADE_API Polarizer<CF16, CF16>;

}  // namespace Blade::Modules
