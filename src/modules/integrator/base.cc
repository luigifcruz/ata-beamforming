#define BL_LOG_DOMAIN "M::INTEGRATOR"

#include <type_traits>
#include <typeindex>

#include "blade/modules/integrator.hh"

#include "integrator.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Integrator<IT, OT>::Integrator(const Config& config,
                             const Input& input,
                             const Stream& stream)
        : Module(integrator_program),
          config(config),
          input(input),
          computeRatio(config.rate) {
    if constexpr (!std::is_same<IT, OT>::value) {
        BL_FATAL("Input ({}) and output ({}) types aren't the same. Casting isn't supported by Integrator yet.",
                 TypeInfo<IT>::name, TypeInfo<OT>::name);
        BL_CHECK_THROW(Result::ERROR);
    }

    if ((input.buf.shape().numberOfTimeSamples() % config.size) != 0) {
        BL_FATAL("Input number of time samples ({}) is not divisible by the integration size ({}).",
                 input.buf.shape().numberOfTimeSamples(), config.size);
        BL_CHECK_THROW(Result::ERROR);
    }

    // Configure kernel instantiation.
    BL_CHECK_THROW(
        this->createKernel(
            // Kernel name.
            "main",
            // Kernel function key.
            "integrator",
            // Kernel grid & block size.
            PadGridSize(
                getInputBuffer().size() / getInputBuffer().shape().numberOfPolarizations() / config.size,
                config.blockSize
            ),
            config.blockSize,
            // Kernel templates.
            TypeInfo<IT>::name,
            TypeInfo<OT>::name,
            config.size,
            getInputBuffer().shape().numberOfPolarizations(),
            getInputBuffer().size() / getInputBuffer().shape().numberOfPolarizations() / config.size
        )
    );

    // Allocate output buffers.
    output.buf = ArrayTensor<Device::CUDA, OT>(getOutputBufferShape());

    // Print configuration values.

    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(),
                               getOutputBuffer().shape());
    BL_INFO("Size: {}", config.size);
    BL_INFO("Rate: {}", config.rate);
}

template<typename IT, typename OT>
Result Integrator<IT, OT>::process(const U64& currentStepCount, const Stream& stream) {
    if (currentStepCount == 0) {
        cudaMemsetAsync(output.buf.data(), 0, output.buf.size_bytes(), stream);
    }
    return runKernel("main", stream, input.buf, output.buf);
}

template class BLADE_API Integrator<CI8, CI8>;
template class BLADE_API Integrator<CF16, CF16>;
template class BLADE_API Integrator<CF32, CF32>;
template class BLADE_API Integrator<F32, F32>;

}  // namespace Blade::Modules
