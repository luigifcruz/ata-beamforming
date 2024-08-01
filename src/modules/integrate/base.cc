#define BL_LOG_DOMAIN "M::INTEGRATE"

#include <type_traits>
#include <typeindex>

#include "blade/modules/integrate.hh"

#include "integrate.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Integrate<IT, OT>::Integrate(const Config& config,
                             const Input& input,
                             const Stream& stream)
        : Module(integrate_program),
          config(config),
          input(input),
          computeRatio(config.rate) {
    if constexpr (!std::is_same<IT, OT>::value) {
        BL_FATAL("Input ({}) and output ({}) types aren't the same. Casting isn't supported by Integrate yet.",
                 TypeInfo<IT>::name, TypeInfo<OT>::name);
        BL_CHECK_THROW(Result::ERROR);
    }

    if (input.buf.shape().numberOfTimeSamples() != 1) {
        BL_FATAL("Number of time samples should be one.");
        BL_CHECK_THROW(Result::ERROR);
    }

    // Configure kernel instantiation.
    BL_CHECK_THROW(
        this->createKernel(
            // Kernel name.
            "main",
            // Kernel function key.
            "integrate",
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

    // Allocate output buffers.
    output.buf = ArrayTensor<Device::CUDA, OT>(input.buf.shape());

    // Print configuration values.

    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(),
                               getOutputBuffer().shape());
    BL_INFO("Rate: {}", config.rate);
}

template<typename IT, typename OT>
Result Integrate<IT, OT>::process(const U64& currentStepCount, const Stream& stream) {
    if (currentStepCount == 0) {
        cudaMemset(output.buf.data(), 0, output.buf.size_bytes());
    }
    return runKernel("main", stream, input.buf, output.buf);
}

template class BLADE_API Integrate<CI8, CI8>;
template class BLADE_API Integrate<CF16, CF16>;
template class BLADE_API Integrate<CF32, CF32>;
template class BLADE_API Integrate<F32, F32>;

}  // namespace Blade::Modules
