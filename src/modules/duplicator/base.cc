#define BL_LOG_DOMAIN "M::DUPLICATOR"

#include <type_traits>
#include <typeindex>

#include "blade/modules/duplicator.hh"

#include "duplicator.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Duplicator<IT, OT>::Duplicator(const Config& config,
                             const Input& input,
                             const Stream& stream)
        : Module(duplicator_program),
          config(config),
          input(input) {
    if constexpr (!std::is_same<IT, OT>::value) {
        BL_FATAL("Input ({}) and output ({}) types aren't the same. Casting isn't supported by Duplicator yet.",
                 TypeInfo<IT>::name, TypeInfo<OT>::name);
        BL_CHECK_THROW(Result::ERROR);
    }

    // Allocate output buffers.
    output.buf = ArrayTensor<Device::CUDA, OT>(input.buf.shape());

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(), 
                               getOutputBuffer().shape());
}

template<typename IT, typename OT>
Result Duplicator<IT, OT>::process(const U64& currentStepCount, const Stream& stream) {
    return Blade::Copy(output.buf, input.buf, stream);
}

template class BLADE_API Duplicator<CI8, CI8>;
template class BLADE_API Duplicator<CF16, CF16>;
template class BLADE_API Duplicator<CF32, CF32>;
template class BLADE_API Duplicator<F32, F32>;

}  // namespace Blade::Modules
