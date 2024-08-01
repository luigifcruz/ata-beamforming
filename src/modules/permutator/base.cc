#define BL_LOG_DOMAIN "M::PERMUTATOR"

#include <type_traits>
#include <typeindex>

#include "blade/modules/permutator.hh"

#include "permutator.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Permutator<IT, OT>::Permutator(const Config& config,
                       const Input& input,
                       const Stream& stream)
        : Module(permutator_program),
          config(config),
          input(input) {
    if constexpr (!std::is_same<IT, OT>::value) {
        BL_FATAL("Input ({}) and output ({}) types aren't the same. Casting isn't supported by Permutator yet.",
                 TypeInfo<IT>::name, TypeInfo<OT>::name);
        BL_CHECK_THROW(Result::ERROR);
    }

    if (config.indexes.dimensions() != input.buf.shape().dimensions()) {
        BL_FATAL("Selected input rank ({}) doesn't match input shape rank ({}).",
                 config.indexes.size(), input.buf.shape().dimensions());
        BL_CHECK_THROW(Result::ERROR);
    }

    std::unordered_set<U64> indexesSet;
    for (U64 i = 0; i < config.indexes.dimensions(); i++) {
        if (indexesSet.find(config.indexes[i]) != indexesSet.end()) {
            BL_FATAL("Repeated index {} in permutator.", config.indexes[i]);
            BL_CHECK_THROW(Result::ERROR);
        }
        indexesSet.insert(config.indexes[i]);
    }

    // TODO: Add bypass for identity permutator.

    // Configure kernels.
    BL_CHECK_THROW(
        this->createKernel(
            // Kernel name.
            "main",
            // Kernel function key.
            "permutator",
            // Kernel grid & block size.
            PadGridSize(
                getInputBuffer().size(),
                config.blockSize
            ),
            config.blockSize,
            // Kernel templates.
            TypeInfo<IT>::cudaName
        )
    );

    // Allocate output buffers.
    output.buf = ArrayTensor<Device::CUDA, OT>(getOutputBufferShape());

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(), 
                               getOutputBuffer().shape());
    BL_INFO("Indexes: {}", config.indexes);
}

template<typename IT, typename OT>
Result Permutator<IT, OT>::process(const U64& currentStepCount, const Stream& stream) {
    return this->runKernel("main", stream, input.buf, output.buf, config.indexes);
}

template class BLADE_API Permutator<CI8, CI8>;
template class BLADE_API Permutator<CF16, CF16>;
template class BLADE_API Permutator<CF32, CF32>;

}  // namespace Blade::Modules
