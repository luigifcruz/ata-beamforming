#define BL_LOG_DOMAIN "M::GATHERER"

#include <type_traits>
#include <typeindex>

#include "blade/modules/gatherer.hh"

#include "gatherer.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Gatherer<IT, OT>::Gatherer(const Config& config,
                       const Input& input,
                       const Stream& stream)
        : Module(gatherer_program),
          config(config),
          input(input),
          computeRatio(config.multiplier) {
    if constexpr (!std::is_same<IT, OT>::value) {
        BL_FATAL("Input ({}) and output ({}) types aren't the same. Casting isn't supported by Gatherer yet.",
                 TypeInfo<IT>::name, TypeInfo<OT>::name);
        BL_CHECK_THROW(Result::ERROR);
    }

    if (config.axis >= input.buf.shape().dimensions()) {
        BL_FATAL("Selected input axis ({}) is larger than input shape dimensions ({}).",
                 config.axis, input.buf.shape().dimensions());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (config.multiplier <= 0) {
        BL_FATAL("Multiplier ({}) should be more than zero.", config.multiplier);
        BL_CHECK_THROW(Result::ERROR);
    }

    widthSize = 1;
    for (U64 i = config.axis; i < input.buf.shape().dimensions(); i++) {
        widthSize *= input.buf.shape()[i];
    }
    widthByteSize = widthSize * sizeof(IT);
    BL_DEBUG("Width size of {} elements.", widthSize);
    BL_DEBUG("Step copy size of {} bytes.", widthByteSize);

    heightSize = 1;
    for (U64 i = 0; i < config.axis; i++) {
        heightSize *= input.buf.shape()[i];
    }
    BL_DEBUG("Height size of {} elements.", heightSize);

    if (widthSize < config.copySizeThreshold) {
        strategy = Strategy::Kernel;
    } else {
        strategy = Strategy::Copy;
    }

    // Allocate output buffers.
    output.buf = ArrayTensor<Device::CUDA, OT>(getOutputBufferShape());

    // Print configuration values.

    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(),
                               getOutputBuffer().shape());
    BL_INFO("Axis: {}", config.axis);
    BL_INFO("Multiplier: {}", computeRatio);
    BL_INFO("Strategy: {}", (strategy == Strategy::Kernel) ? "KERNEL" : "COPY");
}

template<typename IT, typename OT>
Result Gatherer<IT, OT>::process(const U64& currentStepCount, const Stream& stream) {
    if (strategy == Strategy::Kernel) {
        cache
            .get_kernel(
                Template("gatherer")
                    .instantiate(TypeInfo<IT>::name)
            )
            ->configure(
                PadGridSize(input.buf.size(), config.blockSize),
                config.blockSize,
                0,
                stream
            )
            ->launch(
                input.buf,
                output.buf,
                config.axis,
                currentStepCount * input.buf.shape()[config.axis]
            );
    }

    if (strategy == Strategy::Copy) {
        BL_CHECK(
            Copy2D(
                output.buf,
                widthByteSize * computeRatio,
                widthByteSize * currentStepCount,

                input.buf,
                widthByteSize,
                0,

                widthByteSize,
                heightSize,
                stream
            )
        );
    }

    return Result::SUCCESS;
}

template class BLADE_API Gatherer<CI8, CI8>;
template class BLADE_API Gatherer<CF16, CF16>;
template class BLADE_API Gatherer<CF32, CF32>;
template class BLADE_API Gatherer<F32, F32>;

}  // namespace Blade::Modules
