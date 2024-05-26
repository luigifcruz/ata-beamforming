#define BL_LOG_DOMAIN "M::GATHER"

#include <type_traits>
#include <typeindex>

#include "blade/modules/gather.hh"

#include "gather.jit.hh"

namespace Blade::Modules {

template<Device ID, typename IT, Device OD, typename OT>
Gather<ID, IT, OD, OT>::Gather(const Config& config,
                       const Input& input,
                       const Stream& stream)
        : Module(gather_program),
          config(config),
          input(input),
          computeRatio(config.multiplier) {
    if (ID==OD && ID==Device::CPU) {
        BL_FATAL("CPU performed Gather is not implemented yet.");
        BL_CHECK_THROW(Result::ERROR);
    }

    if constexpr (!std::is_same<IT, OT>::value) {
        BL_FATAL("Input ({}) and output ({}) types aren't the same. Casting isn't supported by Gather yet.",
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

    if (widthSize < config.copySizeThreshold && (ID==OD && ID==Device::CUDA)) {
        strategy = Strategy::Kernel;
    }
    else {
        strategy = Strategy::Copy;
    }

    // Allocate output buffers.
    output.buf = ArrayTensor<OD, OT>(getOutputBufferShape());

    // Print configuration values.

    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(),
                               getOutputBuffer().shape());
    BL_INFO("Axis: {}", config.axis);
    BL_INFO("Multiplier: {}", computeRatio);
    BL_INFO("Strategy: {}", (strategy == Strategy::Kernel) ? "KERNEL" : "COPY");
}

template<Device ID, typename IT, Device OD, typename OT>
Result Gather<ID, IT, OD,OT>::process(const U64& currentStepCount, const Stream& stream) {
    if (strategy == Strategy::Kernel) {
        cache
            .get_kernel(
                Template("accumulate")
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

template class BLADE_API Gather<Device::CUDA, CI8, Device::CUDA, CI8>;
template class BLADE_API Gather<Device::CUDA, CF16, Device::CUDA, CF16>;
template class BLADE_API Gather<Device::CUDA, CF32, Device::CUDA, CF32>;
template class BLADE_API Gather<Device::CUDA, F32, Device::CUDA, F32>;

template class BLADE_API Gather<Device::CUDA, CI8, Device::CPU, CI8>;
template class BLADE_API Gather<Device::CUDA, CF16, Device::CPU, CF16>;
template class BLADE_API Gather<Device::CUDA, CF32, Device::CPU, CF32>;
template class BLADE_API Gather<Device::CUDA, F32, Device::CPU, F32>;

template class BLADE_API Gather<Device::CPU, CI8, Device::CUDA, CI8>;
template class BLADE_API Gather<Device::CPU, CF16, Device::CUDA, CF16>;
template class BLADE_API Gather<Device::CPU, CF32, Device::CUDA, CF32>;
template class BLADE_API Gather<Device::CPU, F32, Device::CUDA, F32>;

}  // namespace Blade::Modules
