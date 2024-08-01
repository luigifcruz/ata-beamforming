#ifndef BLADE_BENCHMARK_DETECTOR_GENERIC_HH
#define BLADE_BENCHMARK_DETECTOR_GENERIC_HH

#include "../../helper.hh"

namespace Blade {

template<template<typename, typename> class MUT, typename IT, typename OT>
class DetectorTest : CudaBenchmark {
 public:
    typename MUT<IT, OT>::Config config;
    std::shared_ptr<MUT<IT, OT>> module;
    ArrayTensor<Device::CUDA, IT> deviceInputBuf;

    Result run(benchmark::State& state) {
        const U64 A = state.range(0);
        const U64 I = state.range(1);
        const U64 P = state.range(2);

        InitAndProfile([&](){
            config.integrationRate = I;
            config.numberOfOutputPolarizations = P;
            config.blockSize = 512;

            deviceInputBuf = ArrayTensor<Device::CUDA, IT>({A, 192, 8192, 2});

            BL_DISABLE_PRINT();
            Create(module, config, {
                .buf = deviceInputBuf,
            }, this->getStream());
            BL_ENABLE_PRINT();
        }, state);

        for (auto _ : state) {
            BL_CHECK(this->startIteration());
            BL_CHECK(module->process(0, this->getStream()));
            BL_CHECK(this->finishIteration(state));
        }

        return Result::SUCCESS;
    }
};

}  // namespace Blade

#endif
