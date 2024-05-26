#ifndef BLADE_BENCHMARK_GATHER_GENERIC_HH
#define BLADE_BENCHMARK_GATHER_GENERIC_HH

#include "blade/modules/gather.hh"

#include "../../helper.hh"

namespace Blade {

template<template<Device, typename, Device, typename> class MUT, Device ID, typename IT, Device OD, typename OT>
class GatherTest : CudaBenchmark {
 public:
    typename MUT<ID, IT, OD, OT>::Config config;
    std::shared_ptr<MUT<ID, IT, OD, OT>> module;
    ArrayTensor<ID, IT> deviceInputBuf;

    Result run(benchmark::State& state) {
        const U64 Axis = state.range(0);
        const U64 Mult = state.range(1);
        const U64 F = state.range(2);
        const U64 T = state.range(3);

        InitAndProfile([&](){
            config.axis = Axis;
            config.multiplier = Mult;
            config.blockSize = 512;

            deviceInputBuf = ArrayTensor<ID, IT>({5, F, T, 2});

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
