#ifndef BENCHMARKS_BUNDLES_GENERIC_MODE_X_H
#define BENCHMARKS_BUNDLES_GENERIC_MODE_X_H

#include "blade/base.hh"
#include "blade/bundles/generic/mode_x.hh"

namespace Blade::Generic::ModeX {

template<typename IT, typename OT>
class Benchmark : public Runner {
 public:
    using ModeX = Bundles::Generic::ModeX<IT, OT>;

    using Config = typename ModeX::Config;

    explicit Benchmark(const Config& config)
         : inputBuffer(config.inputShape),
           outputBuffer(config.outputShape) {
        this->connect(modeX, config, {
            .buffer = inputBuffer,
        });
    }

    Result transferIn(const ArrayTensor<Device::CPU, IT>& cpuInputBuffer) {
        BL_CHECK(this->copy(inputBuffer, cpuInputBuffer));
        return Result::SUCCESS;
    }

    Result transferOut(ArrayTensor<Device::CPU, OT>& cpuOutputBuffer) {
        BL_CHECK(this->copy(outputBuffer, modeX->getOutputBuffer()));
        BL_CHECK(this->copy(cpuOutputBuffer, outputBuffer));
        return Result::SUCCESS;
    }

 private:
    std::shared_ptr<ModeX> modeX;

    Duet<ArrayTensor<Device::CUDA, IT>> inputBuffer;
    Duet<ArrayTensor<Device::CUDA, OT>> outputBuffer;
};

template<typename IT, typename OT>
class BenchmarkRunner {
 public:
    BenchmarkRunner() {
        BL_DEBUG("Configuring Pipeline.");
        config = {
            .inputShape = ArrayShape({ 28, 1, 32768, 2 }),
            .outputShape = ArrayShape({ 406, 262144, 1, 4 }),

            .preCorrelationGatherRate = 8,

            .postCorrelationIntegrationRate = 1024,
        };
        pipeline = std::make_shared<Benchmark<IT, OT>>(config);

        for (U64 i = 0; i < pipeline->numberOfStreams(); i++) {
            inputBuffer.push_back(ArrayTensor<Device::CPU, IT>(config.inputShape));
            outputBuffer.push_back(ArrayTensor<Device::CPU, OT>(config.outputShape));
        }
    }

    Result run(const U64& totalIterations) {
        U64 dequeueCount = 0;
        U64 enqueueCount = 0;
        U64 iterations = 0;

        while (iterations < totalIterations) {
            auto inputCallback = [&](){
                const U64 i = enqueueCount++ % 2;
                return pipeline->transferIn(inputBuffer[i]);
            };
            auto outputCallback = [&](){
                const U64 i = dequeueCount++ % 2;
                return pipeline->transferOut(outputBuffer[i]);
            };
            BL_CHECK(pipeline->enqueue(inputCallback, outputCallback, enqueueCount, dequeueCount));

            BL_CHECK(pipeline->dequeue([&](const U64& inputId,
                                           const U64& outputId,
                                           const bool& didOutput){
                if (didOutput) {
                    iterations++;
                }
                return Result::SUCCESS;
            }));
        }

        return Result::SUCCESS;
    }

 private:
    typename Benchmark<IT, OT>::Config config;
    std::shared_ptr<Benchmark<IT, OT>> pipeline;

    std::vector<ArrayTensor<Device::CPU, IT>> inputBuffer;
    std::vector<ArrayTensor<Device::CPU, OT>> outputBuffer;
};

}  // namespace Blade::Generic::ModeX

#endif
