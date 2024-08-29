#include "blade/base.hh"
#include "blade/modules/caster.hh"
#include "blade/modules/stacker.hh"

using namespace Blade;

template<typename IT, typename OT>
class ExamplePipeline : public Runner {
 public:
    struct Config {
        ArrayShape inputShape;
        ArrayShape outputShape;
    };

    explicit ExamplePipeline(const Config& config) : inputBuffer(config.inputShape),
                                                     outputBuffer(config.outputShape) {
        this->connect(inputCaster, {}, {
            .buf = inputBuffer,
        });

        this->connect(stacker, {
            .axis = 2,
            .multiplier = config.outputShape[2] / config.inputShape[2],
        }, {
            .buf = inputCaster->getOutputBuffer(),
        });

        this->connect(outputCaster, {}, {
            .buf = stacker->getOutputBuffer(),
        });
    }

    Result transferIn(const ArrayTensor<Device::CPU, IT>& cpuInputBuffer) {
        BL_CHECK(this->copy(inputBuffer, cpuInputBuffer));
        return Result::SUCCESS;
    }

    Result transferResult() {
        BL_CHECK(this->copy(outputBuffer, outputCaster->getOutputBuffer()));
        return Result::SUCCESS;
    }

    Result transferOut(ArrayTensor<Device::CPU, OT>& cpuOutputBuffer) {
        BL_CHECK(this->copy(cpuOutputBuffer, outputBuffer));
        return Result::SUCCESS;
    }

 private:
    std::shared_ptr<Modules::Caster<IT, F32>> inputCaster;
    std::shared_ptr<Modules::Stacker<F32, F32>> stacker;
    std::shared_ptr<Modules::Caster<F32, OT>> outputCaster;

    Duet<ArrayTensor<Device::CUDA, IT>> inputBuffer;
    Duet<ArrayTensor<Device::CUDA, OT>> outputBuffer;
};

int main() {
    using Pipeline = ExamplePipeline<I8, F32>;

    // Configuring pipeline.
    //
    // This example will take I8 samples as input and produce F32 as output.
    // It will also perform a concatenation in the time samples dimension.

    Pipeline::Config config = {
        .inputShape = ArrayShape({ 1, 1, 2, 1 }),
        .outputShape = ArrayShape({ 1, 1, 2*4, 1 }),
    };

    auto pipeline = std::make_shared<Pipeline>(config);

    // Allocating buffers.
    //
    // The data will be stored in the CPU memory, but the pipeline will
    // transfer it to the GPU memory before processing. We need to allocate
    // multiple buffers to allow the pipeline to process multiple batches
    // in parallel.

    std::vector<ArrayTensor<Device::CPU, I8>> inputBuffer;
    std::vector<ArrayTensor<Device::CPU, F32>> outputBuffer;

    for (U64 i = 0; i < pipeline->numberOfStreams(); i++) {
        inputBuffer.push_back(ArrayTensor<Device::CPU, I8>(config.inputShape));
        outputBuffer.push_back(ArrayTensor<Device::CPU, F32>(config.outputShape));
    }

    for (U64 i = 0; i < inputBuffer.size(); i++) {
        for (U64 j = 0; j < inputBuffer[i].size(); j++) {
            inputBuffer[i][j] = j + 1;
        }
    }

    // Running the pipeline.
    //
    // The pipeline will process the data in two batches. This parallelism is
    // important because while one batch is being uploaded to the GPU, the
    // other one is being processed.
    //
    // The output of each batch will be printed to the console. The output of
    // all batches should be the same because the input data is not changing.
    // Expect the output of each batch to be the input data repeated 10 times.


    U64 dequeueCount = 0;
    U64 enqueueCount = 0;

    U64 iterations = 0;
    const U64 totalIterations = 8;

    while (iterations < totalIterations) {
        auto inputCallback = [&](){
            return pipeline->transferIn(inputBuffer[enqueueCount++ % 2]);
        };
        auto resultCallback = [&](){
            return pipeline->transferResult();
        };
        auto outputCallback = [&](){
            return pipeline->transferOut(outputBuffer[dequeueCount++ % 2]);
        };
        pipeline->enqueue(inputCallback, resultCallback, outputCallback, enqueueCount, dequeueCount);

        pipeline->dequeue([&](const U64& inputId,
                              const U64& outputId,
                              const bool& didOutput){
            BL_INFO("Input ID: {} | Output ID: {} | Did Output: {}", inputId, outputId, didOutput);
            if (didOutput) {
                BL_INFO("Input:  {}", inputBuffer[inputId % 2])
                BL_INFO("Output: {}", outputBuffer[outputId % 2]);
                iterations++;
            }
            return Result::SUCCESS;
        });
    }

    BL_INFO("Example pipeline finished.");

    return 0;
}
