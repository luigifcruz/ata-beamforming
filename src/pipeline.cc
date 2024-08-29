#define BL_LOG_DOMAIN "PIPELINE"

#include "blade/pipeline.hh"

namespace Blade {

Pipeline::Pipeline()
     : _commited(false),
       _computeStepCount(0),
       _computeStepsPerCycle(1),
       _computeLifetimeCycles(0) {
    BL_DEBUG("Creating new pipeline.");

    _streams.resize(2);
    for (U64 i = 0; i < _streams.size(); i++) {
        BL_CUDA_CHECK_THROW(cudaStreamCreateWithFlags(_streams[i], cudaStreamNonBlocking), [&]{
            BL_FATAL("Failed to create CUDA stream: {}", err);
        });
    }

    _events.resize(2);
    for (U64 i = 0; i < _streams.size(); i++) {
        BL_CUDA_CHECK_THROW(cudaEventCreateWithFlags(_events[i], cudaEventDisableTiming), [&]{
            BL_FATAL("Failed to create CUDA event: {}", err);
        });
    }
}

void Pipeline::addModule(const std::shared_ptr<Module>& module) {
    cudaError_t val;
    if ((val = cudaPeekAtLastError()) != cudaSuccess) {
        const char* err = cudaGetErrorString(val);
        BL_FATAL("Error while creating module '{}' in position '{}': '{}'",
                module->name(), _modules.size(), err);
        std::exit(1);
    }

    if ((module->getTaint() & Taint::CHRONOUS) == Taint::CHRONOUS) {
        const auto& localRatio = module->getComputeRatio();
        if (localRatio > 1) {
            _computeStepsPerCycle *= localRatio;
            _computeStepRatios.push_back(localRatio);
        }
    }
    _modules.push_back(module);
}

Pipeline::~Pipeline() {
    BL_DEBUG("Destroying pipeline after {} lifetime compute cycles.", _computeLifetimeCycles);

    for (U64 i = 0; i < _streams.size(); i++) {
        synchronize(i);
        cudaStreamDestroy(_streams[i]);
    }

    for (U64 i = 0; i < _events.size(); i++) {
        cudaEventDestroy(_events[i]);
    }

    U64 pos = 0;
    for (auto& module : _modules) {
        const std::string moduleName = module->name();

        module.reset();

        cudaError_t val;
        if ((val = cudaPeekAtLastError()) != cudaSuccess) {
            const char* err = cudaGetErrorString(val);
            BL_FATAL("Error while destroying module '{}' in position '{}': '{}'", moduleName, pos, err);
            std::exit(1);
        }
        pos += 1;
    }

    _computeStepRatios.clear();
}

Result Pipeline::record(const U64& index) {
    BL_CUDA_CHECK(cudaEventRecord(_events[index], _streams[index]), [&]{
        BL_FATAL("Failed to record event: {}", err);
    });
    return Result::SUCCESS;
}

Result Pipeline::wait(const U64& index, const U64& waitForIndex) {
    BL_CUDA_CHECK(cudaStreamWaitEvent(_streams[index], _events[waitForIndex]), [&]{
        BL_FATAL("Failed to wait event: {}", err);
    });
    return Result::SUCCESS;
}

Result Pipeline::synchronize(const U64& index) {
    BL_CUDA_CHECK(cudaStreamSynchronize(_streams[index]), [&]{
        BL_FATAL("Failed to synchronize stream: {}", err);
    });
    return Result::SUCCESS;
}

bool Pipeline::isSynchronized(const U64& index) {
    return cudaStreamQuery(_streams[index]) == cudaSuccess;
}

Result Pipeline::commit() {
    BL_DEBUG("Commiting pipeline with {} modules and {} compute steps per cycle.",
             _modules.size(), _computeStepsPerCycle);

    // TODO: Validate pipeline topology with Taint (in-place modules).

    _computeStepRatios.push_back(1);

    return Result::SUCCESS;
}

Result Pipeline::compute(const U64& index) {
    if (!_commited) {
        BL_CHECK(commit());
        _commited = true;
    }

    U64 localStepCount = 0;
    U64 localRatioIndex = 0;
    U64 localStepOffset = 1;

    for (auto& module : _modules) {
        localStepCount = _computeStepCount / localStepOffset % _computeStepRatios[localRatioIndex];

        BL_TRACE("[M: '{}', R: {}]: Step Count: {}, Step Offset: {}, Step Ratio: {}",
                 module->name(), module->getComputeRatio(), localStepCount, localStepOffset, _computeStepRatios[localRatioIndex]);

        const auto& result = module->process(localStepCount, _streams[index]);

        if (result == Result::PIPELINE_EXHAUSTED) {
            BL_INFO("Module finished pipeline execution at {} lifetime compute cycles.", _computeLifetimeCycles);
            return Result::PIPELINE_EXHAUSTED;
        }

        if (result != Result::SUCCESS) {
            return result;
        }

        if (module->getComputeRatio() > 1) {
            if (localStepCount == (_computeStepRatios[localRatioIndex] - 1)) {
                localStepOffset += localStepCount;
                localRatioIndex += 1;
            } else {
                break;
            }
        }
    }

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("CUDA compute error: {}", err);
        return Result::CUDA_ERROR;
    });

    _computeStepCount += 1;

    if (_computeStepCount == _computeStepsPerCycle) {
        _computeStepCount = 0;
        _computeLifetimeCycles += 1;
    }

    BL_TRACE("----------------")

    return Result::SUCCESS;
}

}  // namespace Blade
