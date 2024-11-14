#define BL_LOG_DOMAIN "RUNNER"

#include "blade/runner.hh"

namespace Blade {

Runner::Runner() : headIndex(0) {
    BL_DEBUG("Initializing runner state.")
}

Runner::~Runner() {
    if (queue.size() > 0) {
        BL_WARN("Runner destroyed with {} stream left inside queue.", queue.size());
    }
}

Result Runner::enqueue(const std::function<Result()>& inputCallback,
                       const std::function<Result()>& resultCallback,
                       const std::function<Result()>& outputCallback,
                       U64 inputId,
                       U64 outputId) {
    if (queue.size() == numberOfStreams()) {
        std::this_thread::yield();
        return Result::RUNNER_QUEUE_FULL;
    }

    const bool _willOutput = willOutput();
    const auto nextHeadIndex = (headIndex + 1) % numberOfStreams();

    BL_TRACE("[E] Index: {} | Input ID: {} | Output ID: {} | Will Output: {}",
             headIndex, inputId, outputId, _willOutput ? "Y" : "N");

    BL_CHECK(inputCallback());
    BL_CHECK(wait(headIndex, nextHeadIndex));
    BL_CHECK(compute(headIndex));
    if (_willOutput) {
        BL_CHECK(resultCallback());
        BL_CHECK(record(headIndex));
        BL_CHECK(outputCallback());
    } else {
        BL_CHECK(record(headIndex));
    }

    queue.push({headIndex, inputId, outputId, _willOutput});
    headIndex = nextHeadIndex;

    return Result::SUCCESS;
}

Result Runner::dequeue(const DequeueCallback& callback) {
    // If queue is empty, return immediately.
    if (queue.size() == 0) {
        std::this_thread::yield();
        return Result::RUNNER_QUEUE_EMPTY;
    }

    // If queue is not full, check if head is synchronized.
    if (queue.size() < numberOfStreams()) {
        const auto [frontIndex, inputId, outputId, didOutput] = queue.front();
        if (isSynchronized(frontIndex)) {
            queue.pop();
            BL_TRACE("[D] Index: {} | Input ID: {} | Output ID: {} | Did Output: {}",
                     frontIndex, inputId, outputId, didOutput ? "Y" : "N");
            return callback(inputId, outputId, didOutput);
        }
        return Result::RUNNER_QUEUE_NONE_AVAILABLE;
    }

    // If queue is full, wait until head is synchronized.
    const auto [frontIndex, inputId, outputId, didOutput] = queue.front();
    BL_CHECK(synchronize(frontIndex));
    queue.pop();
    BL_TRACE("[D] Index: {} | Input ID: {} | Output ID: {} | Did Output: {}",
             frontIndex, inputId, outputId, didOutput ? "Y" : "N");
    return callback(inputId, outputId, didOutput);
}

}  // namespace Blade
