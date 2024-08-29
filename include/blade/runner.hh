#ifndef BLADE_RUNNER_HH
#define BLADE_RUNNER_HH

#include <queue>
#include <vector>
#include <memory>

#include "blade/logger.hh"
#include "blade/pipeline.hh"
#include "blade/macros.hh"

namespace Blade {

class BLADE_API Runner : public Pipeline {
 public:
    Runner();
    ~Runner();

    typedef std::function<Result(const U64& inputId,
                                 const U64& outputId,
                                 const bool& didOutput)> DequeueCallback;

    Result enqueue(const std::function<Result()>& inputCallback,
                   const std::function<Result()>& resultCallback,
                   const std::function<Result()>& outputCallback,
                   U64 inputId,
                   U64 outputId);
    Result dequeue(const DequeueCallback& callback);

    template<typename DT, typename ST>
    Result copy(DT& dst, const ST& src) {
        return Copy(dst, src, stream(headIndex));
    }

    template<typename DT, typename ST>
    Result copy(Duet<DT>& dst, const ST& src) {
        return Copy(dst[headIndex], src, stream(headIndex));
    }

    template<typename DT, typename ST>
    Result copy(DT& dst, const Duet<ST>& src) {
        return Copy(dst, src.at(headIndex), stream(headIndex));
    }

    template<typename DT, typename ST>
    Result copy(Duet<DT>& dst, const Duet<ST>& src) {
        return Copy(dst[headIndex], src.at(headIndex), stream(headIndex));
    }

 private:
    U64 headIndex;
    std::queue<std::tuple<U64, U64, U64, bool>> queue;
};

}  // namespace Blade

#endif
