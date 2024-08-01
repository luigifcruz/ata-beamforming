#include <chrono>

#include "./mode_x.hh"
#include "../../../helper.hh"

using namespace Blade;

namespace bm = benchmark;
namespace chr = std::chrono;

static void BM_BundleGenericModeX(benchmark::State& state) {
    const uint64_t count = 256;
    std::shared_ptr<Generic::ModeX::BenchmarkRunner<CI8, CF32>> bench;

    BL_DISABLE_PRINT();
    Blade::InitAndProfile([&](){
        bench = std::make_shared<Generic::ModeX::BenchmarkRunner<CI8, CF32>>();
    }, state);
    BL_ENABLE_PRINT();

    for (auto _ : state) {
        auto start = chr::high_resolution_clock::now();

        BL_DISABLE_PRINT();
        BL_CHECK_THROW(bench->run(count));
        BL_ENABLE_PRINT();

        auto end = chr::high_resolution_clock::now();

        auto elapsed_seconds = chr::duration_cast<chr::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count() / count);
    }

    BL_DISABLE_PRINT();
    bench.reset();
    BL_ENABLE_PRINT();
}

BENCHMARK(BM_BundleGenericModeX)
    ->Iterations(2)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);
