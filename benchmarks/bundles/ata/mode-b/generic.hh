#include <chrono>

#include "./mode_b.hh"
#include "../../../helper.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_BundleATAModeB(benchmark::State& state) {
    const uint64_t count = 256;
    std::shared_ptr<ATA::ModeB::BenchmarkRunner<CI8, CF32>> bench;

    BL_DISABLE_PRINT();
    Blade::InitAndProfile([&](){
        bench = std::make_shared<ATA::ModeB::BenchmarkRunner<CI8, CF32>>();
    }, state);
    BL_ENABLE_PRINT();

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();

        BL_DISABLE_PRINT();
        if (bench->run(count) != Result::SUCCESS) {
            BL_CHECK_THROW(Result::ERROR);
        }
        BL_ENABLE_PRINT();

        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds =
          std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);

        state.SetIterationTime(elapsed_seconds.count() / count);
    }

    BL_DISABLE_PRINT();
    bench.reset();
    BL_ENABLE_PRINT();
}

BENCHMARK(BM_BundleATAModeB)
    ->Iterations(2)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);
