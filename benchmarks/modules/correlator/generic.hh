#include "./base.hh"

#include "blade/modules/correlator.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_Correlator_Compute(bm::State& state) {
    CorrelatorTest<Modules::Correlator, CF32, CF32> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_Correlator_Compute)
    ->Iterations(2<<13)
    ->Args({2, 8, 1, 2, 1, 4})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute)
    ->Iterations(2<<13)
    ->Args({200, 1024, 1, 2, 1, 32})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute)
    ->Iterations(2<<13)
    ->Args({28, 262144, 1, 2, 1, 32})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute)
    ->Iterations(2<<6)
    ->Args({28, 192, 8192, 2, 8192, 32})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);