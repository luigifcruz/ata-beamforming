#include "./base.hh"

using namespace Blade;
namespace bm = benchmark;

// CF32 -> CF32

static void BM_Duplicator_Compute_CF32_CF32(bm::State& state) {
    DuplicatorTest<Modules::Duplicator, CF32, CF32> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_Duplicator_Compute_CF32_CF32)
    ->Iterations(2<<13)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

// F32 -> F32

static void BM_Duplicator_Compute_F32_F32(bm::State& state) {
    DuplicatorTest<Modules::Duplicator, F32, F32> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_Duplicator_Compute_F32_F32)
    ->Iterations(2<<13)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);