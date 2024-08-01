#include "./base.hh"

using namespace Blade;
namespace bm = benchmark;

// CF32 -> CF32

static void BM_Integrator_Compute_CF32_CF32(bm::State& state) {
    IntegratorTest<Modules::Integrator, CF32, CF32> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_Integrator_Compute_CF32_CF32)
    ->Iterations(2<<13)
    ->Args({2, 1572864, 1, 2, 64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

// F16 -> F16

static void BM_Integrator_Compute_F16_F16(bm::State& state) {
    IntegratorTest<Modules::Integrator, CF16, CF16> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_Integrator_Compute_F16_F16)
    ->Iterations(2<<13)
    ->Args({2, 1572864, 1, 2, 64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);
