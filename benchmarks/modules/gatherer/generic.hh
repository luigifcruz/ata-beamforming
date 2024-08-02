#include "./base.hh"

using namespace Blade;
namespace bm = benchmark;

// CF32 -> CF32

static void BM_Gatherer_Compute_CF32_CF32(bm::State& state) {
    GathererTest<Modules::Gatherer, CF32, CF32> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_Gatherer_Compute_CF32_CF32)
    ->Iterations(2<<13)
    ->Args({ 2, 64,    192, 8192})  // ATA default config
    ->Args({ 2, 64, 131072,    1})  // VLA default config
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

// F16 -> F16

static void BM_Gatherer_Compute_F16_F16(bm::State& state) {
    GathererTest<Modules::Gatherer, CF16, CF16> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_Gatherer_Compute_F16_F16)
    ->Iterations(2<<13)
    ->Args({ 2, 64,    192, 8192})  // ATA default config
    ->Args({ 2, 64, 131072,    1})  // VLA default config
    ->UseManualTime()
    ->Unit(bm::kMillisecond);