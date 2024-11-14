#include "./base.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_Permutator_Compute(bm::State& state) {
    PermutatorTest<Modules::Permutator, CF32, CF32> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_Permutator_Compute)
    ->Iterations(2<<13)
    ->Args({0, 1, 2, 3})  // AFTP (Identity)
    ->Args({0, 2, 1, 3})  // ATFP
    ->Args({0, 2, 3, 1})  // ATPF
    ->UseManualTime()
    ->Unit(bm::kMillisecond);