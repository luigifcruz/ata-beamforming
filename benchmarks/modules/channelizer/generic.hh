#include "blade/modules/channelizer/base.hh"

#include "./base.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_Channelizer_Compute(bm::State& state) {
    ChannelizerTest<Modules::Channelizer, CF32, CF32> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_Channelizer_Compute)
    ->Iterations(2<<13)
    ->Args({16, 192, 8192, 2, 8192})
    ->Args({28, 1, 65536, 2, 65536})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);