#include <benchmark/benchmark.h>

#include "blade/types.hh"

#if defined(BLADE_BUNDLE_ATA_MODE_B)
#include "./ata/mode-b/generic.hh"
#endif

#if defined(BLADE_BUNDLE_ATA_MODE_B) && \
    defined(BLADE_BUNDLE_GENERIC_MODE_H)
#include "./ata/mode-bh/generic.hh"
#endif

#if defined(BLADE_BUNDLE_GENERIC_MODE_H)
#include "./generic/mode-h/generic.hh"
#endif

#if defined(BLADE_BUNDLE_GENERIC_MODE_X)
#include "./generic/mode-x/generic.hh"
#endif


BENCHMARK_MAIN();
