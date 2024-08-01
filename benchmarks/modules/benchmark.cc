#include <benchmark/benchmark.h>

#include "blade/types.hh"

#if defined(BLADE_MODULE_CAST)
#include "./cast/generic.hh"
#endif

#if defined(BLADE_MODULE_CHANNELIZER)
#include "./channelizer/generic.hh"
#endif

#if defined(BLADE_MODULE_DETECTOR)
#include "./detector/generic.hh"
#endif

#if defined(BLADE_MODULE_CORRELATOR)
#include "./correlator/generic.hh"
#endif

#if defined(BLADE_MODULE_POLARIZER)
#include "./polarizer/generic.hh"
#endif

#if defined(BLADE_MODULE_GATHER)
#include "./gather/generic.hh"
#endif

#if defined(BLADE_MODULE_DUPLICATE)
#include "./duplicate/generic.hh"
#endif

#if defined(BLADE_MODULE_PERMUTATOR)
#include "./permutator/generic.hh"
#endif

#if defined(BLADE_MODULE_ATA_BEAMFORMER)
#include "./beamformer/ata.hh"
#endif

#if defined(BLADE_MODULE_MEERKAT_BEAMFORMER)
#include "./beamformer/meerkat.hh"
#endif

#if defined(BLADE_MODULE_VLA_BEAMFORMER)
#include "./beamformer/vla.hh"
#endif

#if defined(BLADE_MODULE_INTEGRATOR)
#include "./integrator/generic.hh"
#endif

BENCHMARK_MAIN();
