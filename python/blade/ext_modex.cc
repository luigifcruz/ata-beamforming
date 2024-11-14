#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "blade/base.hh"
#include "blade/bundles/base.hh"
#include "blade/memory/custom.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

template<typename IT, typename OT>
void NB_SUBMODULE(auto& m, const auto& in_name, const auto& out_name) {
    using Class = Bundles::Generic::ModeX<IT, OT>;

    auto mm = m.def_submodule(in_name)
               .def_submodule(out_name);

    nb::class_<Class, Bundle> mod(mm, "mod");

    nb::class_<typename Class::Config>(mod, "config")
        .def(nb::init<const ArrayShape&,
                      const ArrayShape&,

                      const U64&,

                      const U64&,
                      const U64&,

                      const U64&,
                      const U64&,
                      const U64&,
                      const U64&>(),
                                     "input_shape"_a,
                                     "output_shape"_a,

                                     "pre_correlator_stacker_rate"_a,

                                     "correlator_integration_rate"_a,
                                     "correlator_conjugate_antenna_index"_a,

                                     "stacker_block_size"_a = 512,
                                     "caster_block_size"_a = 512,
                                     "channelizer_block_size"_a = 512,
                                     "correlator_block_size"_a = 32);

    nb::class_<typename Class::Input>(mod, "input")
        .def(nb::init<const ArrayTensor<Device::CUDA, IT>&>(), "buffer"_a);

    mod
        .def(nb::init<const typename Class::Config&,
                      const typename Class::Input&,
                      const Stream&>(), "config"_a,
                                        "input"_a,
                                        "stream"_a)
        .def("get_config", &Class::getConfig, nb::rv_policy::reference)
        .def("get_input", &Class::getInputBuffer, nb::rv_policy::reference)
        .def("get_output", &Class::getOutputBuffer, nb::rv_policy::reference)
        .def("__repr__", [](Class& obj){
            return bl::fmt::format("ModeX(telescope=bl.generic)");
        });
}

NB_MODULE(_modex_impl, m) {
    NB_SUBMODULE<CF32, CF32>(m, "in_cf32", "out_cf32");
}
