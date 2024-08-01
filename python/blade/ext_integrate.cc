#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "blade/base.hh"
#include "blade/modules/base.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

template<typename IT, typename OT>
void NB_SUBMODULE(auto& m, const auto& in_name, const auto& out_name) {
    using Class = Modules::Integrate<IT, OT>;

    auto mm = m.def_submodule(in_name)
               .def_submodule(out_name);

    nb::class_<Class, Module> mod(mm, "mod");

    nb::class_<typename Class::Config>(mod, "config")
        .def(nb::init<const U64&,
                      const U64&>(), "rate"_a,
                                     "block_size"_a = 512);

    nb::class_<typename Class::Input>(mod, "input")
        .def(nb::init<const ArrayTensor<Device::CUDA, IT>&>(), "buffer"_a);

    mod
        .def(nb::init<const typename Class::Config&,
                      const typename Class::Input&,
                      const Stream&>(), "config"_a,
                                        "input"_a,
                                        "stream"_a)
        .def("process", [](Class& instance, const U64& counter) {
            return instance.process(counter);
        })
        .def("get_config", &Class::getConfig, nb::rv_policy::reference)
        .def("get_input", &Class::getInputBuffer, nb::rv_policy::reference)
        .def("get_output", &Class::getOutputBuffer, nb::rv_policy::reference)
        .def("__repr__", [](Class& obj){
            return bl::fmt::format("Integrate()");
        });
}

NB_MODULE(_integrate_impl, m) {
    NB_SUBMODULE<F32, F32>(m, "in_f64", "out_f64");
    NB_SUBMODULE<CF32, CF32>(m, "in_cf32", "out_cf32");
    NB_SUBMODULE<CF16, CF16>(m, "in_cf16", "out_cf16");
    NB_SUBMODULE<CI8, CI8>(m, "in_ci8", "out_ci8");
}
