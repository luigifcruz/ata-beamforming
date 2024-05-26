#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "blade/base.hh"
#include "blade/modules/base.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

template<Device ID, typename IT, Device OD, typename OT>
void NB_SUBMODULE_TYPE_DIRECTION(auto& m, const auto& in_datatype_name, const auto& out_datatype_name, const auto& in_dev_name, const auto& out_dev_name) {
    using Class = Modules::Gather<ID, IT, OD, OT>;

    auto mm = m.def_submodule(in_datatype_name)
               .def_submodule(out_datatype_name);

    nb::class_<Class, Module> mod(mm, "mod");

    nb::class_<typename Class::Config>(mod, "config")
        .def(nb::init<const U64&,
                      const U64&,
                      const U64&,
                      const U64&>(), "axis"_a,
                                     "multiplier"_a,
                                     "copy_size_threshold"_a = 512,
                                     "block_size"_a = 512);

    nb::class_<typename Class::Input>(mod, "input")
        .def(nb::init<const ArrayTensor<ID, IT>&>(), "buffer"_a);

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
            return bl::fmt::format("Gather()");
        });
}


template<typename DataType>
void NB_SUBMODULE_GATHER_TYPE(auto& m, const auto& in_datatype_name, const auto& out_datatype_name) {
    NB_SUBMODULE_TYPE_DIRECTION<Device::CUDA, DataType, Device::CUDA, DataType>(m, in_datatype_name, out_datatype_name, "src_cuda", "dest_cuda");
    NB_SUBMODULE_TYPE_DIRECTION<Device::CUDA, DataType, Device::CPU, DataType>(m, in_datatype_name, out_datatype_name, "src_cuda", "dest_cpu");
    NB_SUBMODULE_TYPE_DIRECTION<Device::CPU, DataType, Device::CUDA, DataType>(m, in_datatype_name, out_datatype_name, "src_cpu", "dest_cuda");
}

void NB_SUBMODULE_GATHER(auto& m) {
    NB_SUBMODULE_GATHER_TYPE<F32>(m, "in_f64", "out_f64");
    NB_SUBMODULE_GATHER_TYPE<CF32>(m, "in_cf32", "out_cf32");
    NB_SUBMODULE_GATHER_TYPE<CF16>(m, "in_cf16", "out_cf16");
    NB_SUBMODULE_GATHER_TYPE<CI8>(m, "in_ci8", "out_ci8");
}

NB_MODULE(_gather_impl, m) {
    NB_SUBMODULE_GATHER(m);
}
