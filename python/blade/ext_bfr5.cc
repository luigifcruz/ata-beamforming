#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "blade/base.hh"
#include "blade/modules/base.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

template<class Class>
void NB_SUBMODULE(auto& m, const auto& name) {
    nb::class_<Class, Module> mod(m, name);

    nb::class_<typename Class::Config>(mod, "config")
        .def(nb::init<const std::string&,
                      const U64&,
                      const U64&>(), "filepath"_a,
                                     "channelizer_rate"_a,
                                     "block_size"_a = 512);

    nb::class_<typename Class::Input>(mod, "input");

    mod
        .def(nb::init<const typename Class::Config&,
                      const typename Class::Input&,
                      const Stream&>(), "config"_a,
                                        "input"_a,
                                        "stream"_a)
        .def("get_config", &Class::getConfig, nb::rv_policy::reference)
        .def("get_total_shape", &Class::getTotalShape)
        .def("get_reference_position", &Class::getReferencePosition)
        .def("get_boresight_coordinates", &Class::getBoresightCoordinates)
        .def("get_antenna_positions", &Class::getAntennaPositions, nb::rv_policy::reference)
        .def("get_beam_coordinates", &Class::getBeamCoordinates, nb::rv_policy::reference)
        .def("get_antenna_calibrations", &Class::getAntennaCalibrations, nb::rv_policy::reference)
        .def("__repr__", [](Class& obj){
            return bl::fmt::format("Bfr5Reader()");
        });
}

NB_MODULE(_bfr5_impl, m) {
    NB_SUBMODULE<Modules::Bfr5::Reader>(m, "taint_reader");
}
