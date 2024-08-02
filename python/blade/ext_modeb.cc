#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "blade/base.hh"
#include "blade/bundles/base.hh"
#include "blade/memory/custom.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

template<template<typename...> class BaseClass, typename IT, typename OT>
void NB_SUBMODULE(auto& m, const auto& in_name, const auto& out_name) {
    using Class = BaseClass<IT, OT>;

    auto mm = m.def_submodule(in_name)
               .def_submodule(out_name);

    nb::class_<Class, Bundle> mod(mm, "mod");

    nb::class_<typename Class::Config>(mod, "config")
        .def(nb::init<const ArrayShape&,
                      const ArrayShape&,

                      const U64&,

                      const BOOL&,

                      const F64&,
                      const F64&,
                      const F64&,
                      const U64&,
                      const U64&,
                      const LLA&,
                      const RA_DEC&,
                      const std::vector<XYZ>&,
                      const ArrayTensor<Device::CPU, CF64>&,
                      const std::vector<RA_DEC>&,

                      const BOOL&,

                      const BOOL&,
                      const U64&,
                      const U64&,

                      const U64&,
                      const U64&,
                      const U64&,
                      const U64&,
                      const U64&,
                      const U64&>(),
                                     "input_shape"_a,
                                     "output_shape"_a,

                                     "pre_beamformer_channelizer_rate"_a,

                                     "pre_beamformer_polarizer_convert_to_circular"_a,

                                     "phasor_observation_frequency_hz"_a,
                                     "phasor_channel_bandwidth_hz"_a,
                                     "phasor_total_bandwidth_hz"_a,
                                     "phasor_frequency_start_index"_a,
                                     "phasor_reference_antenna_index"_a,
                                     "phasor_array_reference_position"_a,
                                     "phasor_boresight_coordinate"_a,
                                     "phasor_antenna_positions"_a,
                                     "phasor_antenna_calibrations"_a,
                                     "phasor_beam_coordinates"_a,

                                     "beamformer_incoherent_beam"_a,

                                     "detector_enable"_a,
                                     "detector_integration_rate"_a,
                                     "detector_number_of_output_polarizations"_a,

                                     "caster_block_size"_a = 512,
                                     "channelizer_block_size"_a = 512,
                                     "phasor_block_size"_a = 512,
                                     "beamformer_block_size"_a = 512,
                                     "polarizer_block_size"_a = 512,
                                     "detector_block_size"_a = 512);

    nb::class_<typename Class::Input>(mod, "input")
        .def(nb::init<const Tensor<Device::CPU, F64>&,
                      const Tensor<Device::CPU, F64>&,
                      const ArrayTensor<Device::CUDA, IT>&>(), "dut"_a,
                                                               "julian_date"_a,
                                                               "buffer"_a);

    mod
        .def(nb::init<const typename Class::Config&,
                      const typename Class::Input&,
                      const Stream&>(), "config"_a,
                                        "input"_a,
                                        "stream"_a)
        .def("get_config", &Class::getConfig, nb::rv_policy::reference)
        .def("get_input_julian_date", &Class::getInputJulianDate, nb::rv_policy::reference)
        .def("get_input_buffer", &Class::getInputBuffer, nb::rv_policy::reference)
        .def("get_input_dut", &Class::getInputDut, nb::rv_policy::reference)
        .def("get_output", &Class::getOutputBuffer, nb::rv_policy::reference)
        .def("__repr__", [](Class& obj){
            return bl::fmt::format("ModeB(telescope=bl.ata)");
        });
}

NB_MODULE(_modeb_impl, m) {
#ifdef BLADE_MODULE_ATA_BEAMFORMER
    {
        auto mm = m.def_submodule("tel_ata");
        NB_SUBMODULE<Bundles::ATA::ModeB, CI8, CF32>(mm, "in_ci8", "out_cf32");
        NB_SUBMODULE<Bundles::ATA::ModeB, CF32, CF32>(mm, "in_cf32", "out_cf32");
        NB_SUBMODULE<Bundles::ATA::ModeB, CI8, F32>(mm, "in_ci8", "out_f32");
        NB_SUBMODULE<Bundles::ATA::ModeB, CF32, F32>(mm, "in_cf32", "out_f32");
    }
#endif
#ifdef BLADE_MODULE_MEERKAT_BEAMFORMER
#endif
#ifdef BLADE_MODULE_VLA_BEAMFORMER
#endif
}
