#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>

#include "blade/base.hh"
#include "blade/memory/custom.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

template<typename DstType, typename SrcType>
void NB_SUBMODULE_COPY_TYPE_SHAPE_ORDER(auto& mm) {
    mm.def("copy", [](Runner& obj, DstType& dst, const SrcType& src){
        return obj.copy(dst, src);
    }, "dst"_a, "src"_a);
}

template<typename DataType, typename ShapeType>
void NB_SUBMODULE_COPY_TYPE_SHAPE(auto& mm) {
    using CpuTensor = Vector<Device::CPU, DataType, ShapeType>;
    using CudaTensor = Vector<Device::CUDA, DataType, ShapeType>;
    using DuetCpuTensor = Duet<CpuTensor>;
    using DuetCudaTensor = Duet<CudaTensor>;

    NB_SUBMODULE_COPY_TYPE_SHAPE_ORDER<CpuTensor, CpuTensor>(mm);
    NB_SUBMODULE_COPY_TYPE_SHAPE_ORDER<CpuTensor, CudaTensor>(mm);
    NB_SUBMODULE_COPY_TYPE_SHAPE_ORDER<CudaTensor, CpuTensor>(mm);
    NB_SUBMODULE_COPY_TYPE_SHAPE_ORDER<CudaTensor, CudaTensor>(mm);
    NB_SUBMODULE_COPY_TYPE_SHAPE_ORDER<DuetCpuTensor, CpuTensor>(mm);
    NB_SUBMODULE_COPY_TYPE_SHAPE_ORDER<CpuTensor, DuetCpuTensor>(mm);
    NB_SUBMODULE_COPY_TYPE_SHAPE_ORDER<DuetCpuTensor, CudaTensor>(mm);
    NB_SUBMODULE_COPY_TYPE_SHAPE_ORDER<CpuTensor, DuetCudaTensor>(mm);
    NB_SUBMODULE_COPY_TYPE_SHAPE_ORDER<DuetCudaTensor, CpuTensor>(mm);
    NB_SUBMODULE_COPY_TYPE_SHAPE_ORDER<CudaTensor, DuetCpuTensor>(mm);
    NB_SUBMODULE_COPY_TYPE_SHAPE_ORDER<DuetCudaTensor, CudaTensor>(mm);
    NB_SUBMODULE_COPY_TYPE_SHAPE_ORDER<CudaTensor, DuetCudaTensor>(mm);
    NB_SUBMODULE_COPY_TYPE_SHAPE_ORDER<DuetCudaTensor, DuetCpuTensor>(mm);
    NB_SUBMODULE_COPY_TYPE_SHAPE_ORDER<DuetCpuTensor, DuetCudaTensor>(mm);
    NB_SUBMODULE_COPY_TYPE_SHAPE_ORDER<DuetCudaTensor, DuetCudaTensor>(mm);
}

template<typename DataType>
void NB_SUBMODULE_COPY_TYPE(auto& mm) {
    NB_SUBMODULE_COPY_TYPE_SHAPE<DataType, ArrayShape>(mm);
    NB_SUBMODULE_COPY_TYPE_SHAPE<DataType, PhasorShape>(mm);
    NB_SUBMODULE_COPY_TYPE_SHAPE<DataType, VectorShape>(mm);
}

void NB_SUBMODULE_COPY(auto& mm) {
    NB_SUBMODULE_COPY_TYPE<I8>(mm);
    NB_SUBMODULE_COPY_TYPE<F16>(mm);
    NB_SUBMODULE_COPY_TYPE<F32>(mm);
    NB_SUBMODULE_COPY_TYPE<F64>(mm);

    NB_SUBMODULE_COPY_TYPE<CI8>(mm);
    NB_SUBMODULE_COPY_TYPE<CF16>(mm);
    NB_SUBMODULE_COPY_TYPE<CF32>(mm);
    NB_SUBMODULE_COPY_TYPE<CF64>(mm);
}

NB_MODULE(_runner_impl, m) {
    auto mm =
        nb::class_<Runner, Pipeline>(m, "runner")
            .def(nb::init<>())
            .def("enqueue", &Runner::enqueue,
                 "inputCallback"_a, "resultCallback"_a, "outputCallback"_a, "inputId"_a, "outputId"_a)
            .def("dequeue", &Runner::dequeue,
                 "callback"_a)
            .def("__repr__", [](Runner& obj){
                return bl::fmt::format("Runner()");
            });

    NB_SUBMODULE_COPY(mm);
}
