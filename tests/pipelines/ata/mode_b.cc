#include <memory>

#include "blade/base.hh"
#include "blade/logger.hh"
#include "blade/runner.hh"
#include "blade/pipelines/ata/mode_b.hh"

extern "C" {
#include "mode_b.h"
#include "blade/pipelines/ata/mode_b_config.h"
}

using namespace Blade;
using namespace Blade::Pipelines::ATA;

static struct {
    std::unique_ptr<Logger> guard;
    std::unique_ptr<Runner<ModeB>> runner;
} instance;

bool blade_use_device(int device_id) {
    return SetCudaDevice(device_id) == Result::SUCCESS;
}

bool blade_ata_b_initialize(size_t numberOfWorkers) {
    Pipelines::ATA::ModeB::Config config = {
        .inputDims = {
            .NBEAMS = 1,
            .NANTS  = BLADE_ATA_MODE_B_INPUT_NANT,
            .NCHANS = BLADE_ATA_MODE_B_ANT_NCHAN,
            .NTIME  = BLADE_ATA_MODE_B_NTIME,
            .NPOLS  = BLADE_ATA_MODE_B_NPOL,
        },
        .channelizerRate = 4,
        .beamformerBeams = BLADE_ATA_MODE_B_OUTPUT_NBEAM,
        .castBlockSize = 512,
        .channelizerBlockSize = 512,
        .beamformerBlockSize = 512,
    };

    instance.guard = std::make_unique<Logger>();
    instance.runner = Runner<ModeB>::New(numberOfWorkers, config);

    return true;
}

void blade_ata_b_terminate() {
    instance.runner.reset();
    instance.guard.reset();
}

size_t blade_ata_b_get_input_size() {
    assert(instance.runner);
    return instance.runner->getWorker().getInputSize();
}

size_t blade_ata_b_get_output_size() {
    assert(instance.runner);
    return instance.runner->getWorker().getOutputSize();
}

bool blade_pin_memory(void* buffer, size_t size) {
    return Memory::PageLock(Vector<Device::CPU, I8>(buffer, size)) == Result::SUCCESS;
}

bool blade_ata_b_enqueue(void* input_ptr, void* output_ptr, size_t id) {
    assert(instance.runner);
    return instance.runner->enqueue([&](auto& worker){
        auto input = Vector<Device::CPU, CI8>(input_ptr, worker.getInputSize());
        auto output = Vector<Device::CPU, BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T>(output_ptr, worker.getOutputSize());

        worker.run(input, output);

        return id;
    });
}

bool blade_ata_b_dequeue(size_t* id) {
    assert(instance.runner);
    return instance.runner->dequeue(id);
}
