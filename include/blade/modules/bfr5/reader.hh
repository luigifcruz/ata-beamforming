#ifndef BLADE_MODULES_BFR5_READER_HH
#define BLADE_MODULES_BFR5_READER_HH

#include <filesystem>
#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

extern "C" {
#include "bfr5.h"
#include "radiointerferometryc99.h"
}

namespace Blade::Modules::Bfr5 {

class BLADE_API Reader : public Module {
 public:
    // Configuration

    struct Config {
        std::string filepath;
        U64 channelizerRate;

        U64 blockSize = 512;
    };

    // Input

    struct Input {
    };

    // Output

    struct Output {
    };

    // Taint Registers

    constexpr const MemoryTaint getMemoryTaint() {
        return MemoryTaint::NONE; 
    }

    // Constructor & Processing

    explicit Reader(const Config& config, const Input& input,
                    const cudaStream_t& stream);

    // Miscellaneous

    // TODO: This is the data size, right?
    const ArrayDimensions getTotalDims() const {
        return {
            .A = this->bfr5.dim_info.nbeams * this->bfr5.dim_info.nants,
            .F = this->bfr5.dim_info.nchan,
            .T = this->bfr5.dim_info.ntimes,
            .P = this->bfr5.dim_info.npol,
        };
    }

    constexpr const LLA getReferencePosition() const {
        return {
            .LON = this->bfr5.tel_info.longitude,
            .LAT = this->bfr5.tel_info.latitude,
            .ALT = this->bfr5.tel_info.altitude,
        };
    }

    constexpr const RA_DEC getBoresightCoordinates() const {
        return {
            .RA = this->bfr5.obs_info.phase_center_ra,
            .DEC = this->bfr5.obs_info.phase_center_dec
        };
    }

    constexpr const std::vector<XYZ>& getAntennaPositions() const {
        return this->antennaPositions;
    }

    constexpr const std::vector<RA_DEC>& getBeamCoordinates() const {
        return this->beamCoordinates;
    }

    constexpr const ArrayTensor<Device::CPU, CF64>& getAntennaCalibrations() const {
        return this->antennaCalibrations;
    }

 private:
    // Variables

    Config config;
    const Input input;
    Output output;

    BFR5_file_t bfr5;

    // TODO: Update from vector to ArrayTensor. 
    std::vector<XYZ> antennaPositions;
    std::vector<RA_DEC> beamCoordinates;
    ArrayTensor<Device::CPU, CF64> antennaCalibrations;

    const ArrayDimensions getAntennaCalibrationsDims() const{
        return {
            .A = getTotalDims().numberOfAspects(),
            .F = getTotalDims().numberOfFrequencyChannels(), // * config.channelizerRate,
            .T = 1,
            .P = getTotalDims().numberOfPolarizations(),
        };
    }
};

}  // namespace Blade::Modules

#endif

