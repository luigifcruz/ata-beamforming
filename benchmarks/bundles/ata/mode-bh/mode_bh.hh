#ifndef BENCHMARKS_BUNDLES_ATA_MODE_BH_H
#define BENCHMARKS_BUNDLES_ATA_MODE_BH_H

#include "blade/base.hh"
#include "blade/memory/custom.hh"
#include "blade/modules/gather.hh"
#include "blade/bundles/ata/mode_b.hh"
#include "blade/bundles/generic/mode_h.hh"

namespace Blade::ATA::ModeBH {

template<typename IT, typename OT>
class Benchmark : public Runner {
 public:
    using ModeB = Bundles::ATA::ModeB<IT, CF32>;
    using ModeH = Bundles::Generic::ModeH<CF32, OT>;

    struct Config {
        ArrayShape inputShape;
        ArrayShape outputShape;

        BOOL preBeamformerPolarizerConvertToCircular = false;

        F64 phasorObservationFrequencyHz;
        F64 phasorChannelBandwidthHz;
        F64 phasorTotalBandwidthHz;
        U64 phasorFrequencyStartIndex;
        U64 phasorReferenceAntennaIndex;
        LLA phasorArrayReferencePosition;
        RA_DEC phasorBoresightCoordinate;
        std::vector<XYZ> phasorAntennaPositions;
        ArrayTensor<Device::CPU, CF64> phasorAntennaCalibrations;
        std::vector<RA_DEC> phasorBeamCoordinates;

        BOOL beamformerIncoherentBeam = false;

        U64 detectorIntegrationRate;
        U64 detectorNumberOfOutputPolarizations;
    };

    explicit Benchmark(const Config& config)
         : inputDut({1}),
           inputJulianDate({1}),
           inputBuffer(config.inputShape),
           outputBuffer(config.outputShape) {

        ArrayShape gatherInputShape = config.inputShape;
        // Replace the number of antennas with the number of beams.
        gatherInputShape[0] = config.phasorBeamCoordinates.size();
        // Add one for the incoherent beam.
        if (config.beamformerIncoherentBeam) {
            gatherInputShape[0] = gatherInputShape.numberOfAspects() + 1;
        }

        ArrayShape gatherOutputShape = gatherInputShape;
        // Output should be the number of frequency channels of the output.
        gatherOutputShape[2] = config.outputShape[1] / config.inputShape[1];

        U64 gatherMultiplier = gatherOutputShape[2] / gatherInputShape[2];

        BL_DEBUG("Gather input shape: {}", gatherInputShape);
        BL_DEBUG("Gather output shape: {}", gatherOutputShape);
        BL_DEBUG("Gather multiplier: {}", gatherMultiplier);

        BL_DEBUG("Initializing Mode-B Bundle.");
        this->connect(modeB, {
            .inputShape = config.inputShape,
            .outputShape = gatherInputShape,

            .preBeamformerChannelizerRate = 1,

            .preBeamformerPolarizerConvertToCircular = config.preBeamformerPolarizerConvertToCircular,

            .phasorObservationFrequencyHz = config.phasorObservationFrequencyHz,
            .phasorChannelBandwidthHz = config.phasorChannelBandwidthHz,
            .phasorTotalBandwidthHz = config.phasorTotalBandwidthHz,
            .phasorFrequencyStartIndex = config.phasorFrequencyStartIndex,
            .phasorReferenceAntennaIndex = config.phasorReferenceAntennaIndex,
            .phasorArrayReferencePosition = config.phasorArrayReferencePosition,
            .phasorBoresightCoordinate = config.phasorBoresightCoordinate,
            .phasorAntennaPositions = config.phasorAntennaPositions,
            .phasorAntennaCalibrations = config.phasorAntennaCalibrations,
            .phasorBeamCoordinates = config.phasorBeamCoordinates,

            .beamformerIncoherentBeam = config.beamformerIncoherentBeam,
        }, {
            .dut = inputDut,
            .julianDate = inputJulianDate,
            .buffer = inputBuffer,
        });

        BL_DEBUG("Instantiating gather.");
        this->connect(gather, {
            .axis = 2,
            .multiplier = gatherMultiplier,
        }, {
            .buf = modeB->getOutputBuffer(),
        });

        BL_DEBUG("Initializing Mode-H Bundle.");
        this->connect(modeH, {
            .inputShape = gatherOutputShape,
            .outputShape = config.outputShape,

            .polarizerConvertToCircular = false,

            .detectorIntegrationRate = 1,
            .detectorNumberOfOutputPolarizations = config.detectorNumberOfOutputPolarizations,
        }, {
            .buffer = gather->getOutputBuffer(),
        });
    }

    Result transferIn(const Tensor<Device::CPU, F64>& cpuInputDut,
                      const Tensor<Device::CPU, F64>& cpuInputJulianDate,
                      const ArrayTensor<Device::CPU, IT>& cpuInputBuffer) {
        BL_CHECK(this->copy(inputDut, cpuInputDut));
        BL_CHECK(this->copy(inputJulianDate, cpuInputJulianDate));
        BL_CHECK(this->copy(inputBuffer, cpuInputBuffer));
        return Result::SUCCESS;
    }

    Result transferOut(ArrayTensor<Device::CPU, OT>& cpuOutputBuffer) {
        BL_CHECK(this->copy(outputBuffer, modeH->getOutputBuffer()));
        BL_CHECK(this->copy(cpuOutputBuffer, outputBuffer));
        return Result::SUCCESS;
    }

 private:
    std::shared_ptr<ModeB> modeB;
    std::shared_ptr<ModeH> modeH;
    std::shared_ptr<Modules::Gather<CF32, CF32>> gather;

    Duet<Tensor<Device::CPU, F64>> inputDut;
    Duet<Tensor<Device::CPU, F64>> inputJulianDate;
    Duet<ArrayTensor<Device::CUDA, IT>> inputBuffer;
    Duet<ArrayTensor<Device::CUDA, OT>> outputBuffer;
};

template<typename IT, typename OT>
class BenchmarkRunner {
 public:
    BenchmarkRunner() {
        BL_DEBUG("Configuring Pipeline.");
        config = {
            .inputShape = ArrayShape({ 20, 192, 8192, 2 }),
            .outputShape = ArrayShape({ 2+1, 192*8192*8, 1, 1 }),

            .preBeamformerPolarizerConvertToCircular = false,

            .phasorObservationFrequencyHz = 6500.125*1e6,
            .phasorChannelBandwidthHz = 0.5e6,
            .phasorTotalBandwidthHz = 1.024e9,
            .phasorFrequencyStartIndex = 352,
            .phasorReferenceAntennaIndex = 0,
            .phasorArrayReferencePosition = {
                .LON = BL_DEG_TO_RAD(-121.470733),
                .LAT = BL_DEG_TO_RAD(40.815987),
                .ALT = 1020.86,
            },
            .phasorBoresightCoordinate = {
                .RA = 0.64169,
                .DEC = 1.079896295,
            },
            .phasorAntennaPositions = {
                {-2524041.5388905862, -4123587.965024342, 4147646.4222955606},    // 1c
                {-2524068.187873109, -4123558.735413135, 4147656.21282186},       // 1e
                {-2524087.2078100787, -4123532.397416349, 4147670.9866770394},    // 1g
                {-2524103.384010733, -4123511.111598937, 4147682.4133068994},     // 1h
                {-2524056.730228759, -4123515.287949227, 4147706.4850287656},     // 1k
                {-2523986.279601761, -4123497.427940991, 4147766.732988923},      // 2a
                {-2523970.301363642, -4123515.238502669, 4147758.790023165},      // 2b
                {-2523983.5419911123, -4123528.1422073604, 4147737.872218138},    // 2c
                {-2523941.5221860334, -4123568.125040547, 4147723.8292249846},    // 2e
                {-2524074.096220788, -4123468.5182652213, 4147742.0422435375},    // 2h
                {-2524058.6409591637, -4123466.5112451194, 4147753.4513993543},   // 2j
                {-2524026.989692545, -4123480.9405167866, 4147758.2356800516},    // 2l
                {-2524048.5254066754, -4123468.3463909747, 4147757.835369889},    // 2k
                {-2524000.5641107005, -4123498.2984570004, 4147756.815976133},    // 2m
                {-2523945.086670364, -4123480.3638816103, 4147808.127865142},     // 3d
                {-2523950.6822576034, -4123444.7023326857, 4147839.7474427638},   // 3l
                {-2523880.869769226, -4123514.3375464156, 4147813.413426994},     // 4e
                {-2523930.3747946257, -4123454.3080821196, 4147842.6449955846},   // 4g
                {-2523898.1150373477, -4123456.314794732, 4147860.3045849088},    // 4j
                {-2523824.598229116, -4123527.93080514, 4147833.98936114}         // 5b
            },
            .phasorAntennaCalibrations = ArrayTensor<Device::CPU, CF64>({ 20, 192 * 1, 1, 2, }),
            .phasorBeamCoordinates = {
                {0.63722, 1.07552424},
                {0.64169, 1.079896295},
            },

            .beamformerIncoherentBeam = true,

            .detectorIntegrationRate = 4,
            .detectorNumberOfOutputPolarizations = 1,
        };
        pipeline = std::make_shared<Benchmark<IT, OT>>(config);

        for (U64 i = 0; i < pipeline->numberOfStreams(); i++) {
            inputDut1.push_back(Tensor<Device::CPU, F64>({1}));
            inputJulianDate.push_back(Tensor<Device::CPU, F64>({1}));
            inputBuffer.push_back(ArrayTensor<Device::CPU, IT>(config.inputShape));
            outputBuffer.push_back(ArrayTensor<Device::CPU, OT>(config.outputShape));
        }
    }

    Result run(const U64& totalIterations) {
        U64 dequeueCount = 0;
        U64 enqueueCount = 0;
        U64 iterations = 0;

        while (iterations < totalIterations) {
            auto inputCallback = [&](){
                const U64 i = enqueueCount++ % 2;
                return pipeline->transferIn(inputDut1[i], inputJulianDate[i], inputBuffer[i]);
            };
            auto outputCallback = [&](){
                const U64 i = dequeueCount++ % 2;
                return pipeline->transferOut(outputBuffer[i]);
            };
            BL_CHECK(pipeline->enqueue(inputCallback, outputCallback, enqueueCount, dequeueCount));

            BL_CHECK(pipeline->dequeue([&](const U64& inputId,
                                           const U64& outputId,
                                           const bool& didOutput){
                if (didOutput) {
                    iterations++;
                }
                return Result::SUCCESS;
            }));
        }

        return Result::SUCCESS;
    }

 private:
    typename Benchmark<IT, OT>::Config config;
    std::shared_ptr<Benchmark<IT, OT>> pipeline;

    std::vector<Tensor<Device::CPU, F64>> inputDut1;
    std::vector<Tensor<Device::CPU, F64>> inputJulianDate;
    std::vector<ArrayTensor<Device::CPU, IT>> inputBuffer;
    std::vector<ArrayTensor<Device::CPU, OT>> outputBuffer;
};

}  // namespace Blade::ATA::ModeBH

#endif
