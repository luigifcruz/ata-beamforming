#include "blade/memory/base.hh"

//#define DEBUG

using namespace Blade;

// Input Shape:       [A, F, T, P]
// Blocks per Grid:   [A, F / BLOCK_SIZE]
// Threads per Block: [BLOCK_SIZE]
// 
// EXAMPLE (BLOCK_SIZE = 4):
// Input Shape:       [20, 200, 4, 2]
// Blocks per Grid:   [20, 4]
// Threads Per Block: [50]

//
// Global memory version without shared memory.
//

template<typename IT, typename OT, U64 A, U64 C, U64 T, U64 P, U64 BLOCK_SIZE>
__global__ void correlator(const ArrayTensor<Device::CUDA, IT> input, 
                                 ArrayTensor<Device::CUDA, OT> output) {
    // 1. Load antenna A and B data.
    // 2. Do the multiply conjugate (XX = AX * CONJ(BX)).
    // 3. Store the result in the output tensor.

    // Get Block index.
    
    const U64 BIX = blockIdx.x;  // Block Index X
    const U64 BIY = blockIdx.y;  // Block Index Y

    // Get Thread index.

    const U64 TIX = threadIdx.x;  // Thread Index X

    // Calculate constants.

    const U64 OUTPUT_POLS = 4;                // XX, XY, YX, YY
    const U64 AAI = BIX;                      // Antenna A Index
    const U64 CI = TIX + (BIY * BLOCK_SIZE);  // Channel Index

    // Run the correlation and store the result in the output tensor.

    for (U64 ABI = AAI; ABI < A; ABI++) {
        const U64 BASELINE_INDEX = ((AAI * (2 * A - AAI + 1)) / 2) + (ABI - AAI);

        for (U64 TI = 0; TI < T; TI++) {
            const U64 ANTENNA_A_INDEX = (AAI * C * T * P) + (CI * T * P) + (TI * P);

            const auto AVAX = static_cast<CF64>(input[ANTENNA_A_INDEX + 0]);  // Antenna Voltage A Pol X
            const auto AVAY = static_cast<CF64>(input[ANTENNA_A_INDEX + 1]);  // Antenna Voltage A Pol Y

            const U64 ANTENNA_B_INDEX = (ABI * C * T * P) + (CI * T * P) + (TI * P);

            const auto AVBX = static_cast<CF64>(input[ANTENNA_B_INDEX + 0]);  // Antenna Voltage B Pol X
            const auto AVBY = static_cast<CF64>(input[ANTENNA_B_INDEX + 1]);  // Antenna Voltage B Pol Y

            const auto XX = AVAX * AVBX.conj();  // XX
            const auto XY = AVAX * AVBY.conj();  // XY
            const auto YX = AVAY * AVBX.conj();  // YX
            const auto YY = AVAY * AVBY.conj();  // YY

            const U64 OUTPUT_INDEX = (BASELINE_INDEX * C * T * OUTPUT_POLS) + (CI * T * OUTPUT_POLS) + (TI * OUTPUT_POLS);

            output[OUTPUT_INDEX + 0] += static_cast<OT>(XX);
            output[OUTPUT_INDEX + 1] += static_cast<OT>(XY);
            output[OUTPUT_INDEX + 2] += static_cast<OT>(YX);
            output[OUTPUT_INDEX + 3] += static_cast<OT>(YY);
                
#ifdef DEBUG
            printf("-- BIX: %ld/%d, BIY: %ld/%d, TIX: %ld || ABI: %ld, CI: %ld, TI: %ld || AAI: %ld, ABI: %ld || BASELINE_INDEX: %ld, OUTPUT_INDEX: %ld\n",
                   BIX, gridDim.x, BIY, gridDim.y, TIX, ABI, CI, TI, ANTENNA_A_INDEX, ANTENNA_B_INDEX, BASELINE_INDEX, OUTPUT_INDEX);
#endif
        }
    }
}