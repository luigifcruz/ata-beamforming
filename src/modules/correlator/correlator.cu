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

template<typename IT, typename OT, U64 A, U64 C, U64 T, U64 P, U64 BLOCK_SIZE, U64 CONJUGATE_ANTENNA>
__global__ void correlator(const ArrayTensor<Device::CUDA, IT> input, 
                                 ArrayTensor<Device::CUDA, OT> output) {
    // 1. Load antenna A and B data.
    // 2. Create temporary variables to accumulate the result.
    // 3. Add the multiply conjugate (XX = AX * CONJ(BX)) result to the temporary variables.
    // 4. Store the result in the output tensor.

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
        
        OT sumXX = OT(0.0f, 0.0f);
        OT sumXY = OT(0.0f, 0.0f);
        OT sumYX = OT(0.0f, 0.0f);
        OT sumYY = OT(0.0f, 0.0f);

        for (U64 TI = 0; TI < T; TI++) {
            const U64 ANTENNA_A_INDEX = (AAI * C * T * P) + (CI * T * P) + (TI * P);

            const auto AVAX = static_cast<CF64>(input[ANTENNA_A_INDEX + 0]);  // Antenna Voltage A Pol X
            const auto AVAY = static_cast<CF64>(input[ANTENNA_A_INDEX + 1]);  // Antenna Voltage A Pol Y

            const U64 ANTENNA_B_INDEX = (ABI * C * T * P) + (CI * T * P) + (TI * P);

            const auto AVBX = static_cast<CF64>(input[ANTENNA_B_INDEX + 0]);  // Antenna Voltage B Pol X
            const auto AVBY = static_cast<CF64>(input[ANTENNA_B_INDEX + 1]);  // Antenna Voltage B Pol Y

            if constexpr (CONJUGATE_ANTENNA == 1) {
                sumXX += static_cast<OT>(AVAX * AVBX.conj());  // AxBx'
                sumXY += static_cast<OT>(AVAX * AVBY.conj());  // AxBy'
                sumYX += static_cast<OT>(AVAY * AVBX.conj());  // AyBx'
                sumYY += static_cast<OT>(AVAY * AVBY.conj());  // AyBy'
            } else {
                sumXX += static_cast<OT>(AVAX.conj() * AVBX);  // Ax'Bx
                sumXY += static_cast<OT>(AVAX.conj() * AVBY);  // Ax'By
                sumYX += static_cast<OT>(AVAY.conj() * AVBX);  // Ay'Bx
                sumYY += static_cast<OT>(AVAY.conj() * AVBY);  // Ay'By
            }
        }

        const U64 OUTPUT_INDEX = (BASELINE_INDEX * C * OUTPUT_POLS) + (CI * OUTPUT_POLS);

        output[OUTPUT_INDEX + 0] += sumXX;
        output[OUTPUT_INDEX + 1] += sumXY;
        output[OUTPUT_INDEX + 2] += sumYX;
        output[OUTPUT_INDEX + 3] += sumYY;

#ifdef DEBUG
        printf("-- BIX: %ld/%d, BIY: %ld/%d, TIX: %ld || ABI: %ld, CI: %ld || AAI: %ld, ABI: %ld || BASELINE_INDEX: %ld, OUTPUT_INDEX: %ld\n",
               BIX, gridDim.x, BIY, gridDim.y, TIX, ABI, CI, AAI, ABI, BASELINE_INDEX, OUTPUT_INDEX);
#endif
    }
}