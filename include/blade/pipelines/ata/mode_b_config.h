#ifndef BLADE_PIPELINES_ATA_MODE_B_CONFIG_H
#define BLADE_PIPELINES_ATA_MODE_B_CONFIG_H

#define BLADE_ATA_MODE_B_INPUT_NANT 20
#define BLADE_ATA_MODE_B_INPUT_NCOMPLEX_BYTES 2

#define BLADE_ATA_MODE_B_ANT_NCHAN 192
#define BLADE_ATA_MODE_B_NTIME 8192
#define BLADE_ATA_MODE_B_NPOL 2

#define BLADE_ATA_MODE_B_OUTPUT_NBEAM 8
#define BLADE_ATA_MODE_B_OUTPUT_NCOMPLEX_BYTES 4

#define BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_PAD 0 // zero makes memcpy2D effectively 1D
#define BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_WIDTH 8192

#if BLADE_ATA_MODE_B_OUTPUT_NCOMPLEX_BYTES == 8
	#define BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T CF32
#else
	#define BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T CF16
#endif

#if BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_PAD % BLADE_ATA_MODE_B_OUTPUT_NCOMPLEX_BYTES != 0
	#error "BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_PAD must be a multiple of BLADE_ATA_MODE_B_OUTPUT_NCOMPLEX_BYTES"
#endif 

#define BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_DPITCH (BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_WIDTH+BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_PAD)

#endif 