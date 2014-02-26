/*
 * Copyright distributed.net 2014 - All rights reserved.
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: cuda_core_count.h,v 1.0 2014/02/23 21:04:04 zebe Exp $
 */
 
#ifndef CUDA_CORE_COUNT_H
#define CUDA_CORE_COUNT_H

// Number of cores per MP (varies with SM version)
// from NVIDIA CUDA SDK sample code 'deviceQuery'
// http://docs.nvidia.com/cuda/cuda-samples/index.html#device-query
typedef struct
{
  int SM; // 0xMm (hexidecimal notation)
          // M = SM Major version, m = SM minor version
  int Cores;
} sSMtoCores;
sSMtoCores CUDACoresPerSM[] =
{
  { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
  { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
  { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
  { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
  { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
  { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
  { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
  { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
  {   -1, -1 }
};

inline int getCUDACoresPerSM (int major, int minor)
{
  int index = 0;
  int SM = (major << 4) + minor;

  while (CUDACoresPerSM[index].SM != -1) {
    if (CUDACoresPerSM[index].SM == SM ) {
      return CUDACoresPerSM[index].Cores;
    }
    index++;
  }
  return -1;
}

#endif // CUDA_CORE_COUNT_H
