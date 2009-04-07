/*
 * Copyright distributed.net 2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: cuda_info.cpp,v 1.1 2009/04/07 08:54:35 andreasb Exp $
*/

#include "cuda_info.h"
#include "cuda_setup.h"

#include <cuda_runtime.h>

// returns -1 if not supported
// returns 0 if no supported GPU was found
int GetNumberOfDetectedCUDAGPUs()
{
  static int gpucount = -123;

  if (gpucount == -123) {
    gpucount = -1;
    if (InitializeCUDA() == 0) {
      cudaDeviceProp deviceProp;
      cudaError_t rc = cudaGetDeviceCount(&gpucount);
      if (rc != cudaSuccess) {
        gpucount = -1;
      } else {
        rc = cudaGetDeviceProperties(&deviceProp, 0); /* Only supports the first device */
        if (rc != cudaSuccess || (deviceProp.major == 9999 && deviceProp.minor == 9999))
          gpucount = 0;
      }
    }
  }

  return gpucount;
}
