/*
 * Copyright distributed.net 2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: cuda_info.h,v 1.3 2010/05/27 00:38:20 snikkel Exp $
*/

#ifndef CUDA_INFO_H
#define CUDA_INFO_H

// returns -1 if not supported
// returns 0 if no supported GPU was found
int GetNumberOfDetectedCUDAGPUs();

long GetRawCUDAGPUID(const char **cpuname);

// returns the frequency in MHz, or 0.
unsigned int GetCUDAGPUFrequency();

// Number of cores per MP (varies with SM version)
// from NVIDIA CUDA SDK 'deviceQuery'
#define MAX_CUDA_MAJOR 2
static int CUDACoresPerSM[] = { -1, 8, 32 };

#endif // CUDA_INFO_H
