/*
 * Copyright distributed.net 2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: cuda_info.h,v 1.2 2009/04/07 09:23:31 andreasb Exp $
*/

#ifndef CUDA_INFO_H
#define CUDA_INFO_H

// returns -1 if not supported
// returns 0 if no supported GPU was found
int GetNumberOfDetectedCUDAGPUs();

long GetRawCUDAGPUID(const char **cpuname);

// returns the frequency in MHz, or 0.
unsigned int GetCUDAGPUFrequency();

#endif // CUDA_INFO_H
