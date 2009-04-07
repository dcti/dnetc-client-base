/*
 * Copyright distributed.net 2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: cuda_info.h,v 1.1 2009/04/07 08:54:36 andreasb Exp $
*/

#ifndef CUDA_INFO_H
#define CUDA_INFO_H

// returns -1 if not supported
// returns 0 if no supported GPU was found
int GetNumberOfDetectedCUDAGPUs();

#endif // CUDA_INFO_H
