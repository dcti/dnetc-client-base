/*
 * Copyright distributed.net 2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: cuda_setup.h,v 1.2 2009/04/15 19:06:47 thejet Exp $
*/

#ifndef CUDA_SETUP_H
#define CUDA_SETUP_H

#define CUDA_SETUP_INVALID_DRIVER_REVISION 1
#define CUDA_SETUP_MISSING_NVCUDA_DLL 2
#define CUDA_SETUP_INVALID_NVCUDA_PATH 3
#define CUDA_SETUP_NO_FILE_VERSION 4

// returns 0 on success
// i.e. a supported GPU + driver version + CUDA version was found
int InitializeCUDA();

#endif // CUDA_SETUP_H
