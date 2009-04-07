/*
 * Copyright distributed.net 2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: cuda_setup.h,v 1.1 2009/04/07 08:16:28 andreasb Exp $
*/

#ifndef CUDA_SETUP_H
#define CUDA_SETUP_H

// returns 0 on success
// i.e. a supported GPU + driver version + CUDA version was found
int InitializeCUDA();

#endif // CUDA_SETUP_H
