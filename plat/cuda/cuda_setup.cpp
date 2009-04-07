/*
 * Copyright distributed.net 2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: cuda_setup.cpp,v 1.1 2009/04/07 08:16:28 andreasb Exp $
*/

#include "cuda_setup.h"
#include "cputypes.h"
#include "logstuff.h"

#include <cuda.h>

// returns 0 on success
// i.e. a supported GPU + driver version + CUDA version was found
int InitializeCUDA()
{
  static int retval = -123;

  if (retval == -123) {
    retval = -1;
    #if (CLIENT_OS == OS_WIN32)
    // check for a supported minimum driver version
    #else
    retval = 0;
    #endif
  }

  return retval;
}
