// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: csc-6b-i.cpp,v $
// Revision 1.1  1999/07/23 02:43:05  fordbr
// CSC cores added
//
//

#if (!defined(lint) && defined(__showids__))
const char *csc_6b_i_cpp(void) {
return "@(#)$Id: csc-6b-i.cpp,v 1.1 1999/07/23 02:43:05 fordbr Exp $"; }
#endif

#include "problem.h"

#define CSC_SUFFIX 6b_i

#define INLINE_TRANSP
#include "csc-common.h"
#include "csc-6bits-bitslicer.cpp"
#include "csc-6bits-driver.cpp"

