// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: csc-1k-i.cpp,v $
// Revision 1.2  1999/10/11 18:15:08  cyp
// sync'd from release branch
//
// Revision 1.1.2.1  1999/10/07 18:41:13  cyp
// sync'd from head
//
// Revision 1.1  1999/07/23 02:43:05  fordbr
// CSC cores added
//
//

#if (!defined(lint) && defined(__showids__))
const char *csc_1k_i_cpp(void) {
return "@(#)$Id: csc-1k-i.cpp,v 1.2 1999/10/11 18:15:08 cyp Exp $"; }
#endif

#include "problem.h"

#define CSC_SUFFIX 1k_i

#define INLINE_TRANSP
#include "csc-common.h"
#include "csc-1key-bitslicer.cpp"
#include "csc-1key-driver.cpp"

