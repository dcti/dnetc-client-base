// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: csc-1k-mmx.cpp,v $
// Revision 1.1.2.3  1999/12/11 00:34:13  cyp
// made mmx cores not collide with normal cores
//
// Revision 1.1.2.2  1999/11/23 23:39:45  remi
// csc_transP() optimized.
// modified csc_transP() calling convention.
//
// Revision 1.1.2.1  1999/11/22 18:58:11  remi
// Initial commit of MMX'fied CSC cores.
//
// Revision 1.1.2.1  1999/10/07 18:41:13  cyp
// sync'd from head
//
// Revision 1.1  1999/07/23 02:43:05  fordbr
// CSC cores added
//
//

#if (!defined(lint) && defined(__showids__))
const char *csc_1k_cpp(void) {
return "@(#)$Id: csc-1k-mmx.cpp,v 1.1.2.3 1999/12/11 00:34:13 cyp Exp $"; }
#endif

#include "problem.h"

#define CSC_SUFFIX 1k_mmx

//#define INLINE_TRANSP
#include "csc-common-mmx.h"
#include "csc-1key-bitslicer-mmx.cpp"
#include "csc-1key-driver-mmx.cpp"

