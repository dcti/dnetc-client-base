// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: csc-6b-mmx.cpp,v $
// Revision 1.1.2.1  1999/12/12 11:05:59  remi
// Moved from directory csc/x86/
//
// Revision 1.1.2.2  1999/12/11 00:34:13  cyp
// made mmx cores not collide with normal cores
//
// Revision 1.1.2.1  1999/11/22 18:58:11  remi
// Initial commit of MMX'fied CSC cores.
//
// Revision 1.1.2.1  1999/10/07 18:41:14  cyp
// sync'd from head
//
// Revision 1.1  1999/07/23 02:43:05  fordbr
// CSC cores added
//
//

#if (!defined(lint) && defined(__showids__))
const char *csc_6b_cpp(void) {
return "@(#)$Id: csc-6b-mmx.cpp,v 1.1.2.1 1999/12/12 11:05:59 remi Exp $"; }
#endif

#include "problem.h"

#define CSC_SUFFIX 6b_mmx

//#define INLINE_TRANSP
#include "csc-common-mmx.h"
#include "csc-6bits-bitslicer-mmx.cpp"
#include "csc-6bits-driver-mmx.cpp"

