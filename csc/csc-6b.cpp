// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: csc-6b.cpp,v $
// Revision 1.4  2000/06/02 06:32:55  jlawson
// sync, copy files from release branch to head
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
return "@(#)$Id: csc-6b.cpp,v 1.4 2000/06/02 06:32:55 jlawson Exp $"; }
#endif

#include "problem.h"

#define CSC_SUFFIX 6b

//#define INLINE_TRANSP
#include "csc-common.h"
#include "csc-6bits-bitslicer.cpp"
#include "csc-6bits-driver.cpp"

