// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: selftest.h,v $
// Revision 1.2  1999/01/01 02:45:16  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.1  1998/10/11 00:45:32  cyp
// SelfTest() is now standalone. Modified to use the same contest numbering
// conventions used everywhere else, ie 0==RC5, 1==DES.
//
//
//

#ifndef __SELFTEST_H__
#define __SELFTEST_H__

/* returns number of tests if all passed or 
   negative number of the test that failed */
int SelfTest( unsigned int contest, int cputype );

#endif
