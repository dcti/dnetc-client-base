// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: csc-6b-vec.cpp,v $
// Revision 1.3  2000/06/02 06:32:55  jlawson
// sync, copy files from release branch to head
//
// Revision 1.1.2.1  1999/12/09 04:56:49  sampo
// first few files of the altivec bitslicer for CSC.  note: this is known to be broken!  it will not work!  This is just the start.
// things that I still have to do:
// 1) write conversion routines for keyhi/keylo that take 128 bit vectors
// 2) move keyhi/keylo over to vectors instead of ulongs
// 3) any other errors made by code that assumes 64bit is the maximum bitslice?
//
// Revision 1.2  1999/10/11 18:15:09  cyp
// sync'd from release branch
//
// Revision 1.1.2.1  1999/10/07 18:41:14  cyp
// sync'd from head
//
// Revision 1.1  1999/07/23 02:43:05  fordbr
// CSC cores added
//
//

#include "problem.h"
#include "vector_sim.h"

#define CSC_SUFFIX 6b_vec

//#define INLINE_TRANSP
#include "csc-common.h"
#include "csc-6bits-vec-bitslicer.cpp"
#include "csc-6bits-vec-driver.cpp"

