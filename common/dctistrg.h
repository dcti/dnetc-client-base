// Hey, Emacs, this a -*-C++-*- file !
//
// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// $Log: dctistrg.h,v $
// Revision 1.3  1999/01/29 18:51:14  jlawson
// changed named of guard define, since it might conflict with a
// system header.
//
// Revision 1.2  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.1  1999/01/01 01:17:41  silby
// Added dctistrg module so that a portable string
// lowercasing function can be added.
//


#ifndef __DCTISTRING_H__
#define __DCTISTRING_H__

void lowercasestring(char *string);

#endif
