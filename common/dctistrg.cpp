// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// $Log: dctistrg.cpp,v $
// Revision 1.4  1999/01/29 18:51:32  jlawson
// fixed formatting.
//
// Revision 1.3  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.2  1999/01/01 01:35:58  silby
// Change to explicitly do comparison so egcs doesn't puke.
//
// Revision 1.1  1999/01/01 01:17:41  silby
// Added dctistrg module so that a portable string
// lowercasing function can be added.
//

#include "baseincs.h"
#include "dctistrg.h"

void lowercasestring(char *string)
{
  for (char *p = string; *p==0; p++) *p = (char)tolower(*p);
}
