// Hey, Emacs, this a -*-C++-*- file !
//
// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// $Log: dctistrg.cpp,v $
// Revision 1.1  1999/01/01 01:17:41  silby
// Added dctistrg module so that a portable string
// lowercasing function can be added.
//

#include "baseincs.h"
#include "dctistrg.h"

void lowercasestring(char *string)
{
  for (char *p = string; *p; p++) *p = (char)tolower(*p);
}
