// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// $Log: selcore.h,v $
// Revision 1.1.2.2  1998/11/08 11:51:56  remi
// Lots of $Log tags.
//

#ifndef __SELCORE_H__
#define __SELCORE_H__

//returns name for core number (0...) or "" if no such core
const char *GetCoreNameFromCoreType( unsigned int coretype ); 

#endif //__SELCORE_H__
