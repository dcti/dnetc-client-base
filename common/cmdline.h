// Hey, Emacs, this a -*-C++-*- file !
//
// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: cmdline.h,v $
// Revision 1.2  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.1  1998/11/08 18:57:59  cyp
// Created.
//
//

#ifndef __CMDLINE_H__
#define __CMDLINE_H__

/* return the invalid option that triggered -help. 
   Returns NULL, if no invalid option
*/
const char *CmdLineFindInvalidOption(void);

#endif /* __CMDLINE_H__ */
