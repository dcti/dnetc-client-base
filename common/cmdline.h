/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __CMDLINE_H__
#define __CMDLINE_H__ "@(#)$Id: cmdline.h,v 1.6 2000/06/02 06:24:54 jlawson Exp $"

// runlevel=0 = parse cmdline, >0==exec modes && print messages
// for init'd cmdline options. returns !0 if app should be terminated
int ParseCommandline( Client *client, 
                      int run_level, int argc, const char *argv[], 
                      int *retcodeP, int logging_is_initialized );

#endif /* __CMDLINE_H__ */
