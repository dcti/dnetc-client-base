/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __CMDLINE_H__
#define __CMDLINE_H__ "@(#)$Id: cmdline.h,v 1.3.2.2 2000/09/17 11:46:29 cyp Exp $"

// runlevel=0 = parse cmdline, >0==exec modes && print messages
// returns !0 if app should be terminated; (retcodeP then has exit code)
int ParseCommandline( Client *client, 
                      int run_level, int argc, const char *argv[], 
                      int *retcodeP, int restarted );

#endif /* __CMDLINE_H__ */
