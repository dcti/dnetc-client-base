/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * --------------------------------------------------------------------     
 * This file contains functions for getting/setting/clearing
 * "mode" requests (--fetch,--flush) and the like. Client::Run() will
 * clear/run the modes when appropriate.
 *
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * This is a bridge module. Do not muck with the prototypes.
 * --------------------------------------------------------------------     
*/ 

#ifndef __MODEREQ_H__
#define __MODEREQ_H__ "@(#)$Id: modereq.h,v 1.12 1999/07/09 14:09:38 cyp Exp $"

#define MODEREQ_IDENT              0x0001    
#define MODEREQ_CPUINFO            0x0002
#define MODEREQ_TEST               0x0004
#define MODEREQ_CONFIG             0x0008
#define MODEREQ_FETCH              0x0010
#define MODEREQ_FLUSH              0x0020
#define MODEREQ_FQUIET             0x0040 /* fetch/flush quietly */
#define MODEREQ_CONFRESTART        0x0080 /* set restart flag after successful config? */
#define MODEREQ_BENCHMARK_RC5      0x0100
#define MODEREQ_BENCHMARK_DES      0x0200
#define MODEREQ_BENCHMARK_ALL      (MODEREQ_BENCHMARK_DES|MODEREQ_BENCHMARK_RC5)
#define MODEREQ_BENCHMARK_QUICK    0x0400
#define MODEREQ_CMDLINE_HELP       0x0800
#define MODEREQ_UNLOCK             0x1000
#define MODEREQ_IMPORT             0x2000
#define MODEREQ_RESTART            0x8000 /* restart client after mode processing */
#define MODEREQ_ALL                0xBFFF /* mask of all - needed internally */

/* get mode bit(s). if modemask is -1, all bits are returned */
extern int ModeReqIsSet(int modemask);

/* set mode bit(s), returns state of selected bits before the change */
extern int ModeReqSet(int modemask);

/* clear mode bit(s). if modemask is -1, all bits are cleared */
extern int ModeReqClear(int modemask);

/* returns !0 if Client::ModeReqRun(void) is currently running */
extern int ModeReqIsRunning(void);

/* set an optional argument * for a mode. The mode must support it */
extern int ModeReqSetArg( int mode, void *arg );

/* this is the mode runner. bits can be set/cleared while active.
   returns a mask of modebits that were cleared during the run. */
class Client; /* for forward resolution */
extern int ModeReqRun( Client *client ); 

#endif /* __MODEREQ_H__ */
