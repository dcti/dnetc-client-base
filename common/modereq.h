/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * --------------------------------------------------------------------     
 * This file contains functions for getting/setting/clearing
 * "mode" requests (--fetch,--flush) and the like. Client::Run() will
 * clear/run the modes when appropriate.
 *
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 * --------------------------------------------------------------------     
*/ 

#ifndef __MODEREQ_H__
#define __MODEREQ_H__ "@(#)$Id: modereq.h,v 1.19 2000/01/08 23:36:09 cyp Exp $"

#define MODEREQ_IDENT              0x00000001
#define MODEREQ_CPUINFO            0x00000002
#define MODEREQ_CONFIG             0x00000004
#define MODEREQ_CONFRESTART        0x00000008 /* restart if config ok */
#define MODEREQ_FETCH              0x00000010
#define MODEREQ_FLUSH              0x00000020
#define MODEREQ_FQUIET             0x00000040 /* fetch/flush quietly */
#define MODEREQ_CMDLINE_HELP       0x00000080
#define MODEREQ_BENCHMARK          0x00000100 /* "long" benchmark */
#define MODEREQ_BENCHMARK_QUICK    0x00000200 /* "quick" benchmark */
#define MODEREQ_BENCHMARK_ALLCORE  0x00000400 /* all cores for (each) contest */
#define MODEREQ_BENCHMARK_MASK     0x00000700 /* combined mask */
#define MODEREQ_UNLOCK             0x00000800
#define MODEREQ_IMPORT             0x00001000
#define MODEREQ_TEST               0x00002000 /* normal test */
#define MODEREQ_TEST_ALLCORE       0x00004000 /* all cores for (each) contest */
#define MODEREQ_TEST_MASK          0x00006000 /* combined mask */
#define MODEREQ_RESTART            0x00008000 /* restart client after mode processing */
#define MODEREQ_ALL                0x0000FFFF /* mask of all - needed internally */


/* get mode bit(s). if modemask is -1, all bits are returned */
extern int ModeReqIsSet(int modemask);

/* set mode bit(s), returns state of selected bits before the change */
extern int ModeReqSet(int modemask);

/* clear mode bit(s). if modemask is -1, all bits are cleared */
extern int ModeReqClear(int modemask);

/* limit bench/selftest to contest_i (additive) */
int ModeReqLimitProject(int mode, unsigned int contest_i);

/* is bench/selftest being restricted to contest_i? */
int ModeReqIsProjectLimited(int mode, unsigned int contest_i);

/* returns !0 if ModeReqRun(Client *) is currently running */
extern int ModeReqIsRunning(void);

/* set an optional argument * for a mode. The mode must support it */
extern int ModeReqSetArg( int mode, const void *arg );

#if defined(__CLIENT_H__) /* only for use from main/clirun run loop */
/* this is the mode runner. bits can be set/cleared while active.
   returns a mask of modebits that were cleared during the run. */
extern int ModeReqRun( Client *client );
#endif

#endif /* __MODEREQ_H__ */
