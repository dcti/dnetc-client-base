// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// $Log: modereq.h,v $
// Revision 1.5.2.2  1998/11/08 11:51:35  remi
// Lots of $Log tags.
//

#ifndef __MODEREQ_H__
#define __MODEREQ_H__

#define MODEREQ_HELP            0x0001
#define MODEREQ_CPUINFO         0x0002
#define MODEREQ_TEST            0x0004
#define MODEREQ_BENCHMARK_RC5   0x0100
#define MODEREQ_BENCHMARK_DES   0x0200
#define MODEREQ_BENCHMARK_QUICK 0x0400
#define MODEREQ_ALL             0x077F /* needed internally */

/* get mode bit(s). if modemask is -1, all bits are returned */
extern int ModeReqIsSet(int modemask);

/* set mode bit(s), returns state of selected bits before the change */
extern int ModeReqSet(int modemask);

/* clear mode bit(s). if modemask is -1, all bits are cleared */
extern int ModeReqClear(int modemask);

/* returns !0 if Client::ModeReqRun(void) is currently running */
extern int ModeReqIsRunning(void);

/* this is the mode runner. bits can be set/cleared while active.
   returns a mask of modebits that were cleared during the run. */
extern int ModeReqRun( Client *client ); 

#endif /* __MODEREQ_H__ */
