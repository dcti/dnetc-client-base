// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
/* This file contains functions for getting/setting/clearing
   "mode" requests from GUI menus and the like. Client::Run() will
   clear/run the modes when appropriate.
*/    
//
// $Log: modereq.h,v $
// Revision 1.1  1998/10/08 20:49:42  cyp
// Created.
//
//
//

#ifndef __MODEREQ_H__
#define __MODEREQ_H__

#define MODEREQ_IDENT           0x01    
#define MODEREQ_CPUINFO         0x02
#define MODEREQ_TEST            0x04
#define MODEREQ_FETCH           0x11
#define MODEREQ_FLUSH           0x12
#define MODEREQ_FFORCE          0x14
#define MODEREQ_BENCHMARK_RC5   0x20
#define MODEREQ_BENCHMARK_DES   0x21
#define MODEREQ_BENCHMARK_QUICK 0x22
#define MODEREQ_ALL             0x37 /* needed internally */

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
