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
// Revision 1.4  1998/11/01 20:36:12  cyp
// added a 'class Client;' for forward resolution.
//
// Revision 1.3  1998/10/11 05:07:38  cyp
// Fixed MODEREQ_ constant overlaps
//
// Revision 1.2  1998/10/11 00:40:10  cyp
// Added MODEREQ_CONFIG.
//
// Revision 1.1  1998/10/08 20:49:42  cyp
// Created.
//

#ifndef __MODEREQ_H__
#define __MODEREQ_H__

#define MODEREQ_IDENT           0x0001    
#define MODEREQ_CPUINFO         0x0002
#define MODEREQ_TEST            0x0004
#define MODEREQ_CONFIG          0x0008
#define MODEREQ_FETCH           0x0010
#define MODEREQ_FLUSH           0x0020
#define MODEREQ_FFORCE          0x0040
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
class Client; /* for forward resolution */
extern int ModeReqRun( Client *client ); 

#endif /* __MODEREQ_H__ */
