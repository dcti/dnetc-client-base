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
// Revision 1.7  1998/11/10 21:37:48  cyp
// added support for -forceunlock.
//
// Revision 1.6  1998/11/08 19:03:20  cyp
// -help (and invalid command line options) are now treated as "mode" requests.
//
// Revision 1.5  1998/11/02 04:46:09  cyp
// Added check for user break after each mode is processed. Added code to
// automatically trip a restart after mode processing (for use with config).
//
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

#define MODEREQ_IDENT              0x0001    
#define MODEREQ_CPUINFO            0x0002
#define MODEREQ_TEST               0x0004
#define MODEREQ_CONFIG             0x0008
#define MODEREQ_FETCH              0x0010
#define MODEREQ_FLUSH              0x0020
#define MODEREQ_FFORCE             0x0040
#define MODEREQ_CONFRESTART        0x0080 /* set restart flag after successful config? */
#define MODEREQ_BENCHMARK_RC5      0x0100
#define MODEREQ_BENCHMARK_DES      0x0200
#define MODEREQ_BENCHMARK_QUICK    0x0400
#define MODEREQ_CMDLINE_HELP       0x0800
#define MODEREQ_UNLOCK             0x1000
#define MODEREQ_RESTART            0x8000 /* restart client after mode processing */
#define MODEREQ_ALL                0x9FFF /* needed internally */

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
