/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __SETPRIO_H__
#define __SETPRIO_H__ "@(#)$Id: setprio.h,v 1.2.2.1 1999/04/13 19:45:31 jlawson Exp $"

// 'prio' is a value on the scale of 0 to 9, where 0 is the lowest
// priority and 9 is the highest priority [9 is what the priority would 
// be if priority were not set, ie is 'normal' priority.] 
//
// prio 9 code should be valid, since the priority is raised when exiting. 

extern int SetGlobalPriority(unsigned int prio);
extern int SetThreadPriority(unsigned int prio);

#endif /* __SETPRIO_H__ */
