// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: setprio.h,v $
// Revision 1.1  1998/09/28 01:31:40  cyp
// Created. Note: priority is now on a scale of 0-9 (9 being "normal").
//
//

#ifndef __SETPRIO_H__
#define __SETPRIO_H__

// 'prio' is a value on the scale of 0 to 9, where 0 is the lowest
// priority and 9 is the highest priority [9 is what the priority would 
// be if priority were not set, ie is 'normal' priority.] 
//
// prio 9 code should be valid, since the priority is raised when exiting. 

extern int SetGlobalPriority(unsigned int prio);
extern int SetThreadPriority(unsigned int prio);

#endif /* __SETPRIO_H__ */

