// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: checkpt.h,v $
// Revision 1.2  1999/03/04 01:30:15  cyp
// Changed checkpoint interval to the greater of 10% change and 10 minutes.
//
// Revision 1.1  1998/11/26 07:09:42  cyp
// Merged DoCheckpoint(), UndoCheckpoint() and checkpoint deletion code into
// one function and spun it off into checkpt.cpp
//

#ifndef __CHECKPOINT_H__
#define __CHECKPOINT_H__

#define CHECKPOINT_REFRESH 0x00
#define CHECKPOINT_OPEN    0x01
#define CHECKPOINT_CLOSE   0x02

/*
 * Checkpoint frequency is the greater of ...
 * a) an avg block completion % change >= CHECKPOINT_FREQ_PERCDIFF 
 * b) CHECKPOINT_FREQ_SECSDIFF seconds
 *
*/

#define CHECKPOINT_FREQ_PERCDIFF 10
#define CHECKPOINT_FREQ_SECSDIFF (10*60) 

#endif /* __CHECKPOINT_H__ */
