/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __CHECKPT_H__
#define __CHECKPT_H__ "@(#)$Id: checkpt.h,v 1.4.2.1 1999/10/16 16:41:04 cyp Exp $"

/* action flags */
#define CHECKPOINT_REFRESH 0x00
#define CHECKPOINT_OPEN    0x01
#define CHECKPOINT_CLOSE   0x02

/* Checkpoint frequency is the greater of ...
 * a) an avg block completion % change >= CHECKPOINT_FREQ_PERCDIFF 
 * b) CHECKPOINT_FREQ_SECSDIFF seconds
*/
#define CHECKPOINT_FREQ_PERCDIFF 10
#define CHECKPOINT_FREQ_SECSDIFF (10*60) 

int CheckpointAction( Client *client, int action, 
                      unsigned int load_problem_count );
// returns !0 if checkpointing is disabled

#endif /* __CHECKPT_H__ */
