/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __CHECKPT_H__
#define __CHECKPT_H__ "@(#)$Id: checkpt.h,v 1.4.2.4 2000/10/28 15:43:24 cyp Exp $"

/* action flags */
#define CHECKPOINT_REFRESH 0x00       /* save current to checkpoint */
#define CHECKPOINT_OPEN    0x01       /* retreive state from checkpoint */
#define CHECKPOINT_CLOSE   0x02       /* delete any old checkpoint files */

/* CheckpointAction() returns !0 if checkpointing is disabled */
int CheckpointAction( Client *client, int action, 
                      unsigned int load_problem_count );

#endif /* __CHECKPT_H__ */

