/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __CHECKPT_H__
#define __CHECKPT_H__ "@(#)$Id: checkpt.h,v 1.4.2.3 2000/03/04 12:59:04 jlawson Exp $"

// ---------------------------------------------------------------------

/* action flags */
#define CHECKPOINT_REFRESH 0x00       /* save current to checkpoint */
#define CHECKPOINT_OPEN    0x01       /* retreive state from checkpoint */
#define CHECKPOINT_CLOSE   0x02       /* delete any old checkpoint files */

// ---------------------------------------------------------------------

/* Checkpoints are done when either of these conditions are met:
 * a) (CHECKPOINT_FREQ_SECSDIFF seconds since last checkpoint)
 * b) (CHECKPOINT_FREQ_SECSIGNORE seconds since last checkpoint) AND
 *      (summed completion % change >= CHECKPOINT_FREQ_PERCDIFF)
*/
#define CHECKPOINT_FREQ_PERCDIFF 10           /* 10% */
#define CHECKPOINT_FREQ_SECSDIFF (10*60)      /* 10 minutes */
#define CHECKPOINT_FREQ_SECSIGNORE (1*60)     /* 1 minute */

// ---------------------------------------------------------------------

int CheckpointAction( Client *client, int action, 
                      unsigned int load_problem_count );
// returns !0 if checkpointing is disabled

// ---------------------------------------------------------------------


#endif /* __CHECKPT_H__ */

