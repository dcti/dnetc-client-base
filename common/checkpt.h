/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __CHECKPT_H__
#define __CHECKPT_H__ "@(#)$Id: checkpt.h,v 1.4 1999/04/06 11:55:43 cyp Exp $"

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

#endif /* __CHECKPT_H__ */
