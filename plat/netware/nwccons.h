/*
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * client console management functions.
 * written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * $Id: nwccons.h,v 1.2 2002/09/02 00:35:51 andreasb Exp $
*/
#ifndef __CNW_CLIENT_CONSOLE_H__
#define __CNW_CLIENT_CONSOLE_H__ 

#ifdef __cplusplus
extern "C" {
#endif

int nwCliCheckForUserBreak(void); /* RaiseExitRequestTrigger() if ^C */
int nwCliDeinitializeConsole(int dopauseonclose); 
int nwCliInitializeConsole(int hidden, int doingmodes); /* return 0 if ok */
int nwCliKbHit(void);
int nwCliGetCh(void);

#ifdef __cplusplus
}
#endif

#endif /* __CNW_CLIENT_CONSOLE_H_ */

