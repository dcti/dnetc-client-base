/*
 * client console management functions.
 * written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * $Id: nwccons.h,v 1.1.2.1 2001/01/21 15:10:29 cyp Exp $
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

