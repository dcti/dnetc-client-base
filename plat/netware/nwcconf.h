/*
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * misc (unsorted) stuff that gets called from client/common code.
 * written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * $Id: nwcconf.h,v 1.1.2.2 2002/04/12 23:56:49 andreasb Exp $
*/

#ifndef __CNW_CONF_H_
#define __CNW_CONF_H_ 

#ifdef __cplusplus
extern "C" {
#endif

int nwCliGetPollingAllowedFlag(void);
int nwCliAreCrunchersRestartable(void);
int nwCliIsPreemptiveEnv( void );
int nwCliGetUtilizationSupressionFlag(void);

void nwCliLoadSettings(const char *inifile);

#ifdef __cplusplus
}
#endif
#endif /* __CNW_CONF_H_ */
