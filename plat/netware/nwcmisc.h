/*
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * misc stuff that gets called from client/common code.
 * written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * $Id: nwcmisc.h,v 1.2 2002/09/02 00:35:51 andreasb Exp $
*/

#ifndef __CNW_CLIENT_MISC_H_
#define __CNW_CLIENT_MISC_H_ 

#ifdef __cplusplus
extern "C" {
#endif

extern int nwCliInitClient( int argc, char **argv );
extern int nwCliExitClient( void );
extern const char *nwCliGetNLMBaseName( void );
extern void nwCliMillisecSleep( unsigned long millisecs );

#ifdef __cplusplus
}
#endif
#endif /* __CNW_CLIENT_MISC_H_ */
