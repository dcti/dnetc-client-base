/*
 * misc stuff that gets called from client/common code.
 * written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * $Id: nwcmisc.h,v 1.1.2.1 2001/01/21 15:10:29 cyp Exp $
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
