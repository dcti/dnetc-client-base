/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __SELCORE_H__
#define __SELCORE_H__ "@(#)$Id: selcore.h,v 1.5 1999/11/08 02:02:44 cyp Exp $"

/* this is called from Problem::LoadState() */
int selcoreGetSelectedCoreForContest( unsigned int contestid );

/* conf calles these */
int selcoreValidateCoreIndex( unsigned int cont_i, int index );
void selcoreEnumerate( int (*proc)(unsigned int cont, 
                            const char *corename, int idx, void *udata ),
                       void *userdata );
void selcoreEnumerateWide( int (*proc)(
                            const char **corenames, int idx, void *udata ),
                       void *userdata );
/* benchmark/test each core */
int selcoreBenchmark( unsigned int cont_i, unsigned int secs );
int selcoreSelfTest( unsigned int cont_i );

/* ClientMain() calls these */
int InitializeCoreTable( int *coretypes );
int DeinitializeCoreTable( void );

#endif /* __SELCORE_H__ */
