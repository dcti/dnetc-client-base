/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __SELCORE_H__
#define __SELCORE_H__ "@(#)$Id: selcore.h,v 1.3.2.3 1999/10/10 23:28:00 cyp Exp $"

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
                       

/* ClientMain() calls these */
int InitializeCoreTable( int *coretypes );
int DeinitializeCoreTable( void );

#endif /* __SELCORE_H__ */
