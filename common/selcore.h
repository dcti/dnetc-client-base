/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __SELCORE_H__
#define __SELCORE_H__ "@(#)$Id: selcore.h,v 1.3.2.9 2001/01/03 23:21:54 cyp Exp $"

#if defined(__PROBLEM_H__)
/* Set the xx_unit_func vectors/cputype/coresel in the problem. */
/* Returns core # or <0 if error. Called from Prob::LoadState and probfill */
int selcoreSelectCore( unsigned int cont_id, unsigned int thrindex, 
                       int *client_cpuP, struct selcore *selinfo );
#endif                       

/* Get the core # for a contest. Informational use only. */
int selcoreGetSelectedCoreForContest( unsigned int contestid );
const char *selcoreGetDisplayName( unsigned int cont_i, int index );

/* conf calles these */
int selcoreValidateCoreIndex( unsigned int cont_i, int index );
void selcoreEnumerate( int (*enumcoresproc)(unsigned int cont, 
                              const char *corename, int idx, void *udata ),
                       void *userdata );
void selcoreEnumerateWide( int (*enumcoresproc)(
                              const char **corenames, int idx, void *udata ),
                           void *userdata );

/* benchmark/test each core - return < 0 on error, 0 = not supported, > 0=ok */
long selcoreBenchmark( unsigned int cont_i, unsigned int secs );
long selcoreSelfTest( unsigned int cont_i );

/* ClientMain() calls these */
int InitializeCoreTable( int *coretypes );
int DeinitializeCoreTable( void );

#endif /* __SELCORE_H__ */
