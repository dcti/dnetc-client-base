/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __SELCORE_H__
#define __SELCORE_H__ "@(#)$Id: selcore.h,v 1.3.2.11 2001/03/06 04:35:43 sampo Exp $"

#include "cputypes.h"
#include "ccoreio.h"
#if defined(HAVE_OGR_CORES)
#include "ogr.h"
#endif

typedef union
{
    /* this is our generic prototype */
    s32 (*gen)( RC5UnitWork *, u32 *iterations, void *memblk );
    #if (CLIENT_OS == OS_AMIGAOS) && (CLIENT_CPU == CPU_68K)
    u32 __regargs (*rc5)( RC5UnitWork * , u32 iterations );
    #else
    u32 (*rc5)( RC5UnitWork * , u32 iterations );
    #endif
    #if defined(HAVE_DES_CORES)
    u32 (*des)( RC5UnitWork * , u32 *iterations, char *membuf );
    #endif
    #if defined(HAVE_OGR_CORES)
    CoreDispatchTable *ogr;
    #endif
} unit_func_union;

struct selcore
{
  int client_cpu;
  int pipeline_count;
  int use_generic_proto;
  int cruncher_is_asynchronous;
  unit_func_union unit_func;
};

/* ---------------------------------------------------------------------- */

/* Set the xx_unit_func vectors/cputype/coresel in the problem. */
/* Returns core # or <0 if error. Called from Prob::LoadState and probfill */
int selcoreSelectCore( unsigned int cont_id, unsigned int thrindex, 
                       int *client_cpuP, struct selcore *selinfo );

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
long selcoreBenchmark( unsigned int cont_i, unsigned int secs, int corenum );
long selcoreSelfTest( unsigned int cont_i, int corenum );

/* ClientMain() calls these */
int InitializeCoreTable( int *coretypes );
int DeinitializeCoreTable( void );

#endif /* __SELCORE_H__ */
