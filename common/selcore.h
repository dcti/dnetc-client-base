/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __SELCORE_H__
#define __SELCORE_H__ "@(#)$Id: selcore.h,v 1.16.2.5 2003/09/01 19:27:36 jlawson Exp $"

#include "cputypes.h"
#include "ccoreio.h"
#if defined(HAVE_OGR_CORES)
#include "ogr.h"
#endif



/* ---------------------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif

// Definitions of core prototypes for each project.
typedef s32 gen_func( RC5UnitWork *, u32 *, void * );
typedef u32 CDECL rc5_func( RC5UnitWork *, u32 );
typedef u32 des_func( RC5UnitWork *, u32 *, char * );
#if defined(HAVE_OGR_CORES)
typedef CoreDispatchTable *ogr_func;
#endif
typedef s32 CDECL gen_72_func( RC5_72UnitWork *, u32 *, void * );


typedef union
{
  /* generic prototype: RC5-64, DES, CSC */
  s32 (*gen)( RC5UnitWork *, u32 *iterations, void *memblk );

  /* old style: RC5-64, DES */
  u32 CDECL (*rc5)( RC5UnitWork *, u32 iterations );
  #if defined(HAVE_DES_CORES)
  u32 (*des)( RC5UnitWork *, u32 *iterations, char *membuf );
  #endif

  /* OGR */
  #if defined(HAVE_OGR_CORES)
  CoreDispatchTable *ogr;
  #endif

  /* generic prototype: RC5-72 */
  s32 CDECL (*gen_72)( RC5_72UnitWork *, u32 *iterations, void *memblk );

  #if 0
  PROJECT_NOT_HANDLED("in unit_func_union");
  #endif
} unit_func_union;

#ifdef __cplusplus
}
#endif


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
const char **corenames_for_contest( unsigned int cont_i );
unsigned int corecount_for_contest( unsigned int cont_i );

/* conf calls these */
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

/* ---------------------------------------------------------------------- */

/* The following are all for internal use by selcore.cpp functions,
 * but are the basic functions defined by each of the project core
 * files.  All of these functions are available generically through
 * via one of the functions above that take a projectid as an
 * argument.
 */

#ifdef HAVE_RC5_64_CORES
int InitializeCoreTable_rc564(int first_time);

void DeinitializeCoreTable_rc564();

const char **corenames_for_contest_rc564();

int apply_selcore_substitution_rules_rc564(int cindex);

int selcoreGetPreselectedCoreForProject_rc564();

int selcoreSelectCore_rc564( unsigned int threadindex,
                             int *client_cpuP, struct selcore *selinfo );
#endif
#ifdef HAVE_RC5_72_CORES
int InitializeCoreTable_rc572(int first_time);

void DeinitializeCoreTable_rc572();

const char **corenames_for_contest_rc572();

int apply_selcore_substitution_rules_rc572(int cindex);

int selcoreGetPreselectedCoreForProject_rc572();

int selcoreSelectCore_rc572( unsigned int threadindex,
                             int *client_cpuP, struct selcore *selinfo );
#endif
#ifdef HAVE_CSC_CORES
int InitializeCoreTable_csc(int first_time);

void DeinitializeCoreTable_csc();

const char **corenames_for_contest_csc();

int apply_selcore_substitution_rules_csc(int cindex);

int selcoreGetPreselectedCoreForProject_csc();

int selcoreSelectCore_csc( unsigned int threadindex,
                           int *client_cpuP, struct selcore *selinfo );
#endif
#ifdef HAVE_DES_CORES
int InitializeCoreTable_des(int first_time);

void DeinitializeCoreTable_des();

const char **corenames_for_contest_des();

int apply_selcore_substitution_rules_des(int cindex);

int selcoreGetPreselectedCoreForProject_des();

int selcoreSelectCore_des(unsigned int threadindex,
                          int *client_cpuP, struct selcore *selinfo );
#endif
#ifdef HAVE_OGR_CORES
int InitializeCoreTable_ogr(int first_time);

void DeinitializeCoreTable_ogr();

const char **corenames_for_contest_ogr();

int apply_selcore_substitution_rules_ogr(int cindex);

int selcoreGetPreselectedCoreForProject_ogr();

int selcoreSelectCore_ogr( unsigned int threadindex,
                           int *client_cpuP, struct selcore *selinfo );
#endif



/* ---------------------------------------------------------------------- */

#endif /* __SELCORE_H__ */
