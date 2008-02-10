/*
 * Copyright distributed.net 1998-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *core_ogr_ng_cpp(void) {
return "@(#)$Id: core_ogr_ng.cpp,v 1.1 2008/02/10 00:11:13 kakace Exp $"; }

//#define TRACE

#include "cputypes.h"
#include "client.h"    // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "problem.h"   // problem class
#include "cpucheck.h"  // cpu selection, GetTimesliceBaseline()
#include "logstuff.h"  // Log()/LogScreen()
#include "clicdata.h"  // GetContestNameFromID()
#include "bench.h"     // TBenchmark()
#include "selftest.h"  // SelfTest()
#include "selcore.h"   // keep prototypes in sync
#include "probman.h"   // GetManagedProblemCount()
#include "triggers.h"  // CheckExitRequestTriggerNoIO()
#include "util.h"      // TRACE_OUT, DNETC_UNUSED_*
#include "ogr.h"

#if (CLIENT_CPU == CPU_CELLBE)
#include <libspe2.h>
#endif

#if defined(HAVE_OGR_CORES)

/* ======================================================================== */

/* all the core prototypes
   note: we may have more prototypes here than cores in the client
   note2: if you need some 'cdecl' value define it in selcore.h to CDECL */

extern "C" CoreDispatchTable *ogr_ng_get_dispatch_table(void);


/* ======================================================================== */

int InitializeCoreTable_ogr_ng(int first_time)
{
  DNETC_UNUSED_PARAM(first_time);

#if defined(HAVE_MULTICRUNCH_VIA_FORK)
  if (first_time) {
    ogr_ng_get_dispatch_table();
  }
#endif
  return ogr_init_choose();
}


void DeinitializeCoreTable_ogr_ng()
{
   (void) ogr_cleanup_choose();
}


/* ======================================================================== */

const char **corenames_for_contest_ogr_ng()
{
  /*
   When selecting corenames, use names that describe how (what optimization)
   they are different from their predecessor(s). If only one core,
   use the obvious "MIPS optimized" or similar.
  */
  static const char *corenames_table[] =
  {
      "FLEGE 2.0",
      NULL
  };

  return corenames_table;
}

/* -------------------------------------------------------------------- */

/*
** Apply substition according to the same rules enforced by
** selcoreSelectCore() [ie, return the cindex of the core actually used
** after applying appropriate OS/architecture/#define limitations to
** ensure the client doesn't crash]
**
** This is necessary when the list of cores is a superset of the
** cores supported by a particular build. For example, all x86 clients
** display the same core list for RC5, but as not all cores may be
** available in a particular client/build/environment, this function maps
** between the ones that aren't available to the next best ones that are.
**
** Note that we intentionally don't do very intensive validation here. Thats
** selcoreGetSelectedCoreForContest()'s job when the user chooses to let
** the client auto-select. If the user has explicitely specified a core #,
** they have to live with the possibility that the choice will at some point
** no longer be optimal.
*/
int apply_selcore_substitution_rules_ogr_ng(int cindex)
{
  return 0;
}

/* -------------------------------------------------------------------- */

int selcoreGetPreselectedCoreForProject_ogr_ng()
{
  return 0;
}

/* ---------------------------------------------------------------------- */

int selcoreSelectCore_ogr_ng(unsigned int threadindex, int *client_cpuP,
                struct selcore *selinfo, unsigned int contestid)
{
  int use_generic_proto = 0; /* if rc5/des unit_func proto is generic */
  unit_func_union unit_func; /* declared in problem.h */
  int cruncher_is_asynchronous = 0; /* on a co-processor or similar */
  int pipeline_count = 2; /* most cases */
  int client_cpu = CLIENT_CPU; /* usual case */
  int coresel = selcoreGetSelectedCoreForContest(contestid);
  

  if (coresel < 0)
    return -1;

  memset( &unit_func, 0, sizeof(unit_func));


  /* ================================================================== */

  unit_func.ogr = ogr_ng_get_dispatch_table();

  /* ================================================================== */


  if (coresel >= 0 && unit_func.ogr &&
     coresel < ((int)corecount_for_contest(contestid)) )
  {
    if (client_cpuP)
      *client_cpuP = client_cpu;
    if (selinfo)
    {
      selinfo->client_cpu = client_cpu;
      selinfo->pipeline_count = pipeline_count;
      selinfo->use_generic_proto = use_generic_proto;
      selinfo->cruncher_is_asynchronous = cruncher_is_asynchronous;
      memcpy( (void *)&(selinfo->unit_func), &unit_func, sizeof(unit_func));
    }
    return coresel;
  }

  return -1; /* core selection failed */
}

/* ------------------------------------------------------------- */

#endif // defined(HAVE_OGR_CORES)
