/*
 * Copyright distributed.net 1998-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *core_ogr_ng_cpp(void) {
return "@(#)$Id: core_ogr_ng.cpp,v 1.6 2008/10/27 10:14:11 oliver Exp $"; }

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

#if (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_CELLBE)
    CoreDispatchTable *ogrng_get_dispatch_table(void);
    #if defined(HAVE_I64) && !defined(HAVE_FLEGE_PPC_CORES)
    CoreDispatchTable *ogrng64_get_dispatch_table(void);
    #endif
    #if defined(__VEC__) || defined(__ALTIVEC__) /* compiler supports AltiVec */
    CoreDispatchTable *vec_ogrng_get_dispatch_table(void);
    #endif
    #if (CLIENT_CPU == CPU_CELLBE)
    CoreDispatchTable *spe_ogrng_get_dispatch_table(void);
    #endif
#elif (CLIENT_CPU == CPU_ALPHA)
    CoreDispatchTable *ogrng_get_dispatch_table(void);
  #if (CLIENT_OS != OS_VMS)    /* Include for other OSes */
//    CoreDispatchTable *ogrng_get_dispatch_table(void);
  #endif
    CoreDispatchTable *ogrng64_get_dispatch_table(void);
  #if (CLIENT_OS != OS_VMS)    /* Include for other OSes */
//    CoreDispatchTable *ogrng64_get_dispatch_table(void);
  #endif
#elif (CLIENT_CPU == CPU_68K)
    CoreDispatchTable *ogrng_get_dispatch_table_000(void);
    CoreDispatchTable *ogrng_get_dispatch_table_020(void);
    CoreDispatchTable *ogrng_get_dispatch_table_030(void);
    CoreDispatchTable *ogrng_get_dispatch_table_040(void);
    CoreDispatchTable *ogrng_get_dispatch_table_060(void);
#elif (CLIENT_CPU == CPU_X86)
    CoreDispatchTable *ogrng_get_dispatch_table(void); //A
    #if defined(HAVE_I64) && (SIZEOF_LONG == 8)
      CoreDispatchTable *ogrng64_get_dispatch_table(void);
    #else
      CoreDispatchTable *ogrng_get_dispatch_table_asm1(void); //B (asm #1)
    #endif
#elif (CLIENT_CPU == CPU_ARM)
    CoreDispatchTable *ogrng_get_dispatch_table(void);
#elif (CLIENT_CPU == CPU_AMD64)
    CoreDispatchTable *ogrng64_get_dispatch_table(void);
#elif (CLIENT_CPU == CPU_SPARC) && (SIZEOF_LONG == 8)
    CoreDispatchTable *ogrng64_get_dispatch_table(void); 
#elif (CLIENT_CPU == CPU_MIPS) && (SIZEOF_LONG == 8)
    CoreDispatchTable *ogrng64_get_dispatch_table(void); 
#else
    CoreDispatchTable *ogrng_get_dispatch_table(void);
#endif


/* ======================================================================== */

int InitializeCoreTable_ogr_ng(int first_time)
{
  DNETC_UNUSED_PARAM(first_time);

#if defined(HAVE_MULTICRUNCH_VIA_FORK)
  if (first_time) {
    // HACK! for bug #3006
    // call the functions once to initialize the static tables before the client forks
      #if CLIENT_CPU == CPU_X86
        ogrng_get_dispatch_table();
        #if defined(HAVE_I64) && (SIZEOF_LONG == 8)
          ogrng64_get_dispatch_table();
        #else
          ogrng_get_dispatch_table_asm1();
        #endif
      #elif (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_CELLBE)
        ogrng_get_dispatch_table();
        #if defined(HAVE_I64) && !defined(HAVE_FLEGE_PPC_CORES)
          ogrng64_get_dispatch_table();
        #endif
        #if defined(__VEC__) || defined(__ALTIVEC__) /* compiler supports AltiVec */
          vec_ogrng_get_dispatch_table();
        #endif
        #if (CLIENT_CPU == CPU_CELLBE)
          spe_ogrng_get_dispatch_table();
        #endif
      #elif (CLIENT_CPU == CPU_68K)
        ogrng_get_dispatch_table_000();
        ogrng_get_dispatch_table_020();
        ogrng_get_dispatch_table_030();
        ogrng_get_dispatch_table_040();
        ogrng_get_dispatch_table_060();
      #elif (CLIENT_CPU == CPU_ALPHA)
        ogrng_get_dispatch_table();
        #if (CLIENT_OS != OS_VMS)         /* Include for other OSes */
//        ogrng_get_dispatch_table();
        #endif
        ogrng64_get_dispatch_table();
        #if (CLIENT_OS != OS_VMS)         /* Include for other OSes */
//        ogrng64_get_dispatch_table();
        #endif
      #elif (CLIENT_CPU == CPU_VAX)
        ogrng_get_dispatch_table();
      #elif (CLIENT_CPU == CPU_SPARC)
        #if (SIZEOF_LONG == 8)
          ogrng64_get_dispatch_table
        #else
          ogrng_get_dispatch_table();
        #endif
      #elif (CLIENT_CPU == CPU_AMD64)
        //ogr_get_dispatch_table();
        ogrng64_get_dispatch_table();
      #elif (CLIENT_CPU == CPU_S390)
        ogrng_get_dispatch_table();
      #elif (CLIENT_CPU == CPU_S390X)
        ogrng_get_dispatch_table();
      #elif (CLIENT_CPU == CPU_I64)
        ogrng_get_dispatch_table();
      #else
        #error FIXME! call all your *ogr_get_dispatch_table* functions here once
      #endif
  }
#endif

  return ogr_init_choose();
}


void DeinitializeCoreTable_ogr_ng()
{
   ogr_cleanup_choose();
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
  /* ================================================================== */
  #if (CLIENT_CPU == CPU_X86)
      "FLEGE 2.0",
      #if defined(HAVE_I64) && (SIZEOF_LONG == 8)
      "FLEGE-64 2.0",
      #else
      "rt-asm-generic",
      #endif
  #elif (CLIENT_CPU == CPU_AMD64)
      "FLEGE-64 2.0",
  #elif (CLIENT_CPU == CPU_ARM)
      "FLEGE 2.0",
  #elif (CLIENT_CPU == CPU_68K)
      "FLEGE 2.0 68000",
      "FLEGE 2.0 68020",
      "FLEGE 2.0 68030",
      "FLEGE 2.0 68040",
      "FLEGE 2.0 68060",
  #elif (CLIENT_CPU == CPU_ALPHA)
      "FLEGE 2.0",
    #if (CLIENT_OS != OS_VMS)  /* Include for other OSes */
//      "FLEGE 2.0-CIX",
    #endif
      "FLEGE-64 2.0",
    #if (CLIENT_OS != OS_VMS)  /* Include for other OSes */
//      "FLEGE-64 2.0-CIX",
    #endif
  #elif (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_CELLBE)
    #ifdef HAVE_FLEGE_PPC_CORES
      /* Optimized ASM cores */
      "KOGE 3.0 Scalar",
      "KOGE 3.0 Hybrid",            /* altivec only */
    #else
      "FLEGE 2.0 Scalar-32",
      "FLEGE 2.0 Hybrid",           /* altivec only */
      #ifdef HAVE_I64
      "FLEGE-64 2.0",
      #endif
    #endif
    #if (CLIENT_CPU == CPU_CELLBE)
      "Cell v2 SPE (base)",
    #endif
  #elif (CLIENT_CPU == CPU_SPARC) && (SIZEOF_LONG == 8)
      "FLEGE-64 2.0",
  #elif (CLIENT_CPU == CPU_SPARC)
      "FLEGE 2.0",
  #elif (CLIENT_CPU == CPU_MIPS) && (SIZEOF_LONG == 8)
      "FLEGE-64 2.0",
  #elif (CLIENT_OS == OS_PS2LINUX)
      "FLEGE 2.0",
  #else
      "FLEGE 2.0",
  #endif
  /* ================================================================== */
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
# if (CLIENT_CPU == CPU_ALPHA)
  long det = GetProcessorType(1);
  if ((det <  11) && (cindex == 1)) {
    cindex = 0;
  }
# elif (CLIENT_CPU == CPU_68K)
  long det = GetProcessorType(1);
  if (det == 68000) cindex = 0;
# elif (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_CELLBE)

  int feature = 0;
  feature = GetProcessorFeatureFlags();
  if ((feature & CPU_F_ALTIVEC) == 0 && cindex == 1)      /* PPC-vector */
    cindex = 0;                                     /* force PPC-scalar */
#if !defined(HAVE_FLEGE_PPC_CORES) && defined(HAVE_I64)   /* 64-bit cores listed only in this case */
  if ((feature & CPU_F_64BITOPS) == 0 && cindex == 2)     /* PPC-64bit  */
    cindex = 0;                                     /* force PPC-32bit  */
#endif

# elif (CLIENT_CPU == CPU_X86)
#  if !defined(HAVE_I64) || (SIZEOF_LONG < 8)     /* no 64-bit support? */
#  endif
#endif
  return cindex;
}

/* -------------------------------------------------------------------- */

int selcoreGetPreselectedCoreForProject_ogr_ng()
{
  static long detected_type = -123;
  static unsigned long detected_flags = 0;
  int cindex = -1;

  if (detected_type == -123) /* haven't autodetected yet? */
  {
    detected_type = GetProcessorType(1 /* quietly */);
    if (detected_type < 0)
      detected_type = -1;
    detected_flags = GetProcessorFeatureFlags();
  }

  // you may add your pre-selected core depending on arch
  // and cpu here, but leaving the defaults (runs micro-benchmark) is ok

  // ===============================================================
  #if (CLIENT_CPU == CPU_ALPHA)
    if (detected_type > 0)
    {
        cindex = -1;
    }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_68K)
    if (detected_type > 0)
    {
      if (detected_type >= 68060)
        cindex = 4;
      else if (detected_type == 68040)
        cindex = 3;
      else if (detected_type == 68030)
        cindex = 2;
      else if (detected_type == 68020)
        cindex = 1;
      else
        cindex = 0;
    }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_POWER)
    cindex = 0;                         /* only one OGR core on Power */
  #elif (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_CELLBE)
    if (detected_type > 0)
    {
      cindex = 0;                       /* PPC-scalar */

      #if defined(__VEC__) || defined(__ALTIVEC__) /* OS+compiler support altivec */
      if ((detected_flags & CPU_F_ALTIVEC) != 0) //altivec?
      {
        cindex = 1;     // PPC-vector
      }
      #endif
    }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_X86)
      if (detected_type >= 0)
      {
      #if defined(HAVE_I64) && (SIZEOF_LONG == 8) // Need native 64-bit support
        cindex = 1;  /* 64-bit core */
      #else
        cindex = 1;  /* generic asm core */
      #endif
      }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_ARM)
    {
      extern signed int default_ogr_core;

      cindex = default_ogr_core;
    }
  // ===============================================================
  #endif

  return cindex;
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

#if (CLIENT_CPU == CPU_CELLBE)
  // Each Cell has 1 PPE, which is dual-threaded (so in fact the OS sees 2
  // processors), and although we should run 2 threads at a time, the way
  // CPUs are detected precludes us from doing that.
  static unsigned int PPE_count = spe_cpu_info_get(SPE_COUNT_PHYSICAL_CPU_NODES, -1);

  // Threads with threadindex = 0..PPE_count-1 will be scheduled on the PPEs;
  // the rest are scheduled on the SPEs.
  if (threadindex >= PPE_count)
    coresel = 2;
#else
  DNETC_UNUSED_PARAM(threadindex);
#endif


  if (coresel < 0)
    return -1;

  memset( &unit_func, 0, sizeof(unit_func));


  /* ================================================================== */

#if (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_CELLBE)
  #if defined(__VEC__) || defined(__ALTIVEC__) /* compiler+OS supports AltiVec */
  if (coresel == 1)                               /* PPC Vector/Hybrid */
     unit_func.ogr = vec_ogrng_get_dispatch_table();
  #endif

  #if (CLIENT_CPU == CPU_CELLBE)
  if (coresel == 2)
    unit_func.ogr = spe_ogrng_get_dispatch_table();
  #endif

  #if defined(HAVE_I64) && !defined(HAVE_FLEGE_PPC_CORES)
  if (coresel == 2)
    unit_func.ogr = ogrng64_get_dispatch_table();   /* PPC Scalar-64 */
  #endif

  if (!unit_func.ogr) {
    unit_func.ogr = ogrng_get_dispatch_table();     /* PPC Scalar-32 */
    coresel = 0;
  }
#elif (CLIENT_CPU == CPU_68K)
  if (coresel == 4)
    unit_func.ogr = ogrng_get_dispatch_table_060();
  else if (coresel == 3)
    unit_func.ogr = ogrng_get_dispatch_table_040();
  else if (coresel == 2)
    unit_func.ogr = ogrng_get_dispatch_table_030();
  else if (coresel == 1)
    unit_func.ogr = ogrng_get_dispatch_table_020();
  else
  {
    unit_func.ogr = ogrng_get_dispatch_table_000();
    coresel = 0;
  }
#elif (CLIENT_CPU == CPU_ALPHA)
  #if (CLIENT_OS != OS_VMS)       /* Include for other OSes */
//    if (coresel == 1)       
//      unit_func.ogr = ogr_get_dispatch_table_cix();
//    else
  #endif 
    if (coresel == 1)
      unit_func.ogr = ogrng64_get_dispatch_table();
    else
  #if (CLIENT_OS != OS_VMS)       /* Include for other OSes */
//    if (coresel == 5)
//      unit_func.ogr = ogr_get_dispatch_table_cix_64();
//    else
  #endif
      unit_func.ogr = ogrng_get_dispatch_table();
#elif (CLIENT_CPU == CPU_X86)
  #if defined(HAVE_I64) && (SIZEOF_LONG == 8)
    if (coresel == 1)
      unit_func.ogr = ogrng64_get_dispatch_table();
    else
      unit_func.ogr = ogrng_get_dispatch_table();
  #else
    if (coresel == 1)
      unit_func.ogr = ogrng_get_dispatch_table_asm1();
    else
      unit_func.ogr = ogrng_get_dispatch_table();
  #endif
#elif (CLIENT_CPU == CPU_AMD64)
  unit_func.ogr = ogrng64_get_dispatch_table();
  coresel = 0;
#elif (CLIENT_CPU == CPU_ARM)
  unit_func.ogr = ogrng_get_dispatch_table();
#elif (CLIENT_CPU == CPU_SPARC) && (SIZEOF_LONG == 8)
  unit_func.ogr = ogrng64_get_dispatch_table();
  coresel = 0;
#elif (CLIENT_CPU == CPU_MIPS) && (SIZEOF_LONG == 8)
  unit_func.ogr = ogrng64_get_dispatch_table();
  coresel = 0;
#else
  unit_func.ogr = ogrng_get_dispatch_table();
  coresel = 0;
#endif

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