/*
 * Copyright distributed.net 1998-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *core_ogr_cpp(void) {
return "@(#)$Id: core_ogr.cpp,v 1.1.2.11 2003/11/28 00:43:31 snake Exp $"; }

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

#if defined(HAVE_OGR_CORES)

/* ======================================================================== */

/* all the core prototypes
   note: we may have more prototypes here than cores in the client
   note2: if you need some 'cdecl' value define it in selcore.h to CDECL */


#if (CLIENT_CPU == CPU_POWERPC)
    extern "C" CoreDispatchTable *ogr_get_dispatch_table(void);
    #if defined(__VEC__) || defined(__ALTIVEC__) /* compiler supports AltiVec */
    extern "C" CoreDispatchTable *vec_ogr_get_dispatch_table(void);
    #endif
#elif (CLIENT_CPU == CPU_ALPHA)
    extern "C" CoreDispatchTable *ogr_get_dispatch_table(void);
  #if (CLIENT_OS != OS_VMS)    /* Include for other OSes */
    extern "C" CoreDispatchTable *ogr_get_dispatch_table_cix(void);
  #endif
#elif (CLIENT_CPU == CPU_68K)
    extern "C" CoreDispatchTable *ogr_get_dispatch_table_000(void);
    extern "C" CoreDispatchTable *ogr_get_dispatch_table_020(void);
    extern "C" CoreDispatchTable *ogr_get_dispatch_table_030(void);
    extern "C" CoreDispatchTable *ogr_get_dispatch_table_040(void);
    extern "C" CoreDispatchTable *ogr_get_dispatch_table_060(void);
#elif (CLIENT_CPU == CPU_X86)
    extern "C" CoreDispatchTable *ogr_get_dispatch_table(void); //A
    extern "C" CoreDispatchTable *ogr_get_dispatch_table_nobsr(void); //B
#elif (CLIENT_CPU == CPU_ARM)
      extern "C" CoreDispatchTable *ogr_get_dispatch_table_arm1(void);
      extern "C" CoreDispatchTable *ogr_get_dispatch_table_arm2(void);
#else
    extern "C" CoreDispatchTable *ogr_get_dispatch_table(void);
#endif


/* ======================================================================== */


int InitializeCoreTable_ogr(int first_time)
{
  DNETC_UNUSED_PARAM(first_time);

#if defined(HAVE_MULTICRUNCH_VIA_FORK)
  if (first_time) {
    // HACK! for bug #3006
    // call the functions once to initialize the static tables before the client forks
      #if CLIENT_CPU == CPU_X86
        ogr_get_dispatch_table();
        ogr_get_dispatch_table_nobsr();
      #elif CLIENT_CPU == CPU_POWERPC
        ogr_get_dispatch_table();
        #if defined(__VEC__) || defined(__ALTIVEC__) /* compiler supports AltiVec */
          vec_ogr_get_dispatch_table();
        #endif
      #elif (CLIENT_CPU == CPU_68K)
        ogr_get_dispatch_table_000();
        ogr_get_dispatch_table_020();
        ogr_get_dispatch_table_030();
        ogr_get_dispatch_table_040();
        ogr_get_dispatch_table_060();
      #elif (CLIENT_CPU == CPU_ALPHA)
        ogr_get_dispatch_table();
        #if (CLIENT_OS != OS_VMS)         /* Include for other OSes */
           ogr_get_dispatch_table_cix();
        #endif
      #elif (CLIENT_CPU == CPU_VAX)
        ogr_get_dispatch_table();
      #elif (CLIENT_CPU == CPU_SPARC)
        ogr_get_dispatch_table();
      #elif (CLIENT_CPU == CPU_AMD64)
        ogr_get_dispatch_table();
      #else
        #error FIXME! call all your *ogr_get_dispatch_table* functions here once
      #endif
  }
#endif
  return 0;
}


void DeinitializeCoreTable_ogr()
{
  /* ogr does not require any deinitialization */
}

/* ======================================================================== */

const char **corenames_for_contest_ogr()
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
      "GARSP 5.13-A",
      "GARSP 5.13-B",
  #elif (CLIENT_CPU == CPU_AMD64)
      "GARSP 5.13",
  #elif (CLIENT_CPU == CPU_ARM)
      "GARSP 5.13 ARM 1",
      "GARSP 5.13 ARM 2",
  #elif (CLIENT_CPU == CPU_68K)
      "GARSP 5.13 68000",
      "GARSP 5.13 68020",
      "GARSP 5.13 68030",
      "GARSP 5.13 68040",
      "GARSP 5.13 68060",
  #elif (CLIENT_CPU == CPU_ALPHA)
      "GARSP 5.13",
    #if (CLIENT_OS != OS_VMS)  /* Include for other OSes */
      "GARSP 5.13-CIX",
    #endif
  #elif (CLIENT_CPU == CPU_POWERPC)
      "GARSP 5.13 Scalar",
      "GARSP 5.13 Vector",   /* altivec only */
  #elif (CLIENT_CPU == CPU_SPARC)
      "GARSP 5.13",
  #elif (CLIENT_OS == OS_PS2LINUX)
      "GARSP 5.13",
  #else
      "GARSP 5.13",
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
int apply_selcore_substitution_rules_ogr(int cindex)
{
# if (CLIENT_CPU == CPU_ALPHA)
  long det = GetProcessorType(1);
  if (det <  11) cindex = 0;
# elif (CLIENT_CPU == CPU_68K)
  long det = GetProcessorType(1);
  if (det == 68000) cindex = 0;
# elif (CLIENT_CPU == CPU_POWERPC)
  int have_vec = 0;

# if defined(__VEC__) || defined(__ALTIVEC__)
  /* OS+compiler support altivec */
  long det = GetProcessorType(1);
  have_vec = (det >= 0 && (det & 1L<<25)!=0); /* have altivec */
# endif

  if (!have_vec && cindex == 1)     /* PPC-vector */
    cindex = 0;                     /* force PPC-scalar */
#endif

  return cindex;
}

/* -------------------------------------------------------------------- */

int selcoreGetPreselectedCoreForProject_ogr()
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
      if (detected_type >= 11)
        cindex = 1;
      else
        cindex = 0;
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
  #elif (CLIENT_CPU == CPU_POWERPC)
    if (detected_type > 0)
    {
      cindex = 0;                       /* PPC-scalar */

      #if defined(__VEC__) || defined(__ALTIVEC__) /* OS+compiler support altivec */
      if (( detected_type & (1L<<25) ) != 0) //altivec?
      {
        switch ( detected_type & 0xffff) // only compare the low PVR bits
        {
          case 0x000C: cindex = 1; break; // 7400 (G4)   == PPC-vector
          case 0x8000: cindex = 0; break; // 7450 (G4+)  == PPC-scalar
          case 0x8001: cindex = 0; break; // 7455 (G4+)  == PPC-scalar
          case 0x800C: cindex = 1; break; // 7410 (G4)   == PPC-vector
          default:     cindex =-1; break; // no default
        }
      }
      #endif
    }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_X86)
      if (detected_type >= 0)
      {
        switch ( detected_type & 0xff ) // FIXME remove &0xff
        {
          case 0x00: cindex = 1; break; // P5           == without BSR (B)
          case 0x01: cindex = 1; break; // 386/486      == without BSR (B)
          case 0x02: cindex = 0; break; // PII/PIII     == with BSR (A)
          case 0x03: cindex = 0; break; // Cx6x86       == with BSR (A)
          case 0x04: cindex = 1; break; // K5           == without BSR (B)
          #if defined(__GNUC__) || defined(__WATCOMC__) || defined(__BORLANDC__)
          case 0x05: cindex = 1; break; // K6/K6-2/K6-3 == without BSR (B)  #2228
          #elif defined(_MSC_VER)
          case 0x05: cindex = 0; break; // K6/K6-2/K6-3 == with BSR (A)  #2789
          #else
          #warning "FIXME: no OGR core autoselected on a K6 for your compiler"
          #endif
          case 0x06: cindex = 1; break; // Cyrix 486    == without BSR (B)
          case 0x07: cindex = 0; break; // orig Celeron == with BSR (A)
          case 0x08: cindex = 0; break; // PPro         == with BSR (A)
          case 0x09: cindex = 0; break; // AMD K7       == with BSR (A)
          case 0x0A: cindex = 1; break; // Centaur C6   == without BSR (B)
          #if defined(__GNUC__) || defined(__ICC)
          case 0x0B: cindex = 0; break; // Pentium 4    == with BSR (A)
          #elif defined(_MSC_VER) || defined(__WATCOMC__) || defined(__BORLANDC__)
          case 0x0B: cindex = 1; break; // Pentium 4    == without BSR (B)
          #else
          #warning "FIXME: no OGR core autoselected on a P4 for your compiler"
          #endif
          default:   cindex =-1; break; // no default
        }
      }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_ARM)
    {
      extern signed int default_ogr_core;

      cindex = default_ogr_core;
    }
#if 0
    if (detected_type > 0)
    {
      if (detected_type == 0x200  || /* ARM 2 */
          detected_type == 0x250  || /* ARM 250 */
          detected_type == 0x300  || /* ARM 3 */
          detected_type == 0x600  || /* ARM 600 */
          detected_type == 0x610  || /* ARM 610 */
          detected_type == 0x700  || /* ARM 700 */
          detected_type == 0x710  || /* ARM 710 */
          detected_type == 0x7500 || /* ARM 7500 */
          detected_type == 0x7500FE) /* ARM 7500FE */
        cindex = 1;
      else if (detected_type == 0x810 || /* ARM 810 */
               detected_type == 0xA10 || /* StrongARM 110 */
               detected_type == 0xA11 || /* StrongARM 1100 */
               detected_type == 0xB11)   /* StrongARM 1110 */
        cindex = 0;
    }
#endif
  // ===============================================================
  #endif

  return cindex;
}

/* ---------------------------------------------------------------------- */

int selcoreSelectCore_ogr(unsigned int threadindex,
                          int *client_cpuP, struct selcore *selinfo)
{
  int use_generic_proto = 0; /* if rc5/des unit_func proto is generic */
  unit_func_union unit_func; /* declared in problem.h */
  int cruncher_is_asynchronous = 0; /* on a co-processor or similar */
  int pipeline_count = 2; /* most cases */
  int client_cpu = CLIENT_CPU; /* usual case */
  int coresel = selcoreGetSelectedCoreForContest(OGR);

  DNETC_UNUSED_PARAM(threadindex);

  if (coresel < 0)
    return -1;
  memset( &unit_func, 0, sizeof(unit_func));


  /* ================================================================== */

#if (CLIENT_CPU == CPU_POWERPC)
# if defined(__VEC__) || defined(__ALTIVEC__) /* compiler+OS supports AltiVec */
  if (coresel == 1)                           /* "PPC-vector" */
    unit_func.ogr = vec_ogr_get_dispatch_table();
# endif

  if (!unit_func.ogr) {
    unit_func.ogr = ogr_get_dispatch_table(); /* "PPC-scalar" */
    coresel = 0;
  }
#elif (CLIENT_CPU == CPU_68K)
  //extern CoreDispatchTable *ogr_get_dispatch_table_000(void);
  //extern CoreDispatchTable *ogr_get_dispatch_table_020(void);
  //extern CoreDispatchTable *ogr_get_dispatch_table_030(void);
  //extern CoreDispatchTable *ogr_get_dispatch_table_040(void);
  //extern CoreDispatchTable *ogr_get_dispatch_table_060(void);
  if (coresel == 4)
    unit_func.ogr = ogr_get_dispatch_table_060();
  else if (coresel == 3)
    unit_func.ogr = ogr_get_dispatch_table_040();
  else if (coresel == 2)
    unit_func.ogr = ogr_get_dispatch_table_030();
  else if (coresel == 1)
    unit_func.ogr = ogr_get_dispatch_table_020();
  else
  {
    unit_func.ogr = ogr_get_dispatch_table_000();
    coresel = 0;
  }
#elif (CLIENT_CPU == CPU_ALPHA)
  #if (CLIENT_OS != OS_VMS)       /* Include for other OSes */
    if (coresel == 1)       
      unit_func.ogr = ogr_get_dispatch_table_cix();
    else
  #endif 
      unit_func.ogr = ogr_get_dispatch_table();
#elif (CLIENT_CPU == CPU_X86)
  if (coresel == 0) //A
    unit_func.ogr = ogr_get_dispatch_table(); //A
  else
  {
    unit_func.ogr = ogr_get_dispatch_table_nobsr(); //B
    coresel = 1;
  }
#elif (CLIENT_CPU == CPU_AMD64)
  unit_func.ogr = ogr_get_dispatch_table();
  coresel = 0;
#elif (CLIENT_CPU == CPU_ARM)
  if (coresel == 0)
    unit_func.ogr = ogr_get_dispatch_table_arm1();
  else
  {
    unit_func.ogr = ogr_get_dispatch_table_arm2();
    coresel = 1;
  }
#else
  //extern "C" CoreDispatchTable *ogr_get_dispatch_table(void);
  unit_func.ogr = ogr_get_dispatch_table();
  coresel = 0;
#endif

  /* ================================================================== */


  if (coresel >= 0 && unit_func.gen &&
     coresel < ((int)corecount_for_contest(OGR)) )
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
