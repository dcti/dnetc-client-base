/*
 * Copyright distributed.net 1998-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *core_ogr_cpp(void) {
return "@(#)$Id: core_ogr.cpp,v 1.1.2.49 2006/10/01 15:35:38 snikkel Exp $"; }

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

#if defined(HAVE_OGR_CORES) || defined(HAVE_OGR_PASS2)

/* ======================================================================== */

/* all the core prototypes
   note: we may have more prototypes here than cores in the client
   note2: if you need some 'cdecl' value define it in selcore.h to CDECL */

#if (CLIENT_CPU == CPU_POWERPC)
    extern "C" CoreDispatchTable *ogr_get_dispatch_table(void);
    #if defined(HAVE_I64) && !defined(HAVE_KOGE_PPC_CORES)
    /* KOGE cores are faster. The 64-bit core is only enabled when compiling
    ** the client with GARSP 6.0 cores (ANSI).
    */
    extern "C" CoreDispatchTable *ogr64_get_dispatch_table(void);
    #endif
    #if defined(__VEC__) || defined(__ALTIVEC__) /* compiler supports AltiVec */
    extern "C" CoreDispatchTable *vec_ogr_get_dispatch_table(void);
    #endif
#elif (CLIENT_CPU == CPU_ALPHA)
    extern "C" CoreDispatchTable *ogr_get_dispatch_table(void);
  #if (CLIENT_OS != OS_VMS)    /* Include for other OSes */
    extern "C" CoreDispatchTable *ogr_get_dispatch_table_cix(void);
  #endif
    extern "C" CoreDispatchTable *ogr_get_dispatch_table_ev4(void);
    extern "C" CoreDispatchTable *ogr64_get_dispatch_table(void);
    extern "C" CoreDispatchTable *ogr_get_dispatch_table_ev4_64(void);
  #if (CLIENT_OS != OS_VMS)    /* Include for other OSes */
    extern "C" CoreDispatchTable *ogr_get_dispatch_table_cix_64(void);
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
    extern "C" CoreDispatchTable *ogr_get_dispatch_table_asm_gen(void);
    #if defined(HAVE_I64)
    extern "C" CoreDispatchTable *ogr_get_dispatch_table_asm_mmx(void);
    extern "C" CoreDispatchTable *ogr_get_dispatch_table_asm_mmx_amd(void);
    #endif
#elif (CLIENT_CPU == CPU_ARM)
    extern "C" CoreDispatchTable *ogr_get_dispatch_table_arm1(void);
    extern "C" CoreDispatchTable *ogr_get_dispatch_table_arm2(void);
    extern "C" CoreDispatchTable *ogr_get_dispatch_table_arm3(void);
#elif (CLIENT_CPU == CPU_AMD64)
    extern "C" CoreDispatchTable *ogr64_get_dispatch_table(void);
#elif (CLIENT_CPU == CPU_SPARC) && (SIZEOF_LONG == 8)
    extern "C" CoreDispatchTable *ogr64_get_dispatch_table(void); 
#elif (CLIENT_CPU == CPU_MIPS) && (SIZEOF_LONG == 8)
    extern "C" CoreDispatchTable *ogr64_get_dispatch_table(void); 
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
        ogr_get_dispatch_table_asm_gen();
        #if defined(HAVE_I64)
          ogr_get_dispatch_table_asm_mmx();
          ogr_get_dispatch_table_asm_mmx_amd();
        #endif
      #elif CLIENT_CPU == CPU_POWERPC
        ogr_get_dispatch_table();
        #if defined(HAVE_I64) && !defined(HAVE_KOGE_PPC_CORES)
          ogr64_get_dispatch_table();
        #endif
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
        ogr_get_dispatch_table_ev4();
        ogr64_get_dispatch_table();
        ogr_get_dispatch_table_ev4_64();
        #if (CLIENT_OS != OS_VMS)         /* Include for other OSes */
        ogr_get_dispatch_table_cix_64();
        #endif
      #elif (CLIENT_CPU == CPU_VAX)
        ogr_get_dispatch_table();
      #elif (CLIENT_CPU == CPU_SPARC)
        #if (SIZEOF_LONG == 8)
          ogr64_get_dispatch_table
        #else
          ogr_get_dispatch_table();
        #endif
      #elif (CLIENT_CPU == CPU_AMD64)
        //ogr_get_dispatch_table();
        ogr64_get_dispatch_table();
      #elif (CLIENT_CPU == CPU_S390)
        ogr_get_dispatch_table();
      #elif (CLIENT_CPU == CPU_S390X)
        ogr_get_dispatch_table();
      #elif (CLIENT_CPU == CPU_I64)
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
      "GARSP 6.0-A",
      "GARSP 6.0-B",
      "GARSP 6.0-asm-rt1-gen",
      #ifdef HAVE_I64
      "GARSP 6.0-asm-rt1-mmx",
      "GARSP 6.0-asm-rt1-mmx-amd",
      #endif
  #elif (CLIENT_CPU == CPU_AMD64)
      "GARSP 6.0-64",
  #elif (CLIENT_CPU == CPU_ARM)
      "GARSP 6.0 StrongARM",
      "GARSP 6.0 ARM 2/3/6/7",
      "GARSP 6.0 XScale",
  #elif (CLIENT_CPU == CPU_68K)
      "GARSP 6.0 68000",
      "GARSP 6.0 68020",
      "GARSP 6.0 68030",
      "GARSP 6.0 68040",
      "GARSP 6.0 68060",
  #elif (CLIENT_CPU == CPU_ALPHA)
      "GARSP 6.0",
    #if (CLIENT_OS != OS_VMS)  /* Include for other OSes */
      "GARSP 6.0-CIX",
    #endif
      "GARSP 6.0-EV4",
      "GARSP 6.0-64",
      "GARSP 6.0-EV4-64",
    #if (CLIENT_OS != OS_VMS)  /* Include for other OSes */
      "GARSP 6.0-CIX-64",
    #endif
  #elif (CLIENT_CPU == CPU_POWERPC)
    #ifdef HAVE_KOGE_PPC_CORES
      /* Optimized ASM cores */
      "KOGE 2.0 Scalar",            /* KOGE : Kakace's Optimized Garsp Engine */
      "KOGE 2.0 Hybrid",            /* altivec only */
    #else
      "GARSP 6.0 Scalar-32",
      "GARSP 6.0 Hybrid",           /* altivec only */
      #ifdef HAVE_I64
      "GARSP 6.0 Scalar-64",        /* 64-bit core */
      #endif
    #endif
  #elif (CLIENT_CPU == CPU_SPARC) && (SIZEOF_LONG == 8)
      "GARSP 6.0-64",
  #elif (CLIENT_CPU == CPU_SPARC)
      "GARSP 6.0",
  #elif (CLIENT_CPU == CPU_MIPS) && (SIZEOF_LONG == 8)
      "GARSP 6.0-64",
  #elif (CLIENT_OS == OS_PS2LINUX)
      "GARSP 6.0",
  #else
      "GARSP 6.0",
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
  if ((det <  11) && (cindex == 1)) cindex = 0;
  if ((det <  11) && (cindex == 5)) cindex = 3;
# elif (CLIENT_CPU == CPU_68K)
  long det = GetProcessorType(1);
  if (det == 68000) cindex = 0;
# elif (CLIENT_CPU == CPU_ARM)
  long det = GetProcessorType(1);
  int have_clz = (((det >> 12) & 0xf) != 0) &&
                 (((det >> 12) & 0xf) != 7) &&
                 (((det >> 16) & 0xf) >= 3);
  if (!have_clz && (cindex == 2))  cindex = 0;
# elif (CLIENT_CPU == CPU_POWERPC)
  int feature = 0;
  feature = GetProcessorFeatureFlags();
  if ((feature & CPU_F_ALTIVEC) == 0 && cindex == 1)      /* PPC-vector */
    cindex = 0;                                     /* force PPC-scalar */
  if ((feature & CPU_F_64BITOPS) == 0 && cindex == 2)     /* PPC-64bit  */
    cindex = 0;                                     /* force PPC-32bit  */
# elif (CLIENT_CPU == CPU_X86)
  if (cindex >= 3) /* ASM-MMX core requires 64-bit core modules and MMX */
  {
#  if !defined(HAVE_I64) /* no 64-bit support? */
    cindex = 2; /* force ASM-Generic */
#  else /* no MMX? */
    if (!(GetProcessorFeatureFlags() & CPU_F_MMX))
      cindex = 2;
#  endif
  }
# endif

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
    #if (CLIENT_OS != OS_VMS)  /* Include for other OSes */
      if (detected_type >= 11)
        cindex = 5;
      else
    #endif
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
  #elif (CLIENT_CPU == CPU_POWERPC)
    if (detected_type > 0)
    {
      cindex = 0;                       /* PPC-scalar */

      #if defined(__VEC__) || defined(__ALTIVEC__) /* OS+compiler support altivec */
      if ((detected_flags & CPU_F_ALTIVEC) != 0) //altivec?
      {
        cindex = 1;     // PPC-vector

        #if 0 //------------- Yes, PPC-Vector is for any AltiVec capable CPU !
        switch ( detected_type & 0xffff) // only compare the low PVR bits
        {
          case 0x0039: // PPC 970
          case 0x003C: // PPC 970FX
          case 0x0044: // PPC 970MP
            #ifdef HAVE_KOGE_PPC_CORES
              cindex = 1; break;      // PPC-vector
            #else
              cindex = -1; break;     // micro-bench
            #endif
          case 0x000C: // 7400
          case 0x800C: // 7410
          case 0x8000: // 7441/7450/7451
          case 0x8001: // 7445/7455
          case 0x8002: // 7447/7457
          case 0x8003: // 7447A
          case 0x8004: // 7448
              cindex = 1; break;      // PPC-vector
          default:
              cindex =-1; break;      // default : micro-bench
        }
        #endif //------------
      }
      #endif
    }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_X86)
      if (detected_type >= 0)
      {
#ifdef HAVE_I64 // Need 64-bit support and MMX
        if (detected_flags & CPU_F_MMX)
        {
          switch ( detected_type & 0xff ) // FIXME remove &0xff
          {
            case 0x05: cindex = 4; break; // K6/K6-2/K6-3 == asm-rt1-mmx-amd (E)
            case 0x09: cindex = 4; break; // AMD K7/K8  == asm-rt1-mmx-amd (E)
            case 0x11: cindex = 4; break; // Cyrix Model6 == asm-rt1-mmx-amd (E)
            default:   cindex = 3; break; // asm-rt1-mmx (D)
          }
        }
        else
#endif
        switch ( detected_type & 0xff ) // FIXME remove &0xff
        {
          case 0x00: cindex = 2; break; // P5           == asm-rt1-gen (C)
          case 0x01: cindex = 2; break; // 386/486      == asm-rt1-gen (C)
          case 0x02: cindex = 2; break; // PII          == asm-rt1-gen (C)
          case 0x03: cindex = 2; break; // Cyrix Model4 == asm-rt1-gen (C)
          case 0x04: cindex = 2; break; // K5           == asm-rt1-gen (C)
          case 0x05: cindex = 2; break; // K6/K6-2/K6-3 == asm-rt1-gen (C)
          case 0x06: cindex = 2; break; // Cyrix 486    == asm-rt1-gen (C)
          case 0x07: cindex = 2; break; // orig Celeron == asm-rt1-gen (C)
          case 0x08: cindex = 2; break; // PPro         == asm-rt1-gen (C)
          case 0x09: cindex = 2; break; // AMD K7/K8    == asm-rt1-gen (C)
          case 0x0A: cindex = 2; break; // Centaur C6   == asm-rt1-gen (C)
          case 0x0B: cindex = 2; break; // Pentium 4    == asm-rt1-gen (C)
          case 0x0C: cindex = 2; break; // Via C3       == asm-rt1-gen (C)
          case 0x0D: cindex = 2; break; // Pentium M    == asm-rt1-gen (C)
          case 0x0E: cindex = 2; break; // Pentium III  == asm-rt1-gen (C)
          case 0x0F: cindex = 0; break; // Via C3 Nehemiah == (A)
          case 0x10: cindex = 2; break; // Cyrix Model5 == asm-rt1-gen (C)
          case 0x11: cindex = 2; break; // Cyrix Model6 == asm-rt1-gen (C)
          case 0x12: cindex = 2; break; // Intel Core 2 == asm-rt1-gen (C)
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

int selcoreSelectCore_ogr(unsigned int threadindex, int *client_cpuP,
                struct selcore *selinfo, unsigned int contestid)
{
  int use_generic_proto = 0; /* if rc5/des unit_func proto is generic */
  unit_func_union unit_func; /* declared in problem.h */
  int cruncher_is_asynchronous = 0; /* on a co-processor or similar */
  int pipeline_count = 2; /* most cases */
  int client_cpu = CLIENT_CPU; /* usual case */
  int coresel = selcoreGetSelectedCoreForContest(contestid);

  DNETC_UNUSED_PARAM(threadindex);

  if (coresel < 0)
    return -1;
  memset( &unit_func, 0, sizeof(unit_func));


  /* ================================================================== */

#if (CLIENT_CPU == CPU_POWERPC)
  #if defined(__VEC__) || defined(__ALTIVEC__) /* compiler+OS supports AltiVec */
  if (coresel == 1)                               /* PPC Vector/Hybrid */
    unit_func.ogr = vec_ogr_get_dispatch_table();
  #endif

  #if defined(HAVE_I64) && !defined(HAVE_KOGE_PPC_CORES)
  if (coresel == 2)
    unit_func.ogr = ogr64_get_dispatch_table();   /* PPC Scalar-64 */
  #endif

  if (!unit_func.ogr) {
    unit_func.ogr = ogr_get_dispatch_table();     /* PPC Scalar-32 */
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
    if (coresel == 2)
      unit_func.ogr = ogr_get_dispatch_table_ev4();
    else
    if (coresel == 3)
      unit_func.ogr = ogr64_get_dispatch_table();
    else
    if (coresel == 4)
      unit_func.ogr = ogr_get_dispatch_table_ev4_64();
    else
  #if (CLIENT_OS != OS_VMS)       /* Include for other OSes */
    if (coresel == 5)
      unit_func.ogr = ogr_get_dispatch_table_cix_64();
    else
  #endif
      unit_func.ogr = ogr_get_dispatch_table();
#elif (CLIENT_CPU == CPU_X86)
  if (coresel == 1) //B
    unit_func.ogr = ogr_get_dispatch_table_nobsr(); //B
  else
  if (coresel == 2) //C
    unit_func.ogr = ogr_get_dispatch_table_asm_gen();
  #if defined(HAVE_I64)
  else
  if (coresel == 3) //D
    unit_func.ogr = ogr_get_dispatch_table_asm_mmx();
  else
  if (coresel == 4) //E
    unit_func.ogr = ogr_get_dispatch_table_asm_mmx_amd();
  #endif
  else
    unit_func.ogr = ogr_get_dispatch_table(); //A
#elif (CLIENT_CPU == CPU_AMD64)
  //unit_func.ogr = ogr_get_dispatch_table();
  unit_func.ogr = ogr64_get_dispatch_table();
  coresel = 0;
#elif (CLIENT_CPU == CPU_ARM)
  if (coresel == 0)
    unit_func.ogr = ogr_get_dispatch_table_arm1();
  else if (coresel == 2)
    unit_func.ogr = ogr_get_dispatch_table_arm3();
  else
  {
    unit_func.ogr = ogr_get_dispatch_table_arm2();
    coresel = 1;
  }
#elif (CLIENT_CPU == CPU_SPARC) && (SIZEOF_LONG == 8)
  unit_func.ogr = ogr64_get_dispatch_table();
  coresel = 0;
#elif (CLIENT_CPU == CPU_MIPS) && (SIZEOF_LONG == 8)
  unit_func.ogr = ogr64_get_dispatch_table();
  coresel = 0;
#else
  //extern "C" CoreDispatchTable *ogr_get_dispatch_table(void);
  unit_func.ogr = ogr_get_dispatch_table();
  coresel = 0;
#endif

  /* ================================================================== */


  if (coresel >= 0 && unit_func.gen &&
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

unsigned int estimate_nominal_rate_ogr()
{
  unsigned int rate = 0;  /* Unknown - Not available */

  #if (CLIENT_CPU == CPU_POWERPC)
    static long detected_type = -123;
    static int  cpu_count = 0;
    static unsigned long detected_flags = 0;
    static unsigned int  frequency = 0;
    unsigned int noderate = 0;   /* nodes/s/MHz */

    if (detected_type == -123) {
      detected_type  = GetProcessorType(1);
      detected_flags = GetProcessorFeatureFlags();
      frequency      = GetProcessorFrequency();
      cpu_count      = GetNumberOfDetectedProcessors();
    }

    if (detected_type > 0) {
      switch (detected_type & 0xffff) { // only compare the low PVR bits
        case 0x0001:      // 601
          noderate = 5000; break;
        case 0x0003:      // 603
        case 0x0004:      // 604
        case 0x0006:      // 603e
        case 0x0007:      // 603r/603ev
        case 0x0008:      // 740/750
        case 0x0009:      // 604e
        case 0x000A:      // 604ev
        case 0x7000:      // 750FX
          noderate = 12000; break;
        case 0x000C:      // 7400
        case 0x800C:      // 7410
          noderate = (detected_flags & CPU_F_ALTIVEC) ? 14000: 13000; break;
        case 0x8000:      // 7450
        case 0x8001:      // 7455
        case 0x8002:      // 7457/7447
        case 0x8003:      // 7447A
        case 0x8004:      // 7448
          noderate = (detected_flags & CPU_F_ALTIVEC) ? 24000 : 17000; break;
        case 0x0039:      // 970
        case 0x003C:      // 970FX
        case 0x0044:      // 970MP
          noderate = (detected_flags & CPU_F_ALTIVEC) ? 16500 : 12500; break;
      }

      if (cpu_count > 0) {
        /* Assume 70 GNodes per packet */
        rate = (noderate * frequency * cpu_count) / 810000;  /* 810000 = 70E9 / 86400 */
      }
    }
  #endif

  return rate;
}

/* ------------------------------------------------------------- */

#endif // defined(HAVE_OGR_CORES) || defined(HAVE_OGR_PASS2)
