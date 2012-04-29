/*
 * Copyright distributed.net 1998-2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *core_ogr_ng_cpp(void) {
return "@(#)$Id: core_ogr_ng.cpp,v 1.46 2012/04/29 14:30:34 snikkel Exp $"; }

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
#include "ogr.h"


/* ======================================================================== */

/* all the core prototypes
   note: we may have more prototypes here than cores in the client
   note2: if you need some 'cdecl' value define it in selcore.h to CDECL */

#if (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_CELLBE)
    CoreDispatchTable *ogrng_get_dispatch_table(void);
    #if defined(HAVE_I64) && !defined(HAVE_FLEGE_PPC_CORES)
    CoreDispatchTable *ogrng64_get_dispatch_table(void);
    #endif
    #if defined(HAVE_ALTIVEC) /* compiler supports AltiVec */
    CoreDispatchTable *vec_ogrng_get_dispatch_table(void);
    #endif
    #if (CLIENT_CPU == CPU_CELLBE)
    CoreDispatchTable *spe_ogrng_get_dispatch_table(void);
    CoreDispatchTable *spe_ogrng_get_dispatch_table_asm(void);
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
      CoreDispatchTable *ogrng_get_dispatch_table_mmx(void);  //C (asm #2)
      CoreDispatchTable *ogrng_get_dispatch_table_cj1_sse2(void);  //D (asm #3)
      CoreDispatchTable *ogrng_get_dispatch_table_cj1_sse_p4(void);  //E (asm #4)
      CoreDispatchTable *ogrng_get_dispatch_table_cj1_sse_k8(void);  //F (asm #5)
      CoreDispatchTable *ogrng_get_dispatch_table_cj1_sse41(void);  //G (asm #6)
      CoreDispatchTable *ogrng_get_dispatch_table_cj1_sse2_lzcnt(void);  //H (asm #7)
    #endif
#elif (CLIENT_CPU == CPU_ARM)
    CoreDispatchTable *ogrng_get_dispatch_table(void);
    CoreDispatchTable *ogrng_get_dispatch_table_arm1(void);
    CoreDispatchTable *ogrng_get_dispatch_table_arm2(void);
    CoreDispatchTable *ogrng_get_dispatch_table_arm3(void);
#elif (CLIENT_CPU == CPU_AMD64)
    CoreDispatchTable *ogrng64_get_dispatch_table(void);
    CoreDispatchTable *ogrng64_get_dispatch_table_cj1_generic(void);
    CoreDispatchTable *ogrng64_get_dispatch_table_cj1_sse2(void);
    CoreDispatchTable *ogrng64_get_dispatch_table_cj1_sse2_lzcnt(void);
#elif (CLIENT_CPU == CPU_SPARC) && (SIZEOF_LONG == 8)
    CoreDispatchTable *ogrng64_get_dispatch_table(void); 
#elif (CLIENT_CPU == CPU_S390X) && (SIZEOF_LONG == 8)
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
        #ifdef HAVE_I64
          ogrng_get_dispatch_table_mmx();
          ogrng_get_dispatch_table_cj1_sse2();
          ogrng_get_dispatch_table_cj1_sse_p4();
          ogrng_get_dispatch_table_cj1_sse_k8();
          ogrng_get_dispatch_table_cj1_sse41();
          ogrng_get_dispatch_table_cj1_sse2_lzcnt();
        #endif
        #endif
      #elif (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_CELLBE)
        ogrng_get_dispatch_table();
        #if defined(HAVE_I64) && !defined(HAVE_FLEGE_PPC_CORES)
          ogrng64_get_dispatch_table();
        #endif
        #if defined(HAVE_ALTIVEC) /* compiler supports AltiVec */
          vec_ogrng_get_dispatch_table();
        #endif
        #if (CLIENT_CPU == CPU_CELLBE)
          spe_ogrng_get_dispatch_table();
          spe_ogrng_get_dispatch_table_asm();
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
      #elif (CLIENT_CPU == CPU_ARM)
        ogrng_get_dispatch_table();
        ogrng_get_dispatch_table_arm1();
        ogrng_get_dispatch_table_arm2();
        ogrng_get_dispatch_table_arm3();
      #elif (CLIENT_CPU == CPU_SPARC)
        #if (SIZEOF_LONG == 8)
          ogrng64_get_dispatch_table();
        #else
          ogrng_get_dispatch_table();
        #endif
      #elif (CLIENT_CPU == CPU_AMD64)
        //ogr_get_dispatch_table();
        ogrng64_get_dispatch_table();
        ogrng64_get_dispatch_table_cj1_generic();
        ogrng64_get_dispatch_table_cj1_sse2();
        ogrng64_get_dispatch_table_cj1_sse2_lzcnt();
      #elif (CLIENT_CPU == CPU_S390)
        ogrng_get_dispatch_table();
      #elif (CLIENT_CPU == CPU_S390X)
        ogrng64_get_dispatch_table();
      #elif (CLIENT_CPU == CPU_I64)
        ogrng_get_dispatch_table();
      #elif (CLIENT_CPU == CPU_PA_RISC)
        ogrng_get_dispatch_table();
      #elif (CLIENT_CPU == CPU_MIPS)
        ogrng_get_dispatch_table();
        #if (SIZEOF_LONG == 8)
          ogrng64_get_dispatch_table();
        #endif
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
      "rt-asm-mmx",
      "cj-asm-sse2",
      "cj-asm-sse-p4",
      "cj-asm-sse-k8",
      "cj-asm-sse4.1",
      "cj-asm-sse2-lzcnt",
      #endif
  #elif (CLIENT_CPU == CPU_AMD64)
      "FLEGE-64 2.0",
      "cj-asm-generic",
      "cj-asm-sse2",
      "cj-asm-sse2-lzcnt",
  #elif (CLIENT_CPU == CPU_ARM)
      "FLEGE 2.0",
      "FLEGE 2.0 ARMv3",
      "FLEGE 2.0 ARMv5-XScale",
      "FLEGE 2.0 ARMv5",
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
      "KOGE 3.1 Scalar",
      "KOGE 3.1 Hybrid",            /* altivec only */
    #else
      "FLEGE 2.0 Scalar-32",
      "FLEGE 2.0 Hybrid",           /* altivec only */
      #ifdef HAVE_I64
      "FLEGE-64 2.0",
      #endif
    #endif
    #if (CLIENT_CPU == CPU_CELLBE)
      "Cell v2 SPE (base)",
      "Cell v2 SPE (asm)",
    #endif
  #elif (CLIENT_CPU == CPU_SPARC) && (SIZEOF_LONG == 8)
      "FLEGE-64 2.0",
  #elif (CLIENT_CPU == CPU_SPARC)
      "FLEGE 2.0",
  #elif (CLIENT_CPU == CPU_S390X) && (SIZEOF_LONG == 8)
      "FLEGE-64 2.0",
  #elif (CLIENT_CPU == CPU_S390)
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
# elif (CLIENT_CPU == CPU_ARM)
  long det = GetProcessorType(1);
  int have_clz = (((det >> 12) & 0xf) != 0) &&
                 (((det >> 12) & 0xf) != 7) &&
                 (((det >> 16) & 0xf) >= 3);
  if (!have_clz && (cindex == 2))  cindex = 1;
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
#  if defined(HAVE_I64)
#    if (SIZEOF_LONG < 8)   /* classic x86-32 */
       unsigned feature = GetProcessorFeatureFlags();
       if (cindex == 7 && !(feature & CPU_F_LZCNT)) /* Core 7 needs LZCNT */
         cindex = 3; /* If no LZCNT, try SSE2 */
       if (cindex == 6 && !(feature & CPU_F_SSE4_1)) /* Core 6 needs SSE4.1 */
         cindex = 3; /* If no SSE4.1, try SSE2 */
       if (cindex == 5 && !(feature & CPU_F_SSE)) /* Core 5 needs SSE */
         cindex = 2;
       if (cindex == 4 && !(feature & CPU_F_SSE)) /* Core 4 needs SSE */
         cindex = 2;
       if (cindex == 3 && !(feature & CPU_F_SSE2)) /* Core 3 needs SSE2 */
         cindex = 2; /* If no SSE2, try MMX */
       if (cindex == 2 && !(feature & CPU_F_MMX)) /* MMX for MMX core */
         cindex = 1;
#    else                   /* x86-64 */
#    endif
#  else /* No 64-bit support in compiler */
#    if (SIZEOF_LONG < 8)   /* classic x86-32 */
       if (cindex >= 2)  /* mmx/sse cores requires 64-bits types */
         cindex = 1;
#    endif
#  endif
# elif (CLIENT_CPU == CPU_AMD64)
  unsigned feature = GetProcessorFeatureFlags();
  if (cindex == 3 && !(feature & CPU_F_LZCNT)) /* Core 3 needs LZCNT */
    cindex = 2; /* If no LZCNT, try SSE2 */
  if (cindex == 2 && !(feature & CPU_F_SSE2))  /* Core 2 needs SSE2 */
    cindex = 1; /* If no SSE2, try generic */
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

      #if defined(HAVE_ALTIVEC) /* OS+compiler support altivec */
      if ((detected_flags & CPU_F_ALTIVEC) != 0) //altivec?
      {
        switch (detected_type & 0xFFFF)
        {
        case 0x003E:  // POWER6: slow Altivec? Keep scalar (0)
          break; 
        default:
          cindex = 1; // PPC-vector (Altivec)
          break;
        }
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
        #if defined(HAVE_I64)
          /* Assume LZCNT is the best for all CPUs which support it? */
          if (cindex == -1 && detected_flags & CPU_F_LZCNT)
          {
            switch (detected_type)
            {
              default:   cindex = 7;
            }
          }
          /* SSE 4.1 - enable for all supported CPUs or just for group 0x12 (Core2) ? */
          if (cindex == -1 && (detected_flags & CPU_F_SSE4_1))
          {
            switch (detected_type)
            {
              case 0x15: cindex = 3; break; /* Intel i3/i5/i7 */
              default:   cindex = 6; break;
            }
          }
	  
          /* Some CPU types may require SSE core even if SSE2 is available but slow */
          if (cindex == -1 && (detected_flags & CPU_F_SSE))
          {  
            /*
             * Sometimes it's very difficult to choose between -p4 and -k8 cores:
             * results are very close. Note that "dnetc -bench" and "dnetc -bench n"
             * can produce different results.
             */
            switch (detected_type)
            {
              case 0x09: cindex = 5; break; /* AMD: sse-k8. Wrong for sure. Too many different AMDs covered by type 9 */
              case 0x0A: cindex = 2; break; /* VIA C7-M (at least) - mmx */
              case 0x0B: cindex = 4; break; /* P4:  sse-p4 */
              case 0x0D: cindex = 5; break; /* Pentium M: -k8 */
              case 0x13: cindex = 5; break; /* P4-Willamette: -k8 but -k8 and -p4 are very close */
              case 0x17: cindex = 4; break; /* P4:  sse-p4 (but different rc5-72 core) (#4186) */
            }
          }
          /* If core not set above and SSE2 exist, try it */
          if (cindex == -1 && (detected_flags & CPU_F_SSE2))
            cindex = 3;  /* generic sse asm core for fast sse2 like core2 */
          /* Same for MMX */
          if (cindex == -1 && (detected_flags & CPU_F_MMX))
            cindex = 2;  /* mmx asm core */
          if (cindex == -1)
            cindex = 1;  /* no mmx - generic asm core */
        #else
          cindex = 1; /* no 64-bit support - generic asm core */
        #endif
      #endif
      }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_ARM)
    {
      extern signed int default_ogr_core;

      cindex = default_ogr_core;
    }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_AMD64)
    {
      switch (detected_type)
      {
        case 0x09: cindex = 1; break; /* AMD: generic (#4214) */
        /*case 0x20: cindex = 1; break;*/ /* AMD APU (#4429/#4485) */
      }
      if (cindex == -1)
      {
        /* Assume that LZCNT+SSE2 is better then plain SSE2 everywhere */
        if (detected_flags & CPU_F_LZCNT)
          cindex = 3;
        else if (detected_flags & CPU_F_SSE2)
          cindex = 2;  /* sse2 core */
        else
          cindex = 1;  /* generic asm core */
       }
    }
  // ===============================================================
  #endif

  return cindex;
}

/* ---------------------------------------------------------------------- */

int selcoreSelectCore_ogr_ng(Client *client, unsigned int threadindex, 
                int *client_cpuP,
                struct selcore *selinfo, unsigned int contestid)
{
  int use_generic_proto = 0; /* if rc5/des unit_func proto is generic */
  unit_func_union unit_func; /* declared in problem.h */
  int cruncher_is_asynchronous = 0; /* on a co-processor or similar */
  int pipeline_count = 2; /* most cases */
  int client_cpu = CLIENT_CPU; /* usual case */
  int coresel = selcoreGetSelectedCoreForContest(client, contestid);

#if (CLIENT_CPU == CPU_CELLBE)
  // Each Cell has 1 PPE, which is dual-threaded (so in fact the OS sees 2
  // processors), and although we should run 2 threads at a time, the way
  // CPUs are detected precludes us from doing that.

  // Threads with threadindex = 0..PPE_count-1 will be scheduled on the PPEs;
  // the rest are scheduled on the SPEs.
  if (threadindex >= (unsigned)GetNumberOfPhysicalProcessors())
    coresel = 3;
#else
  DNETC_UNUSED_PARAM(threadindex);
#endif


  if (coresel < 0)
    return -1;

  memset( &unit_func, 0, sizeof(unit_func));


  /* ================================================================== */

#if (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_CELLBE)
  #if defined(HAVE_ALTIVEC) /* compiler+OS supports AltiVec */
  if (coresel == 1)                               /* PPC Vector/Hybrid */
     unit_func.ogr = vec_ogrng_get_dispatch_table();
  #endif

  #if (CLIENT_CPU == CPU_CELLBE)
  if (coresel == 2)
    unit_func.ogr = spe_ogrng_get_dispatch_table();
  if (coresel == 3)
    unit_func.ogr = spe_ogrng_get_dispatch_table_asm();
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
  #ifdef HAVE_I64
    else if (coresel == 2)
      unit_func.ogr = ogrng_get_dispatch_table_mmx();
    else if (coresel == 3)
      unit_func.ogr = ogrng_get_dispatch_table_cj1_sse2();
    else if (coresel == 4)
      unit_func.ogr = ogrng_get_dispatch_table_cj1_sse_p4();
    else if (coresel == 5)
      unit_func.ogr = ogrng_get_dispatch_table_cj1_sse_k8();
    else if (coresel == 6)
      unit_func.ogr = ogrng_get_dispatch_table_cj1_sse41();
    else if (coresel == 7)
      unit_func.ogr = ogrng_get_dispatch_table_cj1_sse2_lzcnt();
  #endif
    else
      unit_func.ogr = ogrng_get_dispatch_table();
  #endif
#elif (CLIENT_CPU == CPU_AMD64)
  if (coresel == 1)
    unit_func.ogr = ogrng64_get_dispatch_table_cj1_generic();
  else if (coresel == 2)
    unit_func.ogr = ogrng64_get_dispatch_table_cj1_sse2();
  else if (coresel == 3)
    unit_func.ogr = ogrng64_get_dispatch_table_cj1_sse2_lzcnt();
  else
  {
    unit_func.ogr = ogrng64_get_dispatch_table();
    coresel = 0;
  }
#elif (CLIENT_CPU == CPU_ARM)
  if (coresel == 1)
    unit_func.ogr = ogrng_get_dispatch_table_arm1();
  else if (coresel == 2)
    unit_func.ogr = ogrng_get_dispatch_table_arm2();
  else if (coresel == 3)
    unit_func.ogr = ogrng_get_dispatch_table_arm3();
  else
  {
    unit_func.ogr = ogrng_get_dispatch_table();
    coresel = 0;
  }
#elif (CLIENT_CPU == CPU_SPARC) && (SIZEOF_LONG == 8)
  unit_func.ogr = ogrng64_get_dispatch_table();
  coresel = 0;
#elif (CLIENT_CPU == CPU_S390X) && (SIZEOF_LONG == 8)
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
