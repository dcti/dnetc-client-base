/*
 * Copyright distributed.net 1998-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *core_des_cpp(void) {
return "@(#)$Id: core_des.cpp,v 1.1.2.3 2003/09/01 22:10:46 mweiser Exp $"; }

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
#include "util.h"      // TRACE_OUT


#if defined(HAVE_DES_CORES)

/* ======================================================================== */

/* all the core prototypes
   note: we may have more prototypes here than cores in the client
   note2: if you need some 'cdecl' value define it in selcore.h to CDECL */

/* DES cores take the 'iterations_to_do', adjust it to min/max/nbbits
  and store it back in 'iterations_to_do'. all return 'iterations_done'.
*/
#if (CLIENT_CPU == CPU_ARM)
  //des/arm/des-arm-wrappers.cpp
  extern u32 des_unit_func_slice_arm( RC5UnitWork * , u32 *iter, char *coremem );
  extern u32 des_unit_func_slice_strongarm(RC5UnitWork *, u32 *iter, char *coremem);
#elif (CLIENT_CPU == CPU_ALPHA)
  //des/alpha/des-slice-dworz.cpp
  extern u32 des_unit_func_slice_dworz( RC5UnitWork * , u32 *iter, char *);
#elif (CLIENT_CPU == CPU_X86)
  extern u32 p1des_unit_func_p5( RC5UnitWork * , u32 *iter, char *coremem );
  extern u32 p1des_unit_func_pro( RC5UnitWork * , u32 *iter, char *coremem );
  extern u32 p2des_unit_func_p5( RC5UnitWork * , u32 *iter, char *coremem );
  extern u32 p2des_unit_func_pro( RC5UnitWork * , u32 *iter, char *coremem );
  extern u32 des_unit_func_mmx( RC5UnitWork * , u32 *iter, char *coremem );
  extern u32 des_unit_func_slice( RC5UnitWork * , u32 *iter, char *coremem );
#elif defined(MEGGS)
  //des/des-slice-meggs.cpp
  extern u32 des_unit_func_meggs( RC5UnitWork * , u32 *iter, char *coremem );
#else
  //all rvs based drivers (eg des/ultrasparc/des-slice-ultrasparc.cpp)
  extern u32 des_unit_func_slice( RC5UnitWork * , u32 *iter, char *coremem );
#endif

/* ======================================================================== */

int InitializeCoreTable_des(int /*first_time*/)
{
  /* des does not require any initialization */
  return 0;
}

/* ======================================================================== */

void DeinitializeCoreTable_des()
{
  /* des does not require any initialization */
}


/* ======================================================================== */


const char **corenames_for_contest_des()
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
      "byte Bryd",
      "movzx Bryd",
      "Kwan/Bitslice",
      "MMX/Bitslice",
  #elif (CLIENT_CPU == CPU_X86_64)
      "Generic DES core",
  #elif (CLIENT_CPU == CPU_ARM)
      "Standard ARM core", /* "ARM 3, 610, 700, 7500, 7500FE" or  "ARM 710" */
      "StrongARM core", /* "ARM 810, StrongARM 110, 1100, 1110" or "ARM 2, 250" */
  #elif (CLIENT_CPU == CPU_68K)
      "Generic",
  #elif (CLIENT_CPU == CPU_ALPHA)
      "dworz/amazing",
  #elif (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_POWER)
      "Generic DES core",
  #elif (CLIENT_CPU == CPU_SPARC)
      "Generic DES core",
  #elif (CLIENT_OS == OS_PS2LINUX)
      "Generic DES core",
  #else
      "Generic DES core",
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
int apply_selcore_substitution_rules_des(int cindex)
{
  #if (CLIENT_CPU == CPU_X86)
    long det = GetProcessorType(1);
    int have_mmx = (det >= 0 && (det & 0x100)!=0);

    #if !defined(CLIENT_SUPPORTS_SMP)
      if (cindex == 2)                /* "Kwan/Bitslice" */
        cindex = 1;                   /* "movzx Bryd" */
    #endif
    #if !defined(MMX_BITSLICER)
      if (cindex == 3)                /* "BRF MMX bitslice */
        cindex = 1;                   /* "movzx Bryd" */
    #endif
      if (!have_mmx && cindex == 3)   /* "BRF MMX bitslice */
        cindex = 1;                   /* "movzx Bryd" */

  #endif

  return cindex;
}

/* -------------------------------------------------------------------- */

int selcoreGetPreselectedCoreForProject_des()
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

  // ===============================================================
  #if (CLIENT_CPU == CPU_68K)
    cindex = 0; // only one core
  // ===============================================================
  #elif (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_POWER)
    cindex = 0; /* only one DES core */
  // ===============================================================
  #elif (CLIENT_CPU == CPU_X86)
  {
    int have_mmx = ((detected_flags & CPU_F_MMX) == CPU_F_MMX);

      if (detected_type >= 0)
      {
        switch ( detected_type & 0xff ) // FIXME remove &0xff
        {
          case 0x00: cindex = 0; break; // P5             == standard Bryd
          case 0x01: cindex = 0; break; // 386/486        == standard Bryd
          case 0x02: cindex = 1; break; // PII/PIII       == movzx Bryd
          case 0x03: cindex = 1; break; // Cx6x86         == movzx Bryd
          case 0x04: cindex = 0; break; // K5             == standard Bryd
          case 0x05: cindex = 1; break; // K6             == movzx Bryd
          case 0x06: cindex = 0; break; // Cx486          == movzx Bryd
          case 0x07: cindex = 1; break; // orig Celeron   == movzx Bryd
          case 0x08: cindex = 1; break; // PPro           == movzx Bryd
          case 0x09: cindex = 1; break; // AMD K7         == movzx Bryd
          case 0x0A: cindex = 0; break; // Centaur C6
          case 0x0B: cindex = 1; break; // Pentium 4
          default:   cindex =-1; break; // no default
        }
        #ifdef MMX_BITSLICER
        if (have_mmx)
          cindex = 2; /* mmx bitslicer */
        #endif
      }
  }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_ARM)
    if (detected_type > 0)
    {
      if (detected_type == 0x810 ||  /* ARM 810 */
          detected_type == 0xA10 ||  /* StrongARM 110 */
          detected_type == 0xA11 ||  /* StrongARM 1100 */
          detected_type == 0xB11 ||  /* StrongARM 1110 */
          detected_type == 0x200 ||  /* ARM 2 */
          detected_type == 0x250)    /* ARM 250 */
        cindex = 1;
      else /* "ARM 3, 610, 700, 7500, 7500FE" or  "ARM 710" */
        cindex = 0;
    }
  // ===============================================================
  #endif

  return cindex;
}

/* ---------------------------------------------------------------------- */

int selcoreSelectCore_des(unsigned int threadindex,
                       int *client_cpuP, struct selcore *selinfo )
{
  int use_generic_proto = 0; /* if rc5/des unit_func proto is generic */
  unit_func_union unit_func; /* declared in problem.h */
  int cruncher_is_asynchronous = 0; /* on a co-processor or similar */
  int pipeline_count = 2; /* most cases */
  int client_cpu = CLIENT_CPU; /* usual case */
  int coresel = selcoreGetSelectedCoreForContest(DES);
  if (coresel < 0)
    return -1;
  memset( &unit_func, 0, sizeof(unit_func));

  /* -------------------------------------------------------------- */

    #if (CLIENT_CPU == CPU_ARM)
    {
      //des/arm/des-arm-wrappers.cpp
      //xtern u32 des_unit_func_slice_arm( RC5UnitWork * , u32 *, char * );
      //xtern u32 des_unit_func_slice_strongarm( RC5UnitWork * , u32 *, char * );
      if (coresel == 0)
        unit_func.des = des_unit_func_slice_arm;
      else /* (coresel == 1, default) */
      {
        unit_func.des = des_unit_func_slice_strongarm;
        coresel = 1;
      }
    }
    #elif (CLIENT_CPU == CPU_ALPHA)
    {
      //des/alpha/des-slice-dworz.cpp
      //xtern u32 des_unit_func_slice_dworz( RC5UnitWork * , u32 *, char * );
      unit_func.des = des_unit_func_slice_dworz;
    }
    #elif (CLIENT_CPU == CPU_X86)
    {
      //xtern u32 p1des_unit_func_p5( RC5UnitWork * , u32 *, char * );
      //xtern u32 p1des_unit_func_pro( RC5UnitWork * , u32 *, char * );
      //xtern u32 p2des_unit_func_p5( RC5UnitWork * , u32 *, char * );
      //xtern u32 p2des_unit_func_pro( RC5UnitWork * , u32 *, char * );
      //xtern u32 des_unit_func_mmx( RC5UnitWork * , u32 *, char * );
      //xtern u32 des_unit_func_slice( RC5UnitWork * , u32 *, char * );

      u32 (*kwan)(RC5UnitWork *,u32 *,char *) =
                   ((u32 (*)(RC5UnitWork *,u32 *,char *))0);
      u32 (*mmxslice)(RC5UnitWork *,u32 *,char *) =
                   ((u32 (*)(RC5UnitWork *,u32 *,char *))0);
      u32 (*bryd_fallback)(RC5UnitWork *,u32 *,char *) =
                   ((u32 (*)(RC5UnitWork *,u32 *,char *))0);

      #if defined(CLIENT_SUPPORTS_SMP)
      bryd_fallback = kwan = des_unit_func_slice; //kwan
      #endif
      #if defined(MMX_BITSLICER)
      {
        long det = GetProcessorType(1 /* quietly */);
        if ((det >= 0 && (det & 0x100)!=0)) /* ismmx */
          bryd_fallback = mmxslice = des_unit_func_mmx;
      }
      #endif

      if (coresel == 3 && mmxslice)
      {
        unit_func.des = mmxslice;
      }
      else if (coresel == 2 && kwan) /* Kwan */
      {
        unit_func.des = kwan;
      }
      else if (coresel == 1) /* movzx bryd */
      {
        unit_func.des = p1des_unit_func_pro;
        #if defined(CLIENT_SUPPORTS_SMP)
        if (threadindex > 0)  /* not first thread */
        {
          if (threadindex == 1)  /* second thread */
            unit_func.des = p2des_unit_func_pro;
          else if (threadindex == 2) /* third thread */
            unit_func.des = p1des_unit_func_p5;
          else if (threadindex == 3) /* fourth thread */
            unit_func.des = p2des_unit_func_p5;
          else                           /* fifth...nth thread */
            unit_func.des = bryd_fallback; /* kwan */
        }
        #endif /* if defined(CLIENT_SUPPORTS_SMP)  */
      }
      else             /* normal bryd */
      {
        coresel = 0;
        unit_func.des = p1des_unit_func_p5;
        #if defined(CLIENT_SUPPORTS_SMP)
        if (threadindex > 0)  /* not first thread */
        {
          if (threadindex == 1)          /* second thread */
            unit_func.des = p2des_unit_func_p5;
          else if (threadindex == 2)     /* third thread */
            unit_func.des = p1des_unit_func_pro;
          else if (threadindex == 3)     /* fourth thread */
            unit_func.des = p2des_unit_func_pro;
          else                           /* fifth...nth thread */
            unit_func.des = bryd_fallback;
        }
        #endif /* if defined(CLIENT_SUPPORTS_SMP)  */
      }
    }
    #elif defined(MEGGS)
      //des/des-slice-meggs.cpp
      //xtern u32 des_unit_func_meggs( RC5UnitWork *, u32 *iter, char *coremem);
      unit_func.des = des_unit_func_meggs;
    #else
      //all rvc based drivers (eg des/ultrasparc/des-slice-ultrasparc.cpp)
      //xtern u32 des_unit_func_slice( RC5UnitWork *, u32 *iter, char *coremem);
      unit_func.des = des_unit_func_slice;
    #endif

  /* ================================================================== */


  if (coresel >= 0 && unit_func.gen &&
     coresel < ((int)corecount_for_contest(DES)))
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

  threadindex = threadindex; /* possibly unused. shaddup compiler */
  return -1; /* core selection failed */
}

/* ------------------------------------------------------------- */

#endif /* #ifdef HAVE_DES_CORES */
