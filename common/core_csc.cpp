/*
 * Copyright distributed.net 1998-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *core_csc_cpp(void) {
return "@(#)$Id: core_csc.cpp,v 1.3 2003/11/01 15:00:08 mweiser Exp $"; }

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

#if defined(HAVE_CSC_CORES)

/* ======================================================================== */

/* all the core prototypes
   note: we may have more prototypes here than cores in the client
   note2: if you need some 'cdecl' value define it in selcore.h to CDECL */


extern "C" s32 csc_unit_func_1k  ( RC5UnitWork *, u32 *iterations, void *membuff );
#if (CLIENT_CPU != CPU_ARM) // ARM only has one CSC core
extern "C" s32 csc_unit_func_1k_i( RC5UnitWork *, u32 *iterations, void *membuff );
extern "C" s32 csc_unit_func_6b  ( RC5UnitWork *, u32 *iterations, void *membuff );
extern "C" s32 csc_unit_func_6b_i( RC5UnitWork *, u32 *iterations, void *membuff );
#endif
#if (CLIENT_CPU == CPU_X86) && !defined(HAVE_NO_NASM)
extern "C" s32 csc_unit_func_6b_mmx ( RC5UnitWork *, u32 *iterations, void *membuff );
#endif


/* ======================================================================== */

int InitializeCoreTable_csc(int /*first_time*/)
{
  /* csc does not require any initialization */
  return 0;
}

void DeinitializeCoreTable_csc()
{
  /* csc does not require any deinitialization */
}

/* ======================================================================== */


const char **corenames_for_contest_csc()
{
  /*
   When selecting corenames, use names that describe how (what optimization)
   they are different from their predecessor(s). If only one core,
   use the obvious "MIPS optimized" or similar.
  */
  static const char *corenames_table[] =
    {
  /* ================================================================== */
      "6 bit - inline",
      "6 bit - called",
      "1 key - inline",
      "1 key - called",
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
int apply_selcore_substitution_rules_csc(int cindex)
{
  #if (CLIENT_CPU == CPU_ARM)
    if (cindex != 0) /* "1 key - called" */
      return 0;      /* the only supported core */
  #endif
  return cindex;
}

/* -------------------------------------------------------------------- */

int selcoreGetPreselectedCoreForProject_csc()
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

  // PROJECT_NOT_HANDLED("you may add your pre-selected core depending on arch
  //  and cpu here, but leaving the defaults (runs micro-benchmark) is ok")

  // ===============================================================
  #if (CLIENT_CPU == CPU_X86)
  {
    int have_mmx = ((detected_flags & CPU_F_MMX) == CPU_F_MMX);
      if (detected_type >= 0)
      {
        // this is only valid for nasm'd cores or GCC 2.95 and up
        switch ( detected_type & 0xff ) // FIXME remove &0xff
        {
          case 0x00: cindex = 3; break; // P5           == 1key - called
          case 0x01: cindex = 3; break; // 386/486      == 1key - called
          case 0x02: cindex = 2; break; // PII/PIII     == 1key - inline
          case 0x03: cindex = 3; break; // Cx6x86       == 1key - called
          case 0x04: cindex = 2; break; // K5           == 1key - inline
          case 0x05: cindex = 0; break; // K6/K6-2/K6-3 == 6bit - inline
          case 0x06: cindex = 3; break; // Cyrix 486    == 1key - called
          case 0x07: cindex = 3; break; // orig Celeron == 1key - called
          case 0x08: cindex = 3; break; // PPro         == 1key - called
          case 0x09: cindex = 0; break; // AMD K7       == 6bit - inline
          case 0x0A: cindex = 3; break; // Centaur C6
          case 0x0B: cindex = 0; break; // Pentium 4
          default:   cindex =-1; break; // no default
        }
        #if !defined(HAVE_NO_NASM)
        if (have_mmx)
          cindex = 1; /* == 6bit - called - MMX */
        #endif
      }
  }
  // ===============================================================
  #endif

  return cindex;
}


/* ---------------------------------------------------------------------- */

int selcoreSelectCore_csc(unsigned int threadindex,
                          int *client_cpuP, struct selcore *selinfo)
{
  int use_generic_proto = 0; /* if rc5/des unit_func proto is generic */
  unit_func_union unit_func; /* declared in problem.h */
  int cruncher_is_asynchronous = 0; /* on a co-processor or similar */
  int pipeline_count = 2; /* most cases */
  int client_cpu = CLIENT_CPU; /* usual case */
  int coresel = selcoreGetSelectedCoreForContest(CSC);

  DNETC_UNUSED_PARAM(threadindex);

  if (coresel < 0)
    return -1;
  memset( &unit_func, 0, sizeof(unit_func));

  /* -------------------------------------------------------------- */

    //xtern "C" s32 csc_unit_func_1k  ( RC5UnitWork *, u32 *iterations, void *membuff );
    //xtern "C" s32 csc_unit_func_1k_i( RC5UnitWork *, u32 *iterations, void *membuff );
    //xtern "C" s32 csc_unit_func_6b  ( RC5UnitWork *, u32 *iterations, void *membuff );
    //xtern "C" s32 csc_unit_func_6b_i( RC5UnitWork *, u32 *iterations, void *membuff );
   #if (CLIENT_CPU == CPU_ARM)
    coresel = 0;
    unit_func.gen = csc_unit_func_1k;
   #else
    use_generic_proto = 1; /* all CSC cores use generic form */
    switch( coresel )
    {
      case 0 : unit_func.gen = csc_unit_func_6b_i;
               break;
      case 1 : unit_func.gen = csc_unit_func_6b;
               #if (CLIENT_CPU == CPU_X86) && !defined(HAVE_NO_NASM)
               {               //6b-non-mmx isn't used (by default) on x86
                 long det = GetProcessorType(1 /* quietly */);
                 if ((det >= 0 && (det & 0x100)!=0)) /* ismmx */
                   unit_func.gen = csc_unit_func_6b_mmx;
               }
               #endif
               break;
      default: coresel = 2;
      case 2 : unit_func.gen = csc_unit_func_1k_i;
               break;
      case 3 : unit_func.gen = csc_unit_func_1k;
               break;
    }
   #endif

  /* ================================================================== */


  if (coresel >= 0 && unit_func.gen &&
     coresel < ((int)corecount_for_contest(CSC)))
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

#endif /* #if defined(HAVE_CSC_CORES) */
