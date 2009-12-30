/*
 * Copyright distributed.net 1998-2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *core_r72_cpp(void) {
return "@(#)$Id: core_r72.cpp,v 1.46 2009/12/30 14:57:33 sla Exp $"; }

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

#if defined(HAVE_RC5_72_CORES)

/* ======================================================================== */

/* all the core prototypes
   note: we may have more prototypes here than cores in the client
   note2: if you need some 'cdecl' value define it in selcore.h to CDECL */

// These are the standard ANSI cores that are available for all platforms.
extern "C" s32 CDECL rc5_72_unit_func_ansi_4( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_ansi_2( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_ansi_1( RC5_72UnitWork *, u32 *, void * );

// These are (assembly-)optimized versions for each platform.
#if (CLIENT_CPU == CPU_X86) && !defined(HAVE_NO_NASM)
extern "C" s32 CDECL rc5_72_unit_func_ses( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_ses_2( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_dg_2( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_dg_3( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_dg_3a( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_ss_2( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_go_2( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_go_2a( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_go_2b( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_sgp_3( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_ma_4( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_mmx( RC5_72UnitWork *, u32 *, void *);
#elif (CLIENT_CPU == CPU_AMD64)
extern "C" s32 CDECL rc5_72_unit_func_snjl( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_kbe( RC5_72UnitWork *, u32 *, void *);
#elif (CLIENT_CPU == CPU_ARM)
extern "C" s32 rc5_72_unit_func_arm1( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 rc5_72_unit_func_arm2( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 rc5_72_unit_func_arm3( RC5_72UnitWork *, u32 *, void *);
#elif (CLIENT_CPU == CPU_S390X)
extern "C" s32 rc5_72_unit_func_ansi_1_s390x_gcc32( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 rc5_72_unit_func_ansi_2_s390x_gcc32( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 rc5_72_unit_func_ansi_4_s390x_gcc32( RC5_72UnitWork *, u32 *, void *);
#elif (CLIENT_CPU == CPU_68K) && (defined(__GCC__) || defined(__GNUC__))
extern "C" s32 CDECL rc5_72_unit_func_060_mh_2( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_030_mh_1( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_040_mh_1( RC5_72UnitWork *, u32 *, void *);
#elif (CLIENT_CPU == CPU_POWERPC) && \
      (CLIENT_OS != OS_WIN32)
extern "C" s32 CDECL rc5_72_unit_func_ppc_mh_2( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_mh603e_addi( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_mh604e_addi( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_KKS2pipes( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_KKS604e( RC5_72UnitWork *, u32 *, void *);
# if defined(HAVE_ALTIVEC) /* OS+compiler support altivec */
extern "C" s32 CDECL rc5_72_unit_func_KKS7400( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_KKS7450( RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_KKS970( RC5_72UnitWork *, u32 *, void *);
# endif
#elif (CLIENT_CPU == CPU_CELLBE)
extern "C" s32 CDECL rc5_72_unit_func_cellv1_spe( RC5_72UnitWork *, u32 *, void *);
# if defined(HAVE_ALTIVEC) /* OS+compiler support altivec */
extern "C" s32 CDECL rc5_72_unit_func_cellv1_ppe( RC5_72UnitWork *, u32 *, void *);
# endif
#elif (CLIENT_CPU == CPU_SPARC)
extern "C" s32 CDECL rc5_72_unit_func_KKS_2 ( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_anbe_1( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_anbe_2( RC5_72UnitWork *, u32 *, void * );
#elif (CLIENT_CPU == CPU_MIPS)
extern "C" s32 CDECL rc5_72_unit_func_MIPS_2 ( RC5_72UnitWork *, u32 *, void * );
#elif (CLIENT_CPU == CPU_CUDA)
extern "C" s32 CDECL rc5_72_unit_func_cuda_1_64( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_cuda_1_128( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_cuda_1_256( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_cuda_2_64( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_cuda_2_128( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_cuda_2_256( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_cuda_4_64( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_cuda_4_128( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_cuda_4_256( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_cuda_1_64_bw( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_cuda_1_64_s0( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_cuda_1_64_s1( RC5_72UnitWork *, u32 *, void * );
#elif (CLIENT_CPU == CPU_ATI_STREAM)
extern "C" s32 CDECL rc5_72_unit_func_il4_nand( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_il4a_nand( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_il4_2t( RC5_72UnitWork *, u32 *, void * );
#endif


/* ======================================================================== */

int InitializeCoreTable_rc572(int /*first_time*/)
{
  /* rc5-72 does not require any initialization */
  return 0;
}

void DeinitializeCoreTable_rc572()
{
  /* rc5-72 does not require any initialization */
}

/* ======================================================================== */


const char **corenames_for_contest_rc572()
{
  /*
   When selecting corenames, use names that describe how (what optimization)
   they are different from their predecessor(s). If only one core,
   use the obvious "MIPS optimized" or similar.
  */
  static const char *corenames_table[]=
  /* ================================================================== */
  {
  #if (CLIENT_CPU == CPU_X86)
      #if !defined(HAVE_NO_NASM)
      "SES 1-pipe",
      "SES 2-pipe",
      "DG 2-pipe",
      "DG 3-pipe",
      "DG 3-pipe alt",
      "SS 2-pipe",
      "GO 2-pipe",
      "SGP 3-pipe",
      "MA 4-pipe",
      "MMX 4-pipe",
      "GO 2-pipe alt",
      "GO 2-pipe b",
      #else /* no nasm -> only ansi cores */
      "ANSI 4-pipe",
      "ANSI 2-pipe",
      "ANSI 1-pipe",
      #endif
  #elif (CLIENT_CPU == CPU_AMD64)
      "SNJL 3-pipe",
      "KBE-64 3-pipe",
  #elif (CLIENT_CPU == CPU_ARM)
      "StrongARM 1-pipe",
      "ARM 2/3/6/7 1-pipe",
      "XScale 1-pipe",
  #elif (CLIENT_CPU == CPU_S390X)
      "ANSI 4-pipe",
      "ANSI 2-pipe",
      "ANSI 1-pipe",
      "ANSI 4-pipe gcc32",
      "ANSI 2-pipe gcc32",
      "ANSI 1-pipe gcc32",
  #elif (CLIENT_CPU == CPU_68K) && (defined(__GCC__) || defined(__GNUC__))
      "MH 1-pipe 68020/030",
      "MH 1-pipe 68000/040",
      "MH 2-pipe 68060",
  #elif (CLIENT_CPU == CPU_POWERPC) && (CLIENT_OS != OS_WIN32)
      "MH 2-pipe",     /* gas, TOC and OSX formats */
      "KKS 2-pipe",    /* gas, TOC and OSX formats */
      "KKS 604e",      /* gas, TOC and OSX format */
      "KKS 7400",      /* gas and OSX format, AltiVec only */
      "KKS 7450",      /* gas and OSX format, AltiVec only */
      "MH 1-pipe",     /* gas, TOC and OSX format */
      "MH 1-pipe 604e",/* gas, TOC and OSX format */
      #if 0     // Disabled (kakace)
      "KKS 970",       /* gas and OSX format, AltiVec only */
      #endif
  #elif (CLIENT_CPU == CPU_CELLBE)
      "Cell v1 PPE",
      "Cell v1 SPE",
  #elif (CLIENT_CPU == CPU_SPARC)
      "ANSI 4-pipe",
      "ANSI 2-pipe",
      "ANSI 1-pipe",
      "KKS 2-pipe",
      "AnBe 1-pipe",
      "AnBe 2-pipe",
  #elif (CLIENT_CPU == CPU_MIPS)
      "ANSI 4-pipe",
      "ANSI 2-pipe",
      "ANSI 1-pipe",
      "MIPS 2-pipe",
  #elif (CLIENT_CPU == CPU_CUDA)
      "CUDA 1-pipe 64-thd",
      "CUDA 1-pipe 128-thd",
      "CUDA 1-pipe 256-thd",
      "CUDA 2-pipe 64-thd",
      "CUDA 2-pipe 128-thd",
      "CUDA 2-pipe 256-thd",
      "CUDA 4-pipe 64-thd",
      "CUDA 4-pipe 128-thd",
      "CUDA 4-pipe 256-thd",
      "CUDA 1-pipe 64-thd busy wait",
      "CUDA 1-pipe 64-thd sleep 100us",
      "CUDA 1-pipe 64-thd sleep dynamic",
  #elif (CLIENT_CPU == CPU_ATI_STREAM)
      "IL 4-pipe c",
      "IL 4-pipe c alt",
      "IL 4-pipe 2 threads",
  #else
      "ANSI 4-pipe",
      "ANSI 2-pipe",
      "ANSI 1-pipe",
  #endif
      NULL
  };
  /* ================================================================== */
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
int apply_selcore_substitution_rules_rc572(int cindex)
{
#if (CLIENT_CPU == CPU_POWERPC)
  int have_vec = 0;

# if defined(HAVE_ALTIVEC)
  /* OS+compiler support altivec */
  have_vec = GetProcessorFeatureFlags() & CPU_F_ALTIVEC;
# endif

  if (!have_vec && cindex == 3)   /* KKS 7400 */
    cindex = 1;                   /* KKS 2pipes */
  if (!have_vec && cindex == 4)   /* KKS 7450 */
    cindex = 2;                   /* KKS 604e */
  if (!have_vec && cindex == 7)   /* KKS 970 */
    cindex = 1;                   /* KKS 2pipes, see micro-bench in #3310 */

#elif (CLIENT_CPU == CPU_X86)
  {
    long det = GetProcessorType(1);
    unsigned flags = GetProcessorFeatureFlags();
    int have_3486 = (det >= 0 && (det & 0xff)==1);

    #if !defined(HAVE_NO_NASM)
      if (have_3486 && cindex >= 2)     /* dg-* cores use the bswap instr that's not available on 386 */
        cindex = 0;                     /* "SES 1-pipe" */
    #endif

    if ( (flags & (CPU_F_MMX | CPU_F_AMD_MMX_PLUS)) != (CPU_F_MMX | CPU_F_AMD_MMX_PLUS) &&
         (flags & (CPU_F_MMX | CPU_F_SSE))          != (CPU_F_MMX | CPU_F_SSE)              ) {
      if (cindex == 6 || cindex == 11) {  /* GO2 and GO2-B cores requires extended MMX */
        cindex = 1;      /* default core */
      }
    }

    if ((flags & (CPU_F_SSE | CPU_F_SSE2)) != (CPU_F_SSE | CPU_F_SSE2)) {
      if (cindex == 8) {  /* MA4 core requires SSE2 */
        cindex = 1;     /* default core */
      }
    }

    if (!(flags & CPU_F_MMX)) {
      if (cindex == 9) {  /* MMX core requires MMX */
        cindex = 1;     /* default core */
      }
    }
  }
#elif (CLIENT_CPU == CPU_CUDA)
  // GetProcessorType() currently returns the number of registers
  if (GetProcessorType(1) < 256 * 36) {
    /* the 2/4-pipe cores require 36/35 registers per thread, so the 256-thread cores are only feasible on a GTX */
    if (cindex == 5 || cindex == 8) {
      cindex = 0;    /* default: 1-pipe 64-thread */
    }
  }
#endif

  return cindex;
}

/* ===================================================================== */

int selcoreGetPreselectedCoreForProject_rc572()
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
  #if (CLIENT_CPU == CPU_ALPHA)
    if (detected_type > 0)
    {
      if (detected_type >= 7 /*EV56 and higher*/)
        cindex = 0;
      else if ((detected_type <= 4 /*EV4 and lower*/) ||
              (detected_type == 6 /*EV45*/))
        cindex = 2;
      else /* EV5 */
        cindex = -1;
    }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_68K)
    if (detected_type > 0)
    {
      #if defined(__GCC__) || defined(__GNUC__)
      switch (detected_type)
      {
        case 68020:
        case 68030:
          cindex = 0; /* 030 optimized (best for 68020 too) */
          break;
        case 68000:
        case 68040:
          cindex = 1; /* 040 optimized (best for 68000 too) */
          break;
        case 68060:
          cindex = 2; /* 060 optimized */
          break;
      }
      #else
      cindex = 2; /* ANSI 1-pipe */
      #endif
    }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_POWERPC)
    if (detected_type > 0)
    {
      switch ( detected_type & 0xffff) // only compare the low PVR bits
      {
        case 0x0001: cindex = 5; break; // 601            == MH 1-pipe
        case 0x0003: cindex = 5; break; // 603            == MH 1-pipe
        case 0x0004: cindex = 6; break; // 604            == MH 1-pipe 604e
        case 0x0006: cindex = 5; break; // 603e           == MH 1-pipe
        case 0x0007: cindex = 5; break; // 603r/603ev     == MH 1-pipe
        case 0x0008: cindex = 5; break; // 740/750 (G3)   == MH 1-pipe
        case 0x0009: cindex = 6; break; // 604e           == MH 1-pipe 604e
        case 0x000A: cindex = 6; break; // 604ev          == MH 1-pipe 604e
        case 0x0081: cindex = 5; break; // 8240           == MH 1-pipe
        case 0x4012: cindex = 1; break; // 440GP          == KKS 2pipes
        case 0x4222: cindex = 1; break; // 440EP/440GR    == KKS 2pipes
        case 0x51B2: cindex = 1; break; // 440GX          == KKS 2pipes
        case 0x5322: cindex = 1; break; // 440SP          == KKS 2pipes
        case 0x7000: cindex = 5; break; // 750FX          == MH 1-pipe
        case 0x8081: cindex = 5; break; // 5200 (G2)      == MH 1-pipe
        case 0x8082: cindex = 5; break; // 5200-LE (G2)   == MH 1-pipe
        /* non-altivec defaults, if no OS support */
        case 0x000C: cindex = 0; break; // 7400 (G4)      == MH 2-pipe
        case 0x8000: cindex = 2; break; // 7450 (G4+)     == KKS 604e
        case 0x8001: cindex = 2; break; // 7455 (G4+)     == KKS 604e
        case 0x8002: cindex = 2; break; // 7447/7457 (G4+) == KKS 604e
        case 0x8003: cindex = 2; break; // 7447A (G4+)    == KKS 604e
        case 0x8004: cindex = 2; break; // 7448 (G4+)     == KKS 604e
        case 0x800C: cindex = 0; break; // 7410 (G4)      == MH 2-pipe
        case 0x0039: cindex = 1; break; // 970 (G5)       == KKS 2pipes
        case 0x003C: cindex = 1; break; // 970FX (G5)     == KKS 2pipes
        case 0x0044: cindex = 1; break; // 970MP (G5)     == KKS 2pipes
        default:     cindex =-1; break; // no default
      }

      #if defined(HAVE_ALTIVEC) /* OS+compiler support altivec */
      // Note : KKS 7540 (AltiVec) is now set as default for any unknown CPU ID
      // since new CPUs are likely to be improved G4/G5 class CPUs.
      if ((detected_flags & CPU_F_ALTIVEC) != 0) //altivec?
      {
        switch ( detected_type & 0xffff) // only compare the low PVR bits
        {
            case 0x000C: cindex = 3; break; // 7400 (G4)   == KKS 7400
            case 0x8000: cindex = 4; break; // 7450 (G4+)  == KKS 7450
            case 0x8001: cindex = 4; break; // 7455 (G4+)  == KKS 7450
            case 0x8002: cindex = 4; break; // 7457/7447 (G4+)  == KKS 7450
            case 0x8003: cindex = 4; break; // 7447A (G4+)  == KKS 7450
            case 0x8004: cindex = 4; break; // 7448 (G4+)  == KKS 7450
            case 0x800C: cindex = 3; break; // 7410 (G4)   == KKS 7400
            #if 0       // Disabled (kakace)
            case 0x0039: cindex = 7; break; // 970 (G5)    == KKS 970
            case 0x003C: cindex = 7; break; // 970 FX
            case 0x0044: cindex = 7; break; // 970 MP
            #else
            case 0x0039: cindex = 4; break; // Redirect G5 to KKS 7450
            case 0x003C: cindex = 4; break; // Ditto (970FX)
            case 0x0044: cindex = 4; break; // Ditto (970MP)
            #endif
            default:     cindex = 4; break; // KKS 7450
        }
      }
      #endif
    }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_CELLBE)
    if (detected_type > 0)
    {
      switch ( detected_type & 0xffff) // only compare the low PVR bits
      {
        case 0x0070: cindex = 1; break; // Cell BE        == Cell SPE
        default:     cindex =-1; break; // no default
      }

      #if defined(HAVE_ALTIVEC) /* OS+compiler support altivec */
      if ((detected_flags & CPU_F_ALTIVEC) != 0) //altivec?
      {
        switch ( detected_type & 0xffff) // only compare the low PVR bits
        {
            case 0x0070: cindex = 0; break; // Cell BE
            default:     cindex = 0; break; // KKS 7450
        }
      }
      #endif
    }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_X86)
  {
    int have_mmx = (GetProcessorFeatureFlags() & CPU_F_MMX);
      if (detected_type >= 0)
      {
        #if !defined(HAVE_NO_NASM)
        unsigned long hints = GetProcessorCoreHints();
        if (hints & CH_R72_X86_GO2B) /* force go-2b */
          cindex = 11;
        else switch (detected_type)
        {
          case 0x00: cindex = (have_mmx?9   // P5 MMX     == MMX 4-pipe 
                                       :2); // P5         == DG 2-pipe
		                 break;
          case 0x01: cindex = 0; break; // 386/486        == SES 1-pipe
          case 0x02: cindex =10; break; // Pentium II     == GO 2-pipe-a
          case 0x03: cindex = 7; break; // Cyrix Model 4  == SGP 3-pipe (#3665)
          case 0x04: cindex = 7; break; // K5             == SGP 3-pipe
          case 0x05: cindex = 9; break; // K6             == MMX 4-pipe (#3863)
          case 0x06: cindex = 0; break; // Cx486          == SES 1-pipe
          case 0x07: cindex =-1; break; // orig Celeron   == unused?
          case 0x08: cindex = 1; break; // PPro           == SES 2-pipe (#3708)
          case 0x09: cindex = 6; break; // K7/K8          == GO 2-pipe
          case 0x0A: cindex = 5; break; // Centaur C6     == SS 2-pipe (#3809)
//        case 0x0B: cindex = 6; break; // Most Pentium 4 == GO 2-pipe (#3960, #3265)
          case 0x0B: cindex =10; break; // Most Pentium 4 == GO 2-pipe-a (new)
          case 0x0C: cindex = 4; break; // Via C3         == DG 3-pipe alt (#3477)
          case 0x0D: cindex = 6; break; // Pentium M      == GO 2-pipe (#3870)
          case 0x0E: cindex = 6; break; // Pentium III    == GO 2-pipe (#3602)
          case 0x0F: cindex = 7; break; // Via C3 Nehemiah == SGP 3-pipe (#3621)
          case 0x10: cindex = 5; break; // Cyrix Model 5  == SS 2-pipe (#3580)
          case 0x11: cindex = 4; break; // Cyrix Model 6  == DG 3-pipe alt (#3809)
          case 0x12: cindex =11; break; // Intel Core 2   == GO 2-pipe-b (#4193)
          case 0x13: cindex = 7; break; // Other Pentium 4 == SGP 3-pipe
          case 0x14: cindex = 6; break; // Intel Atom     == GO 2-pipe (#4080)
          case 0x15: cindex =11; break; // Intel Core i7  == GO 2-pipe-b (#4193)
          case 0x16: cindex = 6; break; // AMD Opteron
          case 0x17: cindex = 7; break; // Variation of 0x13 with another OGR-NG core (#4186)
          default:   cindex =-1; break; // no default
        }
        #else
        switch (detected_type)
        {
          case 0x00: cindex = 2; break; // P5             == ANSI 1-pipe
          case 0x01: cindex = 2; break; // 386/486        == ANSI 1-pipe
          case 0x02: cindex = 1; break; // Pentium II     == ANSI 2-pipe
          case 0x03: cindex = 2; break; // Cyrix Model 4  == ANSI 1-pipe
          case 0x04: cindex = 2; break; // K5             == ANSI 1-pipe
          case 0x05: cindex = 1; break; // K6             == ANSI 2-pipe
          case 0x06: cindex = 2; break; // Cx486          == ANSI 1-pipe
          case 0x07: cindex =-1; break; // orig Celeron   == unused?
          case 0x08: cindex =-1; break; // PPro           == ?
          case 0x09: cindex = 0; break; // K7/K8          == ANSI 4-pipe
          case 0x0A: cindex =-1; break; // Centaur C6     == ?
          case 0x0B: cindex = 0; break; // Pentium 4      == ANSI 4-pipe
          case 0x0C: cindex =-1; break; // Via C3         == ?
          case 0x0D: cindex =-1; break; // Pentium M      == ?  
          case 0x0E: cindex =-1; break; // Pentium III    == ?
          case 0x0F: cindex =-1; break; // Via C3 Nehemiah == ?
          case 0x10: cindex =-1; break; // Cyrix Model 5  == ?
          case 0x11: cindex =-1; break; // Cyrix Model 6  == ?
          case 0x12: cindex =-1; break; // Intel Core 2   == ?
          case 0x13: cindex =-1; break; // Other Pentium 4 == ?
          case 0x14: cindex =-1; break; // Intel Atom     == ?
          case 0x15: cindex =-1; break; // Intel Core i7  == ?
          default:   cindex =-1; break; // no default
        }
        #endif
      }
  }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_AMD64)
  {
    if (detected_type >= 0)
    {
      switch (detected_type)
      {
        case 0x09: cindex = 1; break; // K8               == KBE-64 3-pipe
        case 0x0B: cindex = 1; break; // Pentium 4        == KBE-64 3-pipe
        case 0x12: cindex = 1; break; // Core 2           == KBE-64 3-pipe
        case 0x14: cindex = 1; break; // Atom             == KBE-64 3-pipe
        case 0x15: cindex = 0; break; // Intel Core i7    == SNJL 3-pipe (#3817)
        case 0x16: cindex = 0; break; // ??? need more info about other CPUs! AMD Athlon (Model 16) == SNJL 3-pipe (#4223)
        default:   cindex =-1; break; // no default
      }
    }
  }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_ARM)
    {
      extern signed int default_r72_core;

      cindex = default_r72_core;
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
  #elif (CLIENT_CPU == CPU_SPARC)
    #if (CLIENT_OS == OS_SOLARIS)
    if (detected_type > 0)
    {
      switch (detected_type)
      {
        case  1: cindex = 4; break; // SPARCstation SLC == AnBe 1-pipe
        case  2: cindex = 4; break; // SPARCstation ELC == AnBe 1-pipe
        case  3: cindex = 4; break; // SPARCstation IPC == AnBe 1-pipe
        case  4: cindex = 4; break; // SPARCstation IPX == AnBe 1-pipe
        case  5: cindex = 4; break; // SPARCstation 1   == AnBe 1-pipe
        case  6: cindex = 4; break; // SPARCstation 1+  == AnBe 1-pipe
        case  7: cindex = 4; break; // SPARCstation 2   == AnBe 1-pipe
        case  8: cindex = 4; break; // microSPARC       == AnBe 1-pipe
        case  9: cindex = 4; break; // microSPARC II    == AnBe 1-pipe
        case 10: cindex = 4; break; // TurboSPARC       == AnBe 1-pipe
        case 11: cindex = 5; break; // hyperSPARC       == AnBe 2-pipe (#3797)
        case 12: cindex = 5; break; // SuperSPARC       == AnBe 2-pipe
        case 13: cindex = 5; break; // SuperSPARC SC    == AnBe 2-pipe
        case 14: cindex = 5; break; // SuperSPARC II    == AnBe 2-pipe
        case 15: cindex = 5; break; // SuperSPARC II SC == AnBe 2-pipe
        case 16: cindex = 5; break; // UltraSPARC-I     == AnBe 2-pipe
        case 17: cindex = 5; break; // UltraSPARC-II    == AnBe 2-pipe
        case 18: cindex = 5; break; // UltraSPARC-IIe   == AnBe 2-pipe
        case 19: cindex = 5; break; // UltraSPARC-IIi   == AnBe 2-pipe
        case 20: cindex = 5; break; // UltraSPARC-III   == AnBe 2-pipe
        case 21: cindex = 5; break; // UltraSPARC-IIIi  == AnBe 2-pipe
        case 22: cindex = 5; break; // UltraSPARC-IV    == AnBe 2-pipe
        case 23: cindex = 4; break; // UltraSPARC-T1/T2 == AnBe 1-pipe
        case 24: cindex = 5; break; // SPARC64-IV       == AnBe 2-pipe (#3225)
        case 25: cindex = 3; break; // SPARC64-V        == KKS 2-pipe (#3225)
        case 26: cindex =-1; break; // SPARC64-VI       == no default
        default: cindex =-1; break; // no default
      }
    }
    #else /* non-Solaris */
    /* cpu detection and core preselection for all the SPARC OSes needs a
       complete overhaul and generalization
       sparc v8: impl = %psr[31..28], vers = %psr[27..24]
       sparc v9: manufacturer = %ver[63..48], impl = %ver[47..32]
       should be used for the cpu id number (detected_type) and preselection
       derived from this value
    */
    #endif
  // ===============================================================
  #elif (CLIENT_OS == OS_PS2LINUX) // OUCH !!!! SELECT_BY_CPU !!!!! FIXME
  #error Please make this decision by CLIENT_CPU!
  #error CLIENT_OS may only be used for sub-selects (only if neccessary)
    cindex = 1; // now we use ansi-2pipe
  #elif (CLIENT_CPU == CPU_CUDA)
    cindex = 0; // 1-pipe 64-threads
  #elif (CLIENT_CPU == CPU_ATI_STREAM)
    switch (detected_type)
    {
      case 0: 		    // R600 GPU ISA
      case 1:		    // R610 GPU ISA
      case 2:		    // R630 GPU ISA
      case 3:		    // R670 GPU ISA
       cindex = 1;	    // IL 4-pipe alt
       break;

      case 4: 		    // R700 class GPU ISA
      case 5: 		    // RV770 GPU ISA
      case 6: 		    // RV710 GPU ISA
      case 7: 		    // RV730 GPU ISA
      case 8: 		    // RV830 GPU ISA
      case 9: 		    // RV870 GPU ISA
       cindex = 0;	    // IL 4-pipe
       break;
    
      default:
      cindex = 0; 	    // IL 4-pipe
    }
  #endif

  return cindex;
}


/* ---------------------------------------------------------------------- */

int selcoreSelectCore_rc572(unsigned int threadindex,
                            int *client_cpuP, struct selcore *selinfo)
{
  int use_generic_proto = 0; /* if rc5/des unit_func proto is generic */
  unit_func_union unit_func; /* declared in problem.h */
  int cruncher_is_asynchronous = 0; /* on a co-processor or similar */
  int pipeline_count = 2; /* most cases */
  int client_cpu = CLIENT_CPU; /* usual case */
  int coresel = selcoreGetSelectedCoreForContest(RC5_72);
#if (CLIENT_CPU == CPU_CELLBE)
  // Each Cell has 1 PPE, which is dual-threaded (so in fact the OS sees 2
  // processors), but it has been found that running 2 simultaneous threads
  // degrades performance, so let's pretend there's only one PPE.

  // Threads with threadindex = 0..PPE_count-1 will be scheduled on the PPEs
  // (core 0); the rest are scheduled on the SPEs (core 1).
  if (threadindex >= (unsigned)GetNumberOfPhysicalProcessors())
    coresel = 1;
#else
  DNETC_UNUSED_PARAM(threadindex);
#endif

  if (coresel < 0)
    return -1;
  memset( &unit_func, 0, sizeof(unit_func));

  /* -------------------------------------------------------------- */

  {
    use_generic_proto = 1;
    switch (coresel)
    {
     /* architectures without ansi cores */
     #if (CLIENT_CPU == CPU_ARM)
      case 0:
      default:
        unit_func.gen_72 = rc5_72_unit_func_arm1;
        pipeline_count = 1;
        coresel = 0;
        break;
      case 1:
        unit_func.gen_72 = rc5_72_unit_func_arm2;
        pipeline_count = 1;
        break;
      case 2:
        unit_func.gen_72 = rc5_72_unit_func_arm3;
        pipeline_count = 1;
        break;
     // -----------
     #elif (CLIENT_CPU == CPU_68K) && (defined(__GCC__) || defined(__GNUC__))
      case 0:
        unit_func.gen_72 = rc5_72_unit_func_030_mh_1;
        pipeline_count = 1;
        break;
      case 1:
      default:
        unit_func.gen_72 = rc5_72_unit_func_040_mh_1;
        pipeline_count = 1;
        coresel = 1;
        break;
      case 2:
        unit_func.gen_72 = rc5_72_unit_func_060_mh_2;
        pipeline_count = 2;
        break;
     // -----------
     #elif (CLIENT_CPU == CPU_X86) && !defined(HAVE_NO_NASM)
      case 0:
        unit_func.gen_72 = rc5_72_unit_func_ses;
        pipeline_count = 1;
        break;
      case 1:
      default:
        unit_func.gen_72 = rc5_72_unit_func_ses_2;
        pipeline_count = 2;
        coresel = 1;
        break;
      case 2:
        unit_func.gen_72 = rc5_72_unit_func_dg_2;
        pipeline_count = 2;
        break;
      case 3:
        unit_func.gen_72 = rc5_72_unit_func_dg_3;
        pipeline_count = 3;
        break;
      case 4:
        unit_func.gen_72 = rc5_72_unit_func_dg_3a;
        pipeline_count = 3;
        break;
      case 5:
        unit_func.gen_72 = rc5_72_unit_func_ss_2;
        pipeline_count = 2;
        break;
      case 6:
        unit_func.gen_72 = rc5_72_unit_func_go_2;
        pipeline_count = 2;
        break;
      case 7:
        unit_func.gen_72 = rc5_72_unit_func_sgp_3;
        pipeline_count = 3;
        break;
      case 8:
        unit_func.gen_72 = rc5_72_unit_func_ma_4;
        pipeline_count = 4;
        break;
      case 9:
        unit_func.gen_72 = rc5_72_unit_func_mmx;
        pipeline_count = 4;
        break;
      case 10:
        unit_func.gen_72 = rc5_72_unit_func_go_2a;
        pipeline_count = 2;
        break;
      case 11:
        unit_func.gen_72 = rc5_72_unit_func_go_2b;
        pipeline_count = 2;
        break;
     // -----------
     #elif (CLIENT_CPU == CPU_AMD64)
      case 0:
        unit_func.gen_72 = rc5_72_unit_func_snjl;
        pipeline_count = 3;
        break;
      case 1:
      default:
        unit_func.gen_72 = rc5_72_unit_func_kbe;
        pipeline_count = 3;
        break;
    // -----------
    #elif (CLIENT_CPU == CPU_POWERPC) && (CLIENT_OS != OS_WIN32)
      case 0:
          unit_func.gen_72 = rc5_72_unit_func_ppc_mh_2;
          pipeline_count = 2;
          break;
      case 1:
          unit_func.gen_72 = rc5_72_unit_func_KKS2pipes;
          pipeline_count = 2;
          break;
      case 2:
          unit_func.gen_72 = rc5_72_unit_func_KKS604e;
          pipeline_count = 2;
          break;
      #if defined(HAVE_ALTIVEC)
      case 3:
          unit_func.gen_72 = rc5_72_unit_func_KKS7400;
          pipeline_count = 4;
          break;
      case 4:
          unit_func.gen_72 = rc5_72_unit_func_KKS7450;
          pipeline_count = 4;
          break;
      #endif
      case 5:
      default:
        unit_func.gen_72 = rc5_72_unit_func_mh603e_addi;
        pipeline_count = 1;
        coresel = 5;
        break;
      case 6:
        unit_func.gen_72 = rc5_72_unit_func_mh604e_addi;
        pipeline_count = 1;
        break;
      #if defined(HAVE_ALTIVEC)
      #if 0     // Disabled (kakace)
      case 7:
          unit_func.gen_72 = rc5_72_unit_func_KKS970;
          pipeline_count = 4;
          break;
      #endif
      #endif
     // -----------
    #elif (CLIENT_CPU == CPU_CELLBE)
      #if defined(HAVE_ALTIVEC)
      case 0:
        unit_func.gen_72 = rc5_72_unit_func_cellv1_ppe;
        pipeline_count = 4;
        break;
      #endif
      case 1:
        unit_func.gen_72 = rc5_72_unit_func_cellv1_spe;
        pipeline_count = 16;
        break;
     // -----------
     #elif (CLIENT_CPU == CPU_CUDA)
      case 0:
      default:
        unit_func.gen_72 = rc5_72_unit_func_cuda_1_64;
        pipeline_count = 1;
        coresel = 0; // yes, we explicitly set coresel in the default case !
        break;
      case 1:
        unit_func.gen_72 = rc5_72_unit_func_cuda_1_128;
        pipeline_count = 1;
        break;
      case 2:
        unit_func.gen_72 = rc5_72_unit_func_cuda_1_256;
        pipeline_count = 1;
        break;
      case 3:
        unit_func.gen_72 = rc5_72_unit_func_cuda_2_64;
        pipeline_count = 2;
        break;
      case 4:
        unit_func.gen_72 = rc5_72_unit_func_cuda_2_128;
        pipeline_count = 2;
        break;
      case 5:
        unit_func.gen_72 = rc5_72_unit_func_cuda_2_256;
        pipeline_count = 2;
        break;
      case 6:
        unit_func.gen_72 = rc5_72_unit_func_cuda_4_64;
        pipeline_count = 4;
        break;
      case 7:
        unit_func.gen_72 = rc5_72_unit_func_cuda_4_128;
        pipeline_count = 4;
        break;
      case 8:
        unit_func.gen_72 = rc5_72_unit_func_cuda_4_256;
        pipeline_count = 4;
        break;
      case 9:
        unit_func.gen_72 = rc5_72_unit_func_cuda_1_64_bw;
        pipeline_count = 1;
        break;
      case 10:
        unit_func.gen_72 = rc5_72_unit_func_cuda_1_64_s0;
        pipeline_count = 1;
        break;
      case 11:
        unit_func.gen_72 = rc5_72_unit_func_cuda_1_64_s1;
        pipeline_count = 1;
        break;
     // -----------
     #elif (CLIENT_CPU == CPU_ATI_STREAM)
      case 0:             
      default:
        unit_func.gen_72 = rc5_72_unit_func_il4_nand;
        pipeline_count = 4;
      break;               
      case 1:             
        unit_func.gen_72 = rc5_72_unit_func_il4a_nand;
        pipeline_count = 4;
      break;               
      case 2:             
        unit_func.gen_72 = rc5_72_unit_func_il4_2t;
        pipeline_count = 4;
      break;               
    // -----------
     #else /* the ansi cores */
      case 0:
        unit_func.gen_72 = rc5_72_unit_func_ansi_4;
        pipeline_count = 4;
        break;
      case 1:
        unit_func.gen_72 = rc5_72_unit_func_ansi_2;
        pipeline_count = 2;
        break;
      case 2:
      default:
        unit_func.gen_72 = rc5_72_unit_func_ansi_1;
        pipeline_count = 1;
        coresel = 2; // yes, we explicitly set coresel in the default case !
        break;
     #endif

     // -----------
     /* additional cores */
     #if (CLIENT_CPU == CPU_SPARC)
       case 3:
         unit_func.gen_72 = rc5_72_unit_func_KKS_2;
         pipeline_count = 2;
         break;
       case 4:
         unit_func.gen_72 = rc5_72_unit_func_anbe_1;
         pipeline_count = 1;
         break;
       case 5:
         unit_func.gen_72 = rc5_72_unit_func_anbe_2;
         pipeline_count = 2;
         break;
     #endif
     // -----------
     #if (CLIENT_CPU == CPU_S390X)
       case 3:
         unit_func.gen_72 = rc5_72_unit_func_ansi_4_s390x_gcc32;
         pipeline_count = 4;
         break;
       case 4:
         unit_func.gen_72 = rc5_72_unit_func_ansi_2_s390x_gcc32;
         pipeline_count = 2;
         break;
       case 5:
         unit_func.gen_72 = rc5_72_unit_func_ansi_1_s390x_gcc32;
         pipeline_count = 1;
         break;
     #endif
     // -----------
     #if (CLIENT_CPU == CPU_MIPS)
       case 3:
         unit_func.gen_72 = rc5_72_unit_func_MIPS_2;
         pipeline_count = 2;
         break;
     #endif

    }
  }

  /* ================================================================== */

  if (coresel >= 0 && unit_func.gen_72 &&
      coresel < ((int)corecount_for_contest(RC5_72)) )
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

/*
** Estimate the maximum number of work units that can be processed on a daily
** basis.
*/
unsigned int estimate_nominal_rate_rc572()
{
  unsigned int rate = 0;   /* Unknown - Not available */

  #if (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_CELLBE)
    static long detected_type = -123;
    static int  cpu_count = 0;
    static unsigned long detected_flags = 0;
    static unsigned int  frequency = 0;
    unsigned int keyrate = 0;   /* keys/s/MHz */

    if (detected_type == -123) {
      detected_type  = GetProcessorType(1);
      detected_flags = GetProcessorFeatureFlags();
      frequency      = GetProcessorFrequency();
      cpu_count      = GetNumberOfDetectedProcessors();
    }

    if (detected_type > 0) {
      switch (detected_type & 0xffff) { // only compare the low PVR bits
        case 0x0001:      // 601
          keyrate = 1700; break;
        case 0x0003:      // 603
        case 0x0004:      // 604
        case 0x0006:      // 603e
        case 0x0007:      // 603r/603ev
        case 0x0008:      // 740/750
        case 0x0009:      // 604e
        case 0x000A:      // 604ev
        case 0x7000:      // 750FX
          keyrate = 3250; break;
        case 0x000C:      // 7400
        case 0x800C:      // 7410
          keyrate = (detected_flags & CPU_F_ALTIVEC) ? 9200: 3250; break;
        case 0x8000:      // 7450
        case 0x8001:      // 7455
        case 0x8002:      // 7457/7447
        case 0x8003:      // 7447A
        case 0x8004:      // 7448
          keyrate = (detected_flags & CPU_F_ALTIVEC) ? 10700 : 3500; break;
        case 0x0039:      // 970
        case 0x003C:      // 970FX
        case 0x0044:      // 970MP
        case 0x0070:      // Cell Broadband Engine
          keyrate = (detected_flags & CPU_F_ALTIVEC) ? 7500 : 2450; break;
      }

      if (cpu_count > 0)
        rate = (keyrate * frequency * cpu_count) / 49710;  /* 49710 = 2^32 / 86400 */
    }
  #endif

  return rate;
}

/* ------------------------------------------------------------- */

#endif  // HAVE_RC5_72_CORES
