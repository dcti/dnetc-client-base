/*
 * Copyright distributed.net 1998-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *core_rc5_cpp(void) {
return "@(#)$Id: core_rc5.cpp,v 1.2 2003/09/12 23:19:10 mweiser Exp $"; }

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
#if (CLIENT_CPU == CPU_X86) && defined(SMC)
  #if defined(__unix__)
    #include <sys/types.h>
    #include <sys/mman.h>
  #elif defined(USE_DPMI)
    extern "C" smc_dpmi_ds_alias_alloc(void);
    extern "C" smc_dpmi_ds_alias_free(void);
  #endif
  static int x86_smc_initialized = -1;
#endif

#if defined(HAVE_RC5_64_CORES)

/* ======================================================================== */

/* all the core prototypes
   note: we may have more prototypes here than cores in the client
   note2: if you need some 'cdecl' value define it in selcore.h to CDECL */


// available ANSI cores:
// 2 pipeline: rc5/ansi/rc5ansi_2-rg.cpp
//   extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
//   extern "C" s32 rc5_ansi_rg_unified_form( RC5UnitWork *work,
//                                    u32 *iterations, void *scratch_area );
// 1 pipeline: rc5/ansi/rc5ansi1-b2.cpp
//   extern "C" u32 rc5_ansi_1_b2_rg_unit_func( RC5UnitWork *, u32 iterations );

  // default ansi cores
  // rc5/ansi/rc5ansi1-b2.cpp
  extern "C" u32 rc5_ansi_1_b2_rg_unit_func( RC5UnitWork *, u32 iterations );
  // rc5/ansi/rc5ansi_2-rg.cpp
  extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
  // rc5/ansi/rc5ansi_4-rg.cpp
  extern "C" u32 rc5_unit_func_ansi_4_rg( RC5UnitWork *, u32 iterations );

#if (CLIENT_CPU == CPU_X86)
  extern "C" u32 CDECL rc5_unit_func_486( RC5UnitWork * , u32 iterations );
  extern "C" u32 CDECL rc5_unit_func_p5( RC5UnitWork * , u32 iterations );
  extern "C" u32 CDECL rc5_unit_func_p6( RC5UnitWork * , u32 iterations );
  extern "C" u32 CDECL rc5_unit_func_6x86( RC5UnitWork * , u32 iterations );
  extern "C" u32 CDECL rc5_unit_func_k5( RC5UnitWork * , u32 iterations );
  extern "C" u32 CDECL rc5_unit_func_k6( RC5UnitWork * , u32 iterations );
  extern "C" u32 CDECL rc5_unit_func_p5_mmx( RC5UnitWork * , u32 iterations );
  extern "C" u32 CDECL rc5_unit_func_k6_mmx( RC5UnitWork * , u32 iterations );
  extern "C" u32 CDECL rc5_unit_func_486_smc( RC5UnitWork * , u32 iterations );
  extern "C" u32 CDECL rc5_unit_func_k7( RC5UnitWork * , u32 iterations );
  extern "C" u32 CDECL rc5_unit_func_p7( RC5UnitWork *, u32 iterations );
#elif (CLIENT_CPU == CPU_ARM)
  extern "C" u32 rc5_unit_func_arm_1( RC5UnitWork * , u32 );
  extern "C" u32 rc5_unit_func_arm_2( RC5UnitWork * , u32 );
  extern "C" u32 rc5_unit_func_arm_3( RC5UnitWork * , u32 );
  #if (CLIENT_OS == OS_RISCOS) && defined(HAVE_X86_CARD_SUPPORT)
  extern "C" u32 rc5_unit_func_x86( RC5UnitWork * , u32 );
  #endif
#elif (CLIENT_CPU == CPU_PA_RISC)
  // rc5/parisc/parisc.cpp encapulates parisc.s, 2 pipelines
  // extern "C" u32 rc5_parisc_unit_func( RC5UnitWork *, u32 );
  // unused ???
#elif (CLIENT_CPU == CPU_MIPS)
  #if (CLIENT_OS == OS_SINIX) || (CLIENT_OS == OS_QNX) || \
      (CLIENT_OS == OS_PS2LINUX)
    //rc5/mips/mips-crunch.cpp or rc5/mips/mips-irix.S
    extern "C" u32 rc5_unit_func_mips_crunch( RC5UnitWork *, u32 );
  #endif
#elif (CLIENT_CPU == CPU_SPARC)
  #if (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS) || \
      (CLIENT_OS == OS_LINUX)
    //rc5/ultra/rc5-ultra-crunch.cpp
    extern "C" u32 rc5_unit_func_ultrasparc_crunch( RC5UnitWork * , u32 );
  #endif
#elif (CLIENT_CPU == CPU_68K)
  #if (CLIENT_OS != OS_NEXTSTEP) && (defined(__GCC__) || defined(__GNUC__)) || \
      (CLIENT_OS == OS_AMIGAOS)// || (CLIENT_OS == OS_MACOS)
    extern "C" u32 CDECL rc5_unit_func_000_010re( RC5UnitWork *, u32 );
    extern "C" u32 CDECL rc5_unit_func_020_030( RC5UnitWork *, u32 );
    extern "C" u32 CDECL rc5_unit_func_060re( RC5UnitWork *, u32 );
  #elif (CLIENT_OS == OS_MACOS)
    // rc5/68k/crunch.68k.a.o
    extern "C" u32 rc5_68k_crunch_unit_func( RC5UnitWork *, u32 );
  #endif
#elif (CLIENT_CPU == CPU_POWERPC)
  #if (CLIENT_OS != OS_WIN32) //NT has poor PPC assembly
    /* rc5/ppc/rc5_*.cpp
    ** although Be OS isn't supported on 601 machines and there is is
    ** no 601 PPC board for the Amiga, lintilla depends on allitnil,
    ** so we have both anyway, we may as well support both. */
    extern "C" u32 rc5_unit_func_allitnil_compat( RC5UnitWork *, u32 );
    extern "C" u32 rc5_unit_func_lintilla_compat( RC5UnitWork *, u32 );
    extern "C" u32 rc5_unit_func_lintilla_604_compat( RC5UnitWork *, u32 );
    #if defined(__VEC__) || defined(__ALTIVEC__) /* OS+compiler support altivec */
      extern "C" u32 rc5_unit_func_vec_compat( RC5UnitWork *, u32 );
      extern "C" u32 rc5_unit_func_vec_7450_compat( RC5UnitWork *, u32 );
    #endif
  #endif
#elif (CLIENT_CPU == CPU_ALPHA)
  #if (CLIENT_OS == OS_WIN32) /* little-endian asm */
    //rc5/alpha/rc5-alpha-nt.s
    extern "C" u32 rc5_unit_func_ntalpha_michmarc( RC5UnitWork *, u32 );
  #else
    //axp-bmeyer.cpp around axp-bmeyer.s
    extern "C" u32 rc5_unit_func_axp_bmeyer( RC5UnitWork *, u32 );
  #endif
#elif (CLIENT_CPU == CPU_UNKNOWN)   || (CLIENT_CPU == CPU_S390) || \
      (CLIENT_CPU == CPU_S390X)     || (CLIENT_CPU == CPU_IA64) || \
      (CLIENT_CPU == CPU_SH4)       || (CLIENT_CPU == CPU_88K)  || \
      (CLIENT_CPU == CPU_VAX)       || (CLIENT_CPU == CPU_POWER)
  // only use already prototyped ansi cores
#else
  #error "How did you get here?"
#endif


/* ======================================================================== */

int InitializeCoreTable_rc564(int first_time)
{
  DNETC_UNUSED_PARAM(first_time);

  #if (CLIENT_CPU == CPU_X86) && defined(SMC)
  {                      /* self-modifying code needs initialization */

    #if defined(USE_DPMI) && ((CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN16))
    /*
    ** Unlike all other targets, the dpmi based ones need to initialize on
    ** InitializeCoreTable(), and deinitialize on DeinitializeCoreTable()
    */
    if (x86_smc_initialized < 0) /* didn't fail before */
    {
      if (smc_dpmi_ds_alias_alloc() > 0)
        x86_smc_initialized = +1;
    }
    #elif defined(__unix__)
    if (x86_smc_initialized < 0) /* one shot */
    {
      char *addr = (char *)&rc5_unit_func_486_smc;
      addr -= (((unsigned long)addr) & (4096-1));
      if (mprotect( addr, 4096*3, PROT_READ|PROT_WRITE|PROT_EXEC )==0)
        x86_smc_initialized = +1;
    }
    #elif (CLIENT_OS == OS_NETWARE)
    if (x86_smc_initialized < 0) /* one shot */
    {
      if (GetFileServerMajorVersionNumber() <= 4)
        x86_smc_initialized = +1; /* kernel module, all pages are xrw */
    }
    #elif (CLIENT_OS == OS_WIN32)
    if (x86_smc_initialized < 0) /* one shot */
    {
      if (winGetVersion() < 2500) // SMC core doesn't run under WinXP/Win2K
      {
        HANDLE h = OpenProcess(PROCESS_VM_OPERATION,
                               FALSE,GetCurrentProcessId());
        if (h)
        {
          DWORD old = 0;
          if (VirtualProtectEx(h, rc5_unit_func_486_smc, 4096*3,
                                   PAGE_EXECUTE_READWRITE, &old))
            x86_smc_initialized = +1;
          CloseHandle(h);
        }
      }
    }
    #endif
    if (x86_smc_initialized < 0)
      x86_smc_initialized = 0;
  }
  #endif /* ifdef SMC */

  return 0;
}


void DeinitializeCoreTable_rc564()
{
  #if (CLIENT_CPU == CPU_X86) && defined(SMC)
  {          /* self-modifying code may need deinitialization */
    #if defined(USE_DPMI) && ((CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN16))
    if (x86_smc_initialized > 0)
    {
      smc_dpmi_ds_alias_free();
      x86_smc_initialized = -1;
    }
    #endif
  }
  #endif /* ifdef SMC */
}


/* ======================================================================== */


const char **corenames_for_contest_rc564()
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
      /* we should be using names that tell us how the cores are different
         (just like "bryd" and "movzx bryd")
      */
      "RG/BRF class 5",        /* 0. P5/Am486 */
      "RG class 3/4",          /* 1. 386/486 */
      "RG class 6",            /* 2. PPro/II/III */
      "RG re-pair I",          /* 3. Cyrix 486/6x86[MX]/MI */
      "RG RISC-rotate I",      /* 4. K5 */
      "RG RISC-rotate II",     /* 5. K6 */
      "RG/HB re-pair II",      /* 6. K7 Athlon and Cx-MII, based on Cx re-pair */
      "RG/BRF self-mod",       /* 7. SMC */
      "AK class 7",            /* 8. P4 */
      "jasonp P5/MMX",         /* 9. P5/MMX *only* - slower on PPro+ */
  #elif (CLIENT_CPU == CPU_X86_64)
      "Generic RC5 core",
  #elif (CLIENT_CPU == CPU_ARM)
      "Series A", /* (autofor for ARM 3/6xx/7xxx) "ARM 3, 610, 700, 7500, 7500FE" */
      "Series B", /* (autofor ARM 8xx/StrongARM) "ARM 810, StrongARM 110, 1100, 1110" */
      "Series C", /* (autofor ARM 2xx) "ARM 2, 250" */
  #elif (CLIENT_CPU == CPU_68K)
      #if defined(__GCC__) || defined(__GNUC__) || \
          (CLIENT_OS == OS_AMIGAOS)// || (CLIENT_OS == OS_MACOS)
      "68000/010", /* 68000/010 */
      "68020/030", /* 68020/030 */
      "68040/060", /* 68040/060 */
      #else
      "Generic",
      #endif
  #elif (CLIENT_CPU == CPU_ALPHA)
      #if (CLIENT_OS == OS_WIN32)
      "Marcelais",
      #else
      "axp bmeyer",
      #endif
  #elif (CLIENT_CPU == CPU_POWERPC)
      /* lintilla depends on allitnil, and since we need both even on OS's
      ** that don't support the 601, we may as well "support" them visually.  */
      "allitnil",
      "lintilla",
      "lintilla-604",    /* Roberto Ragusa's core optimized for PPC 604e */
      "crunch-vec",      /* altivec only */
      "crunch-vec-7450", /* altivec only */
  #elif (CLIENT_CPU == CPU_SPARC)
      #if ((CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS) || (CLIENT_OS == OS_LINUX))
        "Ultrasparc RC5 core",
        "Generic RC5 core",
      #else
        "Generic RC5 core",
      #endif
  #elif (CLIENT_OS == OS_PS2LINUX)
      "Generic RC5 core",
      "mips-crunch RC5 core",
  #else
      "Generic RC5 core",
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
int apply_selcore_substitution_rules_rc564(int cindex)
{
#if (CLIENT_CPU == CPU_POWERPC)
  int have_vec = 0;

  /* OS+compiler support altivec */
# if defined(__VEC__) || defined(__ALTIVEC__)
  long det = GetProcessorType(1);
  have_vec = (det >= 0 && (det & 1L<<25)!=0); /* have altivec */
# endif

  /* no crunch-vec or crunch-vec-7450 on non-altivec machine */
  if (!have_vec && (cindex == 3 || cindex == 4))
    cindex = 1;                         /* force lintilla */
# elif (CLIENT_CPU == CPU_X86)
  long det = GetProcessorType(1);
  int have_mmx = (det >= 0 && (det & 0x100)!=0);
  int have_3486 = (det >= 0 && (det & 0xff)==1);
  int have_smc = 0;
  int have_nasm = 0;

# if !defined(HAVE_NO_NASM)
  have_nasm = 1;
# endif
# if defined(SMC)
  have_smc = (x86_smc_initialized > 0);
# endif

  if (!have_nasm && cindex == 6)        /* "RG/HB re-pair II" */
    cindex = ((have_3486 && have_smc)?(7):(3)); /* "RG self-mod" or
                                                   "RG/HB re-pair I" */
  if (!have_smc && cindex == 7)         /* "RG self-modifying" */
    cindex = 1;                         /* "RG class 3/4" */
  if (have_smc && cindex == 7 && GetManagedProblemCount() > 1)
                                        /* "RG self-modifying" */
    cindex = 1;                         /* "RG class 3/4" */
  if (!have_nasm && cindex == 8)        /* "AK Class 7" */
    cindex = 2;                         /* "RG Class 6" */
  if (!have_mmx && cindex == 9)         /* "jasonp P5/MMX" */
    cindex = 0;                         /* "RG Class 5" */
  if (!have_nasm && cindex == 9)        /* "jasonp P5/MMX" */
    cindex = 0;                         /* "RG Class 5" */
#endif

  return cindex;
}

/* -------------------------------------------------------------------- */

int selcoreGetPreselectedCoreForProject_rc564()
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
  #if (CLIENT_CPU == CPU_68K)
    if (detected_type > 0)
    {
      cindex = 0;
      #if defined(__GCC__) || defined(__GNUC__) || \
          (CLIENT_OS == OS_AMIGAOS)// || (CLIENT_OS == OS_MACOS)
      if (detected_type >= 68040)
        cindex = 2; /* rc5-060-re-jg.s */
      else if (detected_type >= 68020)
        cindex = 1; /* rc5-020_030-jg.s */
      #endif
    }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_POWERPC)
    if (detected_type > 0)
    {
      switch ( detected_type & 0xffff) // only compare the low PVR bits
      {
        case 0x0001: cindex = 0; break; // 601            == allitnil
        case 0x0003: cindex = 1; break; // 603            == lintilla
        case 0x0004: cindex = 2; break; // 604            == lintilla-604
        case 0x0006: cindex = 1; break; // 603e           == lintilla
        case 0x0007: cindex = 1; break; // 603r/603ev     == lintilla
        case 0x0008: cindex = 1; break; // 740/750 (G3)   == lintilla
        case 0x0009: cindex = 2; break; // 604e           == lintilla-604
        case 0x000A: cindex = 2; break; // 604ev          == lintilla-604
//        Let G4s do a minibench if AltiVec is unavailable
//        case 0x000C: cindex = 1; break; // 7400 (G4)      == lintilla
        default:     cindex =-1; break; // no default (used to be lintilla)
      }
      #if defined(__VEC__) || defined(__ALTIVEC__) /* OS+compiler support altivec */
      if (( detected_type & (1L<<25) ) != 0) //altivec?
        {
          switch ( detected_type & 0xffff) // only compare the low PVR bits
          {
            case 0x000C: cindex = 4; break; // 7400 (G4)   == crunch-vec
            case 0x8000: cindex = 5; break; // 7450 (G4+)  == crunch-vec-7450
            case 0x8001: cindex = 5; break; // 7455 (G4+)  == crunch-vec-7450
            case 0x800C: cindex = 4; break; // 7410 (G4)   == crunch-vec
            default:     cindex =-1; break; // no default
          }
        }
      #endif
    }
  // ===============================================================
  #elif (CLIENT_CPU == CPU_X86)

    int have_mmx = ((detected_flags & CPU_F_MMX) == CPU_F_MMX);

    if (detected_type >= 0)
      {
        int have_smc = 0;
        #if defined(SMC)
        have_smc = (x86_smc_initialized > 0);
        #endif
        switch (detected_type & 0xff) // FIXME remove &0xff
        {
          case 0x00: cindex = ((have_mmx)?(9):(0)); break; // P5 == RG/BRF class 5
          case 0x01: cindex = ((have_smc)?(7 /*#99*/):(6 /*#1939*/)); break; // 386/486
          case 0x02: cindex = 2; break; // PII/PIII   == RG class 6
          case 0x03: cindex = 6; break; // Cx6x86/MII == RG re-pair II
          case 0x04: cindex = 4; break; // K5         == RG RISC-rotate I
          case 0x05: cindex = 5; break; // K6-1/2/3   == RG RISC-rotate II
          case 0x06: cindex = ((have_smc)?(7 /*#99*/):(6 /*#804*/)); break; // cx486
          case 0x07: cindex = 2; break; // Celeron    == RG class 6
          case 0x08: cindex = 2; break; // PPro       == RG class 6
          case 0x09: cindex = 6; break; // AMD>=K7/Cx>MII == RG/HB re-pair II
          case 0x0A: cindex = 6; break; // Centaur C6 == RG/HB re-pair II (#2082)
          case 0x0B: cindex = 8; break; // P4         == ak/P4
          default:   cindex =-1; break; // no default
        }
        #if defined(HAVE_NO_NASM)
        if (cindex == 6)   /* ("RG/HB re-pair II") */
        {
          cindex = 3;      /* ("RG re-pair I") */
          if ((detected_type & 0xff) == 0x01) /* 386/486 */
            cindex = 1;    /* "RG class 3/4" */
        }
        if (cindex == 8)   /* "AK Class 7" */
          cindex = 2;      /* "RG class 6" */
        if (cindex == 9)   /* "jasonp P5/MMX" */
          cindex = 0;      /* "RG Class 5" */
        #endif
      }

  // ===============================================================
  #elif (CLIENT_CPU == CPU_ARM)

    if (detected_type > 0)
    {
      if (detected_type == 0x300  || /* ARM 3 */
          detected_type == 0x600  || /* ARM 600 */
          detected_type == 0x610  || /* ARM 610 */
          detected_type == 0x700  || /* ARM 700 */
          detected_type == 0x710  || /* ARM 710 */
          detected_type == 0x7500 || /* ARM 7500 */
          detected_type == 0x7500FE) /* ARM 7500FE */
        cindex = 0;
      else if (detected_type == 0x810 || /* ARM 810 */
               detected_type == 0xA10 || /* StrongARM 110 */
               detected_type == 0xA11 || /* StrongARM 1100 */
               detected_type == 0xB11)   /* StrongARM 1110 */
        cindex = 1;
      else if (detected_type == 0x200 || /* ARM 2 */
               detected_type == 0x250)   /* ARM 250 */
        cindex = 2;
    }

  // ===============================================================
  #elif (CLIENT_CPU == CPU_SPARC)

    #if (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS)
    cindex = 0; // ultra-crunch is faster on everything I found ...
    #elif (CLIENT_OS == OS_LINUX)
    // run micro benchmark
    #endif

  // ===============================================================
  #elif (CLIENT_OS == OS_PS2LINUX) // OUCH !!!! SELECT_BY_CPU !!!!! FIXME
  #error Please make this decision by CLIENT_CPU!
  #error CLIENT_OS may only be used for sub-selects (only if neccessary)

    cindex = 1; // now we use mips-crunch.cpp

  #endif

  return cindex;
}


/* ---------------------------------------------------------------------- */

int selcoreSelectCore_rc564(unsigned int threadindex,
                            int *client_cpuP, struct selcore *selinfo)
{
  int use_generic_proto = 0; /* if rc5/des unit_func proto is generic */
  unit_func_union unit_func; /* declared in problem.h */
  int cruncher_is_asynchronous = 0; /* on a co-processor or similar */
  int pipeline_count = 2; /* most cases */
  int client_cpu = CLIENT_CPU; /* usual case */
  int coresel = selcoreGetSelectedCoreForContest(RC5);

  DNETC_UNUSED_PARAM(threadindex);

  if (coresel < 0)
    return -1;
  memset( &unit_func, 0, sizeof(unit_func));

  /* -------------------------------------------------------------- */

    #if (CLIENT_CPU == CPU_UNKNOWN)
    {
      // rc5/ansi/rc5ansi_2-rg.cpp
      //xtern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
      unit_func.rc5 = rc5_unit_func_ansi_2_rg;
      pipeline_count = 2;
      coresel = 0;
    }
    #elif (CLIENT_CPU == CPU_ARM)
    {
      if (coresel == 0)
      {
        unit_func.rc5 = rc5_unit_func_arm_1;
        pipeline_count = 1;
      }
      else if (coresel == 1)
      {
        unit_func.rc5 = rc5_unit_func_arm_2;
        pipeline_count = 2;
      }
      else /* (coresel == 2, default) */
      {
        unit_func.rc5 = rc5_unit_func_arm_3;
        pipeline_count = 3;
        coresel = 2;
      }
      #if (CLIENT_OS == OS_RISCOS) && defined(HAVE_X86_CARD_SUPPORT)
      if (threadindex == 1 && /* threadindex 1 is reserved for x86 */
          GetNumberOfDetectedProcessors() > 1) /* have x86 card */
      {
        client_cpu = CPU_X86;
        unit_func.gen = rc5_unit_func_x86;
        use_generic_proto = 1; /* unit_func proto is generic */
        cruncher_is_asynchronous = 1; /* on a co-processor or similar */
        pipeline_count = 1;
        coresel = 0;
      }
      #endif
    }
    #elif (CLIENT_CPU == CPU_S390)
    {
      // rc5/ansi/rc5ansi_2-rg.cpp
      //xtern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
      unit_func.rc5 = rc5_unit_func_ansi_2_rg;
      pipeline_count = 2;
      coresel = 0;
    }
    #elif (CLIENT_CPU == CPU_S390X)
    {
      // rc5/ansi/rc5ansi_2-rg.cpp
      //xtern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
      unit_func.rc5 = rc5_unit_func_ansi_2_rg;
      pipeline_count = 2;
      coresel = 0;
    }
    #elif (CLIENT_CPU == CPU_IA64)
    {
      // rc5/ansi/rc5ansi_2-rg.cpp
      //xtern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
      unit_func.rc5 = rc5_unit_func_ansi_4_rg;
      pipeline_count = 4;
      coresel = 0;
    }
    #elif (CLIENT_CPU == CPU_PA_RISC)
    {
      // /rc5/parisc/parisc.cpp encapulates parisc.s, 2 pipelines
      //extern "C" u32 rc5_parisc_unit_func( RC5UnitWork *, u32 );
      unit_func.rc5 = rc5_unit_func_ansi_2_rg;
      pipeline_count = 2;
      coresel = 0;
    }
    #elif (CLIENT_CPU == CPU_SH4)
    {
      //xtern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 );
      unit_func.rc5 = rc5_unit_func_ansi_2_rg;
      pipeline_count = 2;
      coresel = 0;
    }
    #elif (CLIENT_CPU == CPU_88K) //OS_DGUX
    {
      // rc5/ansi/rc5ansi_2-rg.cpp
      //xtern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 );
      unit_func.rc5 = rc5_unit_func_ansi_2_rg;
      pipeline_count = 2;
      coresel = 0;
    }
    #elif (CLIENT_CPU == CPU_MIPS)
    {
      #if (CLIENT_OS == OS_ULTRIX) || (CLIENT_OS == OS_IRIX) || \
          (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_NETBSD) || \
          (CLIENT_OS == OS_QNX)
      {
        // rc5/ansi/rc5ansi_2-rg.cpp
        //xtern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 );
        unit_func.rc5 = rc5_unit_func_ansi_2_rg;
        pipeline_count = 2;
        coresel = 0;
      }
      #elif (CLIENT_OS == OS_SINIX) || (CLIENT_OS == OS_QNX)
      {
        //rc5/mips/mips-crunch.cpp or rc5/mips/mips-irix.S
        //xtern "C" u32 rc5_unit_func_mips_crunch( RC5UnitWork *, u32 );
        unit_func.rc5 = rc5_unit_func_mips_crunch;
        pipeline_count = 2;
        coresel = 0;
      }
      #elif (CLIENT_OS == OS_PS2LINUX)
      if (coresel == 0)
      {
        // rc5/ansi/rc5ansi_2-rg.cpp
        //xtern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 );
        unit_func.rc5 = rc5_unit_func_ansi_2_rg;
        pipeline_count = 2;
      }
      else  /* coresel=1 (now default, using mips-crunch) */
      {
        //rc5/mips/mips-crunch.cpp or rc5/mips/mips-irix.S
        //xtern "C" u32 rc5_unit_func_mips_crunch( RC5UnitWork *, u32 );
        unit_func.rc5 = rc5_unit_func_mips_crunch;
        pipeline_count = 2;
        coresel = 1;
      }
      #else
        #error "What's up, Doc?"
      #endif
    }
    #elif (CLIENT_CPU == CPU_SPARC)
    {
      #if ((CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS) || (CLIENT_OS == OS_LINUX))
      if (coresel == 0)
      {
        //rc5/ultra/rc5-ultra-crunch.cpp
        unit_func.rc5 = rc5_unit_func_ultrasparc_crunch;
        pipeline_count = 2;
      }
      else
      {
        // rc5/ansi/rc5ansi_2-rg.cpp
        unit_func.rc5 = rc5_unit_func_ansi_2_rg;
        pipeline_count = 2;
        coresel = 1;
      }
      #else
      {
        // rc5/ansi/rc5ansi_2-rg.cpp
        //xtern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 );
        unit_func.rc5 = rc5_unit_func_ansi_2_rg;
        pipeline_count = 2;
        coresel = 0;
      }
      #endif
    }
    #elif (CLIENT_CPU == CPU_68K)
    {
      #if (CLIENT_OS != OS_NEXTSTEP) && (defined(__GCC__) || defined(__GNUC__)) || \
          (CLIENT_OS == OS_AMIGAOS)// || (CLIENT_OS == OS_MACOS)
      {
        //xtern "C" u32 rc5_unit_func_000_010re( RC5UnitWork *, u32 );
        //xtern "C" u32 rc5_unit_func_020_030( RC5UnitWork *, u32 );
        //xtern "C" u32 rc5_unit_func_060re( RC5UnitWork *, u32 );
        pipeline_count = 2;
        if (coresel == 2)
          unit_func.rc5 = rc5_unit_func_060re;  // used for 040 too (faster)
        else if (coresel == 1)
          unit_func.rc5 = rc5_unit_func_020_030;
        else
        {
          pipeline_count = 2;
          unit_func.rc5 = rc5_unit_func_000_010re;
          coresel = 0;
        }
      }
      #elif (CLIENT_OS == OS_MACOS)
      {
        // rc5/68k/crunch.68k.a.o
        //xtern "C" u32 rc5_68k_crunch_unit_func( RC5UnitWork *, u32 );
        unit_func.rc5 = rc5_68k_crunch_unit_func;
        pipeline_count = 1; //the default is 2
        coresel = 0;
      }
      #else
      {
        // rc5/ansi/rc5ansi1-b2.cpp
        //xtern "C" u32 rc5_ansi_1_b2_rg_unit_func( RC5UnitWork *, u32 );
        unit_func.rc5 = rc5_ansi_1_b2_rg_unit_func;
        pipeline_count = 1; //the default is 2
        coresel = 0;
      }
      #endif
    }
    #elif (CLIENT_CPU == CPU_VAX)
    {
      // rc5/ansi/rc5ansi1-b2.cpp
      //xtern "C" u32 rc5_ansi_1_b2_rg_unit_func( RC5UnitWork *, u32 );
      unit_func.rc5 = rc5_ansi_1_b2_rg_unit_func;
      pipeline_count = 1; //the default is 2
      coresel = 0;
    }
    #elif (CLIENT_CPU == CPU_POWER)
    {
      /* rc5/ansi/rc5ansi_2-rg.cpp
      ** extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 ); */
      unit_func.rc5 = rc5_unit_func_ansi_2_rg; //POWER CPU
      pipeline_count = 2;
      coresel = 0;
    }
    #elif (CLIENT_CPU == CPU_POWERPC)
    {
      #if (CLIENT_OS == OS_WIN32)
      {
        //  rc5/ansi/rc5ansi_2-rg.cpp
        //  xtern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32  );
        unit_func.rc5 = rc5_unit_func_ansi_2_rg;
        pipeline_count = 2;
        coresel = 3;
      }
      #else
      {
        if (coresel == 0)         // G1 (PPC 601)
        {
          unit_func.rc5 = rc5_unit_func_allitnil_compat;
          pipeline_count = 1;
        }
        else if (coresel == 2)    // G2 (PPC 604/604e/604ev only)
        {
          unit_func.rc5 = rc5_unit_func_lintilla_604_compat;
          pipeline_count = 1;
        }
        #if defined(__VEC__) || defined(__ALTIVEC__)
        else if (coresel == 3)    // G4 (PPC 7400/7410)
        {
          unit_func.rc5 = rc5_unit_func_vec_compat;
          pipeline_count = 1;
        }
        else if (coresel == 4)    // G4 (PPC 7450)
        {
          unit_func.rc5 = rc5_unit_func_vec_7450_compat;
          pipeline_count = 1;
        }
        #endif
        else                      // the rest (G2/G3)
        {
          unit_func.rc5 = rc5_unit_func_lintilla_compat;
          pipeline_count = 1;
          coresel = 1;
        }
      }
      #endif
    }
    #elif (CLIENT_CPU == CPU_X86)
    {
      pipeline_count = 2; /* most cases */
      if (coresel == 0)
        unit_func.rc5 = rc5_unit_func_p5;
      else if (coresel == 1) // Intel 386/486
        unit_func.rc5 = rc5_unit_func_486;
      else if (coresel == 2) // Ppro/PII
        unit_func.rc5 = rc5_unit_func_p6;
      else if (coresel == 3) // 6x86(mx)
        unit_func.rc5 = rc5_unit_func_6x86;
      else if (coresel == 4) // K5
        unit_func.rc5 = rc5_unit_func_k5;
      else if (coresel == 5)
        unit_func.rc5 = rc5_unit_func_k6;
      #if !defined(HAVE_NO_NASM)
      else if (coresel == 6)
        unit_func.rc5 = rc5_unit_func_k7;
      #endif
      #if defined(SMC) /* plus first thread or benchmark/test cap */
      else if (coresel == 7 && x86_smc_initialized > 0 && threadindex == 0)
         unit_func.rc5 = rc5_unit_func_486_smc;
      #endif
      #if !defined(HAVE_NO_NASM)
      else if (coresel == 8) // RC5 P4 core is 3 pipelines
        { unit_func.rc5 = rc5_unit_func_p7; pipeline_count = 3; }
      #endif
      #if !defined(HAVE_NO_NASM)
      else if (coresel == 9)
        { unit_func.rc5 = rc5_unit_func_p5_mmx; pipeline_count = 4; }
      #endif
      /* no default since we already validated the 'coresel' */
    }
    #elif (CLIENT_CPU == CPU_ALPHA)
    {
      #if (CLIENT_OS == OS_WIN32) /* little-endian asm */
      {
        //rc5/alpha/rc5-alpha-nt.s
        //xtern "C" u32 rc5_unit_func_ntalpha_michmarc( RC5UnitWork *, u32 );
        unit_func.rc5 = rc5_unit_func_ntalpha_michmarc;
        pipeline_count = 2;
        coresel = 0;
      }
      #else
      {
        //axp-bmeyer.cpp around axp-bmeyer.s
        //xtern "C" u32 rc5_unit_func_axp_bmeyer( RC5UnitWork *, u32 );
        unit_func.rc5 = rc5_unit_func_axp_bmeyer;
        pipeline_count = 2;
        coresel = 0;
      }
      #endif
    }
    #else
    {
      #error "How did you get here?"
      coresel = -1;
    }
    #endif

  /* ================================================================== */

  if (coresel >= 0 && unit_func.gen &&
     coresel < ((int)corecount_for_contest(RC5)) )
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

#endif  // HAVE_RC5_64_CORES
