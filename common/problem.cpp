/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * -------------------------------------------------------------------
 * Eagleson's Law:
 *    Any code of your own that you haven't looked at for six or more
 *    months, might as well have been written by someone else.  (Eagleson
 *    is an optimist, the real number is more like three weeks.)
 * -------------------------------------------------------------------
*/
const char *problem_cpp(void) {
return "@(#)$Id: problem.cpp,v 1.136 1999/12/09 12:47:32 cyp Exp $"; }

/* ------------------------------------------------------------- */

#include "cputypes.h"
#include "baseincs.h"
#include "client.h"   //CONTEST_COUNT
#include "clitime.h"  //CliClock()
#include "logstuff.h" //LogScreen()
#include "probman.h"  //GetProblemPointerFromIndex()
#include "selcore.h"  //selcoreGetSelectedCoreForContest()
#include "cpucheck.h" //hardware detection
#include "console.h"  //ConOutErr
#include "triggers.h" //RaiseExitRequestTrigger()
#include "problem.h"  //ourselves
#if (CLIENT_OS == OS_RISCOS)
#include "../platforms/riscos/riscos_x86.h"
#endif

//#define STRESS_THREADS_AND_BUFFERS /* !be careful with this! */

/* ------------------------------------------------------------------- */

#if (CLIENT_CPU == CPU_X86)
  extern "C" u32 rc5_unit_func_486( RC5UnitWork * , u32 iterations );
  extern "C" u32 rc5_unit_func_p5( RC5UnitWork * , u32 iterations );
  extern "C" u32 rc5_unit_func_p6( RC5UnitWork * , u32 iterations );
  extern "C" u32 rc5_unit_func_6x86( RC5UnitWork * , u32 iterations );
  extern "C" u32 rc5_unit_func_k5( RC5UnitWork * , u32 iterations );
  extern "C" u32 rc5_unit_func_k6( RC5UnitWork * , u32 iterations );
  extern "C" u32 rc5_unit_func_p5_mmx( RC5UnitWork * , u32 iterations );
  extern "C" u32 rc5_unit_func_k6_mmx( RC5UnitWork * , u32 iterations );
  extern "C" u32 rc5_unit_func_486_smc( RC5UnitWork * , u32 iterations );
#elif (CLIENT_CPU == CPU_ARM)
  extern "C" u32 rc5_unit_func_arm_1( RC5UnitWork * , u32 );
  extern "C" u32 rc5_unit_func_arm_2( RC5UnitWork * , u32 );
  extern "C" u32 rc5_unit_func_arm_3( RC5UnitWork * , u32 );
#elif (CLIENT_CPU == CPU_S390)
  //rc5/ansi/2-rg.c
  extern "C" u32 rc5_ansi_2_rg_unit_func( RC5UnitWork *, u32 );
#elif (CLIENT_CPU == CPU_PA_RISC)
  // rc5/parisc/parisc.cpp encapulates parisc.s, 2 pipelines
  extern "C" u32 rc5_parisc_unit_func( RC5UnitWork *, u32 );
#elif (CLIENT_CPU == CPU_88K) //OS_DGUX
  //rc5/ansi/2-rg.c
  extern "C" u32 rc5_ansi_2_rg_unit_func( RC5UnitWork *, u32 );
#elif (CLIENT_CPU == CPU_MIPS)
  #if (CLIENT_OS == OS_ULTRIX) || (CLIENT_OS == OS_IRIX)
    //rc5/ansi/2-rg.c
    extern "C" u32 rc5_ansi_2_rg_unit_func( RC5UnitWork *, u32 );
  #elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_SINIX)
    //rc5/mips/mips-crunch.cpp or rc5/mips/mips-irix.S
    extern "C" u32 rc5_unit_func_mips_crunch( RC5UnitWork *, u32 );
  #else
    #error "What's up, Doc?"
  #endif
#elif (CLIENT_CPU == CPU_SPARC)
  #if (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS)
    //rc5/ultra/rc5-ultra-crunch.cpp
    extern "C" u32 rc5_unit_func_ultrasparc_crunch( RC5UnitWork * , u32 );
  #else
    //rc5/ansi/2-rg.c
    extern "C" u32 rc5_ansi_2_rg_unit_func( RC5UnitWork *, u32 );
  #endif
#elif (CLIENT_CPU == CPU_68K)
  #if (CLIENT_OS == OS_MACOS) || (CLIENT_OS == OS_AMIGAOS)
    // rc5/68k/rc5_68k_crunch.c around rc5/68k/rc5-0x0_0y0-jg.s
    extern "C" u32 rc5_unit_func_000_030( RC5UnitWork *, u32 );
    extern "C" u32 rc5_unit_func_040_060( RC5UnitWork *, u32 );
  #elif defined(__GCC__) || defined(__GNUC__) /* hpux, next, linux, sun3 */
    // rc5/68k/rc5_68k_gcc_crunch.c around rc5/68k/crunch.68k.gcc.s
    extern "C" u32 rc5_68k_crunch_unit_func( RC5UnitWork *, u32 );
  #else
    // rc5/ansi/rc5ansi1-b2.cpp
    extern "C" u32 rc5_ansi_1_b2_rg_unit_func( RC5UnitWork *, u32 );
  #endif
#elif (CLIENT_CPU == CPU_VAX)
  // rc5/ansi/rc5ansi1-b2.cpp
  extern "C" u32 rc5_ansi_1_b2_rg_unit_func( RC5UnitWork *, u32 );
#elif (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_POWER)
  #if (CLIENT_CPU == CPU_POWER) || defined(_AIXALL)
    // rc5/ansi/2-rg.c
    extern "C" u32 rc5_ansi_2_rg_unit_func( RC5UnitWork *, u32 );
  #endif
  #if (CLIENT_CPU == CPU_POWERPC) || defined(_AIXALL)
    #if (CLIENT_OS == OS_WIN32) //NT has poor PPC assembly
      //rc5/ansi/2-rg.c
      extern "C" u32 rc5_ansi_2_rg_unit_func( RC5UnitWork *, u32 );
      #define rc5_unit_func_lintilla_compat rc5_ansi_2_rg_unit_func
      #define rc5_unit_func_allitnil_compat rc5_ansi_2_rg_unit_func
      #define rc5_unit_func_vec_compat      rc5_ansi_2_rg_unit_func
    #else
      // rc5/ppc/rc5_*.cpp
      // although Be OS isn't supported on 601 machines and there is
      // is no 601 PPC board for the Amiga, lintilla depends on allitnil,
      // so we have both anyway, we may as well support both.
      extern "C" u32 rc5_unit_func_allitnil_compat( RC5UnitWork *, u32 );
      extern "C" u32 rc5_unit_func_lintilla_compat( RC5UnitWork *, u32 );
      #if (CLIENT_OS == OS_MACOS)
        extern "C" u32 rc5_unit_func_vec_compat( RC5UnitWork *, u32 );
      #else /* MacOS currently is the only one to support altivec cores */
        #define rc5_unit_func_vec_compat  rc5_unit_func_lintilla_compat
      #endif
    #endif
  #endif
#elif (CLIENT_CPU == CPU_ALPHA)
  #if (CLIENT_OS == OS_DEC_UNIX)
    //rc5/alpha/rc5-digital-unix-alpha-ev[4|5].cpp
    extern "C" u32 rc5_alpha_osf_ev4( RC5UnitWork *, u32 );
    extern "C" u32 rc5_alpha_osf_ev5( RC5UnitWork *, u32 );
  #elif (CLIENT_OS == OS_WIN32) /* little-endian asm */
    //rc5/alpha/rc5-alpha-nt.s
    extern "C" u32 rc5_unit_func_ntalpha_michmarc( RC5UnitWork *, u32 );
  #else
    //axp-bmeyer.cpp around axp-bmeyer.s
    extern "C" u32 rc5_unit_func_axp_bmeyer( RC5UnitWork *, u32 );
  #endif
#else
  #error "How did you get here?" 
#endif    

/* ------------------------------------------------------------- */

#if defined(HAVE_DES_CORES)
/* DES cores take the 'iterations_to_do', adjust it to min/max/nbbits
  and store it back in 'iterations_to_do'. all return 'iterations_done'.
*/   
#if (CLIENT_CPU == CPU_ARM)
   //des/arm/des-arm-wrappers.cpp
   extern u32 des_unit_func_slice_arm( RC5UnitWork * , u32 *iter, char *coremem );
   extern u32 des_unit_func_slice_strongarm(RC5UnitWork *, u32 *iter, char *coremem);
#elif (CLIENT_CPU == CPU_ALPHA) 
   #if (CLIENT_OS == OS_DEC_UNIX) && defined(DEC_UNIX_CPU_SELECT)
     extern u32 des_alpha_osf_ev4( RC5UnitWork * , u32 *iter, char *coremem );
     extern u32 des_alpha_osf_ev5( RC5UnitWork * , u32 *iter, char *coremem );
   #else
     //des/alpha/des-slice-dworz.cpp
     extern u32 des_unit_func_slice_dworz( RC5UnitWork * , u32 *iter, char *);
   #endif
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
#endif

/* ------------------------------------------------------------- */

#if defined(HAVE_OGR_CORES)
  extern CoreDispatchTable *ogr_get_dispatch_table(void);
#endif  

/* ------------------------------------------------------------- */

#if defined(HAVE_CSC_CORES)
  extern "C" s32 csc_unit_func_1k  ( RC5UnitWork *, u32 *iterations, void *membuff );
  extern "C" s32 csc_unit_func_1k_i( RC5UnitWork *, u32 *iterations, void *membuff );
  extern "C" s32 csc_unit_func_6b  ( RC5UnitWork *, u32 *iterations, void *membuff );
  extern "C" s32 csc_unit_func_6b_i( RC5UnitWork *, u32 *iterations, void *membuff );
#if (CLIENT_CPU == CPU_X86) && defined(MMX_CSC)
  extern "C" s32 csc_unit_func_6b_mmx ( RC5UnitWork *, u32 *iterations, void *membuff );
#endif
#endif

/* ------------------------------------------------------------- */
static int __core_picker(Problem *problem, unsigned int contestid)
{                               /* must return a valid core selection # */
  int coresel;
  problem->pipeline_count = 2; /* most cases */
  problem->client_cpu = CLIENT_CPU; /* usual case */

  coresel = selcoreGetSelectedCoreForContest( contestid );
  if (coresel < 0)
    return -1;

  if (contestid == RC5) /* avoid switch */
  {
    #if (CLIENT_CPU == CPU_ARM)
    if (coresel == 0)
    {
      problem->rc5_unit_func = rc5_unit_func_arm_1;
      problem->pipeline_count = 1;
    }
    else if (coresel == 1)
    {
      problem->rc5_unit_func = rc5_unit_func_arm_2;
      problem->pipeline_count = 2;
    }
    else /* (coresel == 2, default) */
    {
      problem->rc5_unit_func = rc5_unit_func_arm_3;
      problem->pipeline_count = 3;
      coresel = 2;
    }
    #elif (CLIENT_CPU == CPU_S390)
    {
      //rc5/ansi/2-rg.c
      //xtern "C" u32 rc5_ansi_2_rg_unit_func( RC5UnitWork *, u32 );
      problem->rc5_unit_func = rc5_ansi_2_rg_unit_func;
      problem->pipeline_count = 2;
      coresel = 0;
    }
    #elif (CLIENT_CPU == CPU_PA_RISC)
    {
      // /rc5/parisc/parisc.cpp encapulates parisc.s, 2 pipelines
      //xtern "C" u32 rc5_parisc_unit_func( RC5UnitWork *, u32 );
      problem->rc5_unit_func = rc5_parisc_unit_func;
      problem->pipeline_count = 2;
      coresel = 0;
    }
    #elif (CLIENT_CPU == CPU_88K) //OS_DGUX
    {
      //rc5/ansi/2-rg.c
      //xtern "C" u32 rc5_ansi_2_rg_unit_func( RC5UnitWork *, u32 );
      problem->rc5_unit_func = rc5_ansi_2_rg_unit_func;
      problem->pipeline_count = 2;
      coresel = 0;
    }
    #elif (CLIENT_CPU == CPU_MIPS)
    {
      #if (CLIENT_OS == OS_ULTRIX) || (CLIENT_OS == OS_IRIX)
      {
        //rc5/ansi/2-rg.c
        //xtern "C" u32 rc5_ansi_2_rg_unit_func( RC5UnitWork *, u32 );
        problem->rc5_unit_func = rc5_ansi_2_rg_unit_func;
        problem->pipeline_count = 2;
        coresel = 0;
      }
      #elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_SINIX)
      {
        //rc5/mips/mips-crunch.cpp or rc5/mips/mips-irix.S
        //xtern "C" u32 rc5_unit_func_mips_crunch( RC5UnitWork *, u32 );
        problem->rc5_unit_func = rc5_unit_func_mips_crunch;
        problem->pipeline_count = 2;
        coresel = 0;
      }  
      #else
        #error "What's up, Doc?"
      #endif
    }
    #elif (CLIENT_CPU == CPU_SPARC)
    {
      #if (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS)
      {
        //rc5/ultra/rc5-ultra-crunch.cpp
        //xtern "C" u32 rc5_unit_func_ultrasparc_crunch( RC5UnitWork * , u32 );
        rc5_unit_func = rc5_unit_func_ultrasparc_crunch;
        problem->pipeline_count = 1;
        coresel = 0;
      }
      #else
      {
        //rc5/ansi/2-rg.c
        //xtern "C" u32 rc5_ansi_2_rg_unit_func( RC5UnitWork *, u32 );
        rc5_unit_func = rc5_ansi_2_rg_unit_func;
        problem->pipeline_count = 2;
        coresel = 0;
      }
      #endif
    }
    #elif (CLIENT_CPU == CPU_68K)
    {
      #if (CLIENT_OS == OS_MACOS) || (CLIENT_OS == OS_AMIGAOS)
      {
        // rc5/68k/rc5_68k_crunch.c around rc5/68k/rc5-0x0_0y0-jg.s
        //xtern "C" u32 rc5_unit_func_000_030( RC5UnitWork *, u32 );
        //xtern "C" u32 rc5_unit_func_040_060( RC5UnitWork *, u32 );
        if (coresel == 1 )
        {
          problem->pipeline_count = 2;
          problem->rc5_unit_func = rc5_unit_func_040_060;
          coresel = 1;
        }
        else
        {
          problem->pipeline_count = 2;
          problem->rc5_unit_func = rc5_unit_func_000_030;
          coresel = 0;
        }
      }
      #elif defined(__GCC__) || defined(__GNUC__) /* hpux, next, linux, sun3 */
      {
        // rc5/68k/rc5_68k_gcc_crunch.c around rc5/68k/crunch.68k.gcc.s
        //xtern "C" u32 rc5_68k_crunch_unit_func( RC5UnitWork *, u32 );
        problem->rc5_unit_func = rc5_68k_crunch_unit_func;
        problem->pipeline_count = 1; //the default is 2
        coresel = 0;
      }
      #else 
      {
        // rc5/ansi/rc5ansi1-b2.cpp
        //xtern "C" u32 rc5_ansi_1_b2_rg_unit_func( RC5UnitWork *, u32 );
        problem->rc5_unit_func = rc5_ansi_1_b2_rg_unit_func;
        problem->pipeline_count = 1; //the default is 2
        coresel = 0;
      }
      #endif
    }
    #elif (CLIENT_CPU == CPU_VAX)
    {
      // rc5/ansi/rc5ansi1-b2.cpp
      //xtern "C" u32 rc5_ansi_1_b2_rg_unit_func( RC5UnitWork *, u32 );
      problem->rc5_unit_func = rc5_ansi_1_b2_rg_unit_func;
      problem->pipeline_count = 1; //the default is 2
      coresel = 0;
    }
    #elif (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_POWER)
    {
      #if (CLIENT_CPU == CPU_POWER) && !defined(_AIXALL) //not hybrid
      {
        // rc5/ansi/2-rg.c
        //xtern "C" u32 rc5_ansi_2_rg_unit_func( RC5UnitWork *, u32 );
        problem->rc5_unit_func = rc5_ansi_2_rg_unit_func ; //POWER cpu
        problem->pipeline_count = 2;
      }
      #else //((CLIENT_CPU == CPU_POWERPC) || defined(_AIXALL))
      { 
        //#if (CLIENT_OS == OS_WIN32) //NT has poor PPC assembly
        //  //rc5/ansi/2-rg.c
        //  xtern "C" u32 rc5_ansi_2_rg_unit_func( RC5UnitWork *, u32 );
        //  #define rc5_unit_func_lintilla_compat rc5_ansi_2_rg_unit_func
        //  #define rc5_unit_func_allitnil_compat rc5_ansi_2_rg_unit_func
        //  #define rc5_unit_func_vec_compat      rc5_ansi_2_rg_unit_func
        //#else
        //  // rc5/ppc/rc5_*.cpp
        //  // although Be OS isn't supported on 601 machines and there is
        //  // is no 601 PPC board for the Amiga, lintilla depends on allitnil,
        //  // so we have both anyway, we may as well support both.
        //  xtern "C" u32 rc5_unit_func_allitnil_compat( RC5UnitWork *, u32 );
        //  xtern "C" u32 rc5_unit_func_lintilla_compat( RC5UnitWork *, u32 );
        //  #if (CLIENT_OS == OS_MACOS)
        //    extern "C" u32 rc5_unit_func_vec_compat( RC5UnitWork *, u32 );
        //  #else /* MacOS currently is the only one to support altivec cores */
        //    #define rc5_unit_func_vec_compat  rc5_unit_func_lintilla_compat
        //  #endif
        //#endif
        int gotcore = 0;

        problem->client_cpu = CPU_POWERPC;
        #if defined(_AIXALL) //ie POWER/POWERPC hybrid client
        if ((GetProcessorType(1) & (1L<<24)) != 0) //ARCH_IS_POWER
        {
          problem->client_cpu = CPU_POWER;
          problem->rc5_unit_func = rc5_ansi_2_rg_unit_func ; // rc5/ansi/2-rg.c
          problem->pipeline_count = 2;
          coresel = 0; //core #0 is "RG AIXALL" on POWER, and allitnil on PPC
          gotcore = 1;
        }
        #endif
        if (!gotcore && coresel == 0)     // G1 (PPC 601)
        {  
          problem->rc5_unit_func = rc5_unit_func_allitnil_compat;
          problem->pipeline_count = 1;
          gotcore = 1;
        }  
        else if (!gotcore && coresel == 2) // G4 (PPC 7500)
        {
          problem->rc5_unit_func = rc5_unit_func_vec_compat;
          problem->pipeline_count = 1;
          gotcore = 1;
        }
        if (!gotcore)                     // the rest (G2/G3)
        {
          problem->rc5_unit_func = rc5_unit_func_lintilla_compat;
          problem->pipeline_count = 1;
          coresel = 1;
        }
      }
      #endif
    }
    #elif (CLIENT_CPU == CPU_X86)
    {
      static int ismmx = -1;
      if (coresel < 0 || coresel > 5)
        coresel = 0;
      if (ismmx == -1)
      {
        long det = GetProcessorType(1 /* quietly */);
        ismmx = (det >= 0) ? (det & 0x100) : 0;
      }
      problem->pipeline_count = 2; /* most cases */
      if (coresel == 1)   // Intel 386/486
      {
        problem->rc5_unit_func = rc5_unit_func_486;
        #if defined(SMC) 
        if (problem->threadindex == 0)
          problem->rc5_unit_func =  rc5_unit_func_486_smc;
        #endif
      }
      else if (coresel == 2) // Ppro/PII
        problem->rc5_unit_func = rc5_unit_func_p6;
      else if (coresel == 3) // 6x86(mx)
        problem->rc5_unit_func = rc5_unit_func_6x86;
      else if (coresel == 4) // K5
        problem->rc5_unit_func = rc5_unit_func_k5;
      else if (coresel == 5) // K6/K6-2/K7
      {
        problem->rc5_unit_func = rc5_unit_func_k6;
        #if defined(MMX_RC5_AMD)
        if (ismmx)
        { 
          problem->rc5_unit_func = rc5_unit_func_k6_mmx;
          problem->pipeline_count = 4;
        }
        #endif
      }
      else // Pentium (0/6) + others
      {
        problem->rc5_unit_func = rc5_unit_func_p5;
        #if defined(MMX_RC5)
        if (ismmx)
        { 
          problem->rc5_unit_func = rc5_unit_func_p5_mmx;
          problem->pipeline_count = 4; // RC5 MMX core is 4 pipelines
        }
        #endif
        coresel = 0;
      }
    }
    #elif (CLIENT_CPU == CPU_ALPHA)
    {
      #if (CLIENT_OS == OS_DEC_UNIX)
      {
        //rc5/alpha/rc5-digital-unix-alpha-ev[4|5].cpp
        //xtern "C" u32 rc5_alpha_osf_ev4( RC5UnitWork *, u32 );
        //xtern "C" u32 rc5_alpha_osf_ev5( RC5UnitWork *, u32 );
        if (coresel == 1) /* EV5, EV56, PCA56, EV6 */
        {
          problem->pipeline_count = 2;
          problem->rc5_unit_func = rc5_alpha_osf_ev5;
        }
        else // EV3_CPU, EV4_CPU, LCA4_CPU, EV45_CPU and default
        {
          problem->pipeline_count = 2;
          #if defined(DEC_UNIX_CPU_SELECT)
          problem->rc5_unit_func = rc5_alpha_osf_ev4; 
          #else
          problem->rc5_unit_func = rc5_alpha_osf_ev5; 
          #endif
          coresel = 0;
        }
      }
      #elif (CLIENT_OS == OS_WIN32) /* little-endian asm */
      {
        //rc5/alpha/rc5-alpha-nt.s
        //xtern "C" u32 rc5_unit_func_ntalpha_michmarc( RC5UnitWork *, u32 );
        problem->rc5_unit_func = rc5_unit_func_ntalpha_michmarc;
        problem->pipeline_count = 2;
        coresel = 0;
      }
      #else
      {
        //axp-bmeyer.cpp around axp-bmeyer.s
        //xtern "C" u32 rc5_unit_func_axp_bmeyer( RC5UnitWork *, u32 );
        problem->rc5_unit_func = rc5_unit_func_axp_bmeyer;
        problem->pipeline_count = 2;
        coresel = 0;
      }
      #endif
    }
    #else
    {
      #error "How did you get here?"  
    }
    #endif
    return coresel;
  }
  
  /* ================================================================== */
  
  #ifdef HAVE_DES_CORES
  if (contestid == DES)
  {
    #if (CLIENT_CPU == CPU_ARM)
    {
      //des/arm/des-arm-wrappers.cpp
      //xtern u32 des_unit_func_slice_arm( RC5UnitWork * , u32 *, char * );
      //xtern u32 des_unit_func_slice_strongarm( RC5UnitWork * , u32 *, char * );
      if (coresel == 0)
        problem->des_unit_func = des_unit_func_slice_arm;
      else /* (coresel == 1, default) */
      {
        problem->des_unit_func = des_unit_func_slice_strongarm;
        coresel = 1;
      }
    }
    #elif (CLIENT_CPU == CPU_ALPHA) 
    {
      #if (CLIENT_OS == OS_DEC_UNIX) && defined(DEC_UNIX_CPU_SELECT)
      {
        //xtern u32 des_alpha_osf_ev4( RC5UnitWork * , u32 *, char * );
        //xtern u32 des_alpha_osf_ev5( RC5UnitWork * , u32 *, char * );
        if (coresel == 1) /* EV5, EV56, PCA56, EV6 */
          problem->des_unit_func = des_alpha_osf_ev5;
        else // EV3_CPU, EV4_CPU, LCA4_CPU, EV45_CPU and default
          problem->des_unit_func = des_alpha_osf_ev4;
      }
      #else
      {
        //des/alpha/des-slice-dworz.cpp
        //xtern u32 des_unit_func_slice_dworz( RC5UnitWork * , u32 *, char * );
        problem->des_unit_func = des_unit_func_slice_dworz;
      }
      #endif
    }
    #elif (CLIENT_CPU == CPU_X86)
    {
      //xtern u32 p1des_unit_func_p5( RC5UnitWork * , u32 *, char * );
      //xtern u32 p1des_unit_func_pro( RC5UnitWork * , u32 *, char * );
      //xtern u32 p2des_unit_func_p5( RC5UnitWork * , u32 *, char * );
      //xtern u32 p2des_unit_func_pro( RC5UnitWork * , u32 *, char * );
      //xtern u32 des_unit_func_mmx( RC5UnitWork * , u32 *, char * );
      //xtern u32 des_unit_func_slice( RC5UnitWork * , u32 *, char * );
      u32 (*slicit)(RC5UnitWork *,u32 *,char *) = 
                   ((u32 (*)(RC5UnitWork *,u32 *,char *))0);
      #if defined(CLIENT_SUPPORTS_SMP)
      slicit = des_unit_func_slice; //kwan
      #endif
      #if defined(MMX_BITSLICER) 
      {
        static int ismmx = -1;
        if (ismmx == -1)
        {
          long det = GetProcessorType(1 /* quietly */);
          ismmx = (det >= 0) ? (det & 0x100) : 0;
        }
        if (ismmx) 
          slicit = des_unit_func_mmx;
      }
      #endif  
      if (slicit && coresel > 1) /* not standard bryd and not ppro bryd */
      {                /* coresel=2 is valid only if we have a slice core */
        coresel = 2;
        problem->des_unit_func = slicit;
      }
      else 
      {
        #if defined(CLIENT_SUPPORTS_SMP) 
        // bryd is not thread safe, so make sure that when 
        // running benchmark/test asychronously (ie from a gui), 
        // we pick a core that isn't in use.
        unsigned int thrindex = problem->threadindex;
        if (thrindex == 0 && !problem->threadindex_is_valid)
        { /* !threadindex_is_valid==not probman controlled==benchmark/test*/
          while (GetProblemPointerFromIndex(thrindex))
            thrindex++;
        }
        #endif
        if (coresel == 1) /* movzx bryd */
        {
          problem->des_unit_func = p1des_unit_func_pro;
          #if defined(CLIENT_SUPPORTS_SMP) 
          if (thrindex > 0)  /* not first thread */
          {
            if (thrindex == 1)  /* second thread */
              problem->des_unit_func = p2des_unit_func_pro;
            else if (thrindex == 2) /* third thread */
              problem->des_unit_func = p1des_unit_func_p5;
            else if (thrindex == 3) /* fourth thread */
              problem->des_unit_func = p2des_unit_func_p5;
            else                    /* fifth...nth thread */
              problem->des_unit_func = slicit;
          }
          #endif /* if defined(CLIENT_SUPPORTS_SMP)  */
        }
        else             /* normal bryd */
        {
          coresel = 0;
          problem->des_unit_func = p1des_unit_func_p5;
          #if defined(CLIENT_SUPPORTS_SMP) 
          if (thrindex > 0)  /* not first thread */
          {
            if (thrindex == 1)  /* second thread */
              problem->des_unit_func = p2des_unit_func_p5;
            else if (thrindex == 2) /* third thread */
              problem->des_unit_func = p1des_unit_func_pro;
            else if (thrindex == 3) /* fourth thread */
              problem->des_unit_func = p2des_unit_func_pro;
            else                    /* fifth...nth thread */
              problem->des_unit_func = slicit;
          }
          #endif /* if defined(CLIENT_SUPPORTS_SMP)  */
        }
      }
    }
    #elif defined(MEGGS)
      //des/des-slice-meggs.cpp
      //xtern u32 des_unit_func_meggs( RC5UnitWork *, u32 *iter, char *coremem);
      problem->des_unit_func = des_unit_func_meggs;
    #else
      //all rvc based drivers (eg des/ultrasparc/des-slice-ultrasparc.cpp)
      //xtern u32 des_unit_func_slice( RC5UnitWork *, u32 *iter, char *coremem);
      problem->des_unit_func = des_unit_func_slice;
    #endif
    return coresel;
  }
  #endif /* #ifdef HAVE_DES_CORES */

  /* ================================================================== */

  #if defined(HAVE_OGR_CORES)
  if (contestid == OGR)
  {
    return 0;
  }
  #endif

  /* ================================================================== */

  #ifdef HAVE_CSC_CORES
  if( contestid == CSC ) // CSC
  {
    //xtern "C" s32 csc_unit_func_1k  ( RC5UnitWork *, u32 *iterations, void *membuff );
    //xtern "C" s32 csc_unit_func_1k_i( RC5UnitWork *, u32 *iterations, void *membuff );
    //xtern "C" s32 csc_unit_func_6b  ( RC5UnitWork *, u32 *iterations, void *membuff );
    //xtern "C" s32 csc_unit_func_6b_i( RC5UnitWork *, u32 *iterations, void *membuff );

    problem->unit_func = csc_unit_func_1k_i; /* default */
    switch( coresel ) 
    {
      case 0 : problem->unit_func = csc_unit_func_6b_i;
               break;
      case 1 : problem->unit_func = csc_unit_func_6b;
               break;
      default: coresel = 2;
      case 2 : problem->unit_func = csc_unit_func_1k_i;
               break;
      case 3 : problem->unit_func = csc_unit_func_1k;
               break;
#if defined(MMX_CSC)
      case 4 : problem->unit_func = csc_unit_func_6b_mmx;
               break;
#endif
    }
    return coresel;
  }
  #endif /* #ifdef HAVE_CSC_CORES */

  /* ================================================================== */

  return -1; /* core selection failed */
}

/* ------------------------------------------------------------- */

Problem::Problem(long _threadindex /* defaults to -1L */)
{
  threadindex_is_valid = (_threadindex!=-1L);
  threadindex = ((threadindex_is_valid)?((unsigned int)_threadindex):(0));

  /* this next part is essential for alpha, but is probably beneficial to
     all platforms. If it fails for your os/cpu, we may need to redesign 
     how objects are allocated/how rc5unitwork is addressed, so let me know.
                                                       -cyp Jun 14 1999
  */
  
  {
    RC5UnitWork *w = &rc5unitwork;
    unsigned long ww = ((unsigned long)w);
  
    #if (CLIENT_CPU == CPU_ALPHA) /* sizeof(long) can be either 4 or 8 */
    ww &= 0x7; /* (sizeof(longword)-1); */
    #else
    ww &= (sizeof(int)-1); /* int alignment */
    #endif        
    if (ww) 
    {
      Log("rc5unitwork for problem %d is misaligned!\n", threadindex);
      RaiseExitRequestTrigger();
      return;
    }  
  }
//LogScreen("Problem created. threadindex=%u\n",threadindex);

  initialized = 0;
  started = 0;
  
  #ifdef STRESS_THREADS_AND_BUFFERS 
  {
    static int runlevel = 0;
    if (runlevel != -12345)
    {
      if ((++runlevel) != 1)
      {
        --runlevel;
        return;
      }
      RaisePauseRequestTrigger();
      LogScreen("Warning! STRESS_THREADS_AND_BUFFERS is defined.\n"
                "Are you sure that the client is pointing at\n"
                "a test proxy? If so, type 'yes': ");
      char getyes[10];
      ConInStr(getyes,4,0);
      ClearPauseRequestTrigger();
      if (strcmpi(getyes,"yes") != 0)
      {
        runlevel = +12345;
        RaiseExitRequestTrigger();
        return;
      }
      runlevel = -12345;
    }
  }
  #endif    
}

/* ------------------------------------------------------------------- */

Problem::~Problem()
{
  started = 0; // nothing to do. - suppress compiler warning

#if (CLIENT_OS == OS_RISCOS) && defined(HAVE_X86_CARD_SUPPORT)
  if (GetNumberOfDetectedProcessors() > 1 && /* have x86 card */
      GetProblemIndexFromPointer(this) == 1)
  {
    _kernel_swi_regs r;
    r.r[0] = 0;
    _kernel_swi(RC5PC_RetriveBlock,&r,&r);
    _kernel_swi(RC5PC_Off,&r,&r);
  }
#endif
}

/* ------------------------------------------------------------------- */

// for some odd reasons, the RC5 algorithm requires keys in reversed order
//         key.hi   key.lo
// ie key 01234567:89ABCDEF is sent to rc5_unit_func like that :
//        EFCDAB89:67452301
// This function switches from one format to the other.
//
// [Even if it looks like a little/big endian problem, it isn't. Whatever
//  endianess the underlying system has, we must swap every byte in the key
//  before sending it to rc5_unit_func()]
//
// Note that DES has a similiar but far more complex system, but everything
// is handled by des_unit_func().

static void  __SwitchRC5Format(u64 *_key)                               
{                                                                       
    register u32 tempkeylo = _key->hi; /* note: we switch the order */  
    register u32 tempkeyhi = _key->lo;                                  
                                                                        
    _key->lo =                                                          
      ((tempkeylo >> 24) & 0x000000FFL) |                               
      ((tempkeylo >>  8) & 0x0000FF00L) |                               
      ((tempkeylo <<  8) & 0x00FF0000L) |                               
      ((tempkeylo << 24) & 0xFF000000L);                                
    _key->hi =                                                          
      ((tempkeyhi >> 24) & 0x000000FFL) |                               
      ((tempkeyhi >>  8) & 0x0000FF00L) |                               
      ((tempkeyhi <<  8) & 0x00FF0000L) |                               
      ((tempkeyhi << 24) & 0xFF000000L);                                
}                                                                       

/* ------------------------------------------------------------------- */

// Input:  - an RC5 key in 'mangled' (reversed) format or a DES key
//         - an incrementation count
//         - a contest identifier (0==RC5 1==DES 2==OGR 3==CSC)
//
// Output: the key incremented

static void __IncrementKey(u64 *key, u32 iters, int contest)        
{                                                                   
  switch (contest)                                                  
  {                                                                 
    case RC5:
      __SwitchRC5Format (key);                                      
      key->lo += iters;                                             
      if (key->lo < iters) key->hi++;                               
      __SwitchRC5Format (key);                                      
      break;                                                        
    case DES:
    case CSC:
      key->lo += iters;                                             
      if (key->lo < iters) key->hi++; /* Account for carry */       
      break;                                                        
    case OGR:
      /* This should never be called for OGR */                     
      break;                                                        
  }                                                                 
}

/* ------------------------------------------------------------- */

u32 Problem::CalcPermille() /* % completed in the current block, to nearest 0.1%. */
{ 
  u32 retpermille = 0;
  if (initialized && last_resultcode >= 0)
  {
    if (!started)
      retpermille = startpermille;
    else if (last_resultcode != RESULT_WORKING)
      retpermille = 1000;
    else
    {
      switch (contest)
      {
        case RC5:
        case DES:
        case CSC:
                {
                retpermille = (u32)( ((double)(1000.0)) *
                (((((double)(contestwork.crypto.keysdone.hi))*((double)(4294967296.0)))+
                             ((double)(contestwork.crypto.keysdone.lo))) /
                ((((double)(contestwork.crypto.iterations.hi))*((double)(4294967296.0)))+
                             ((double)(contestwork.crypto.iterations.lo)))) ); 
                break;
                }
        case OGR:
                WorkStub curstub;
                ogr->getresult(core_membuffer, &curstub, sizeof(curstub));
                // This is just a quick&dirty calculation that resembles progress.
                retpermille = curstub.stub.diffs[contestwork.ogr.workstub.stub.length]*10
                            + curstub.stub.diffs[contestwork.ogr.workstub.stub.length+1]/10;
                break;
      }
    }
    if (retpermille > 1000)
      retpermille = 1000;
  }
  return retpermille;
}

/* ------------------------------------------------------------------- */

int Problem::LoadState( ContestWork * work, unsigned int contestid, 
                              u32 _iterations, int /* was _cputype */ )
{
  unsigned int sz = sizeof(int);

  if (sz < sizeof(u32)) /* need to do it this way to suppress compiler warnings. */
  {
    LogScreen("FATAL: sizeof(int) < sizeof(u32)\n");
    //#error "everything assumes a 32bit CPU..."
    RaiseExitRequestTrigger();
    return -1;
  }
  if (!IsProblemLoadPermitted(threadindex, contestid))
    return -1;

  last_resultcode = -1;
  started = initialized = 0;
  timehi = timelo = 0;
  runtime_sec = runtime_usec = 0;
  last_runtime_sec = last_runtime_usec = 0;
  memset((void *)&profiling, 0, sizeof(profiling));
  startpermille = permille = 0;
  loaderflags = 0;
  contest = contestid;
  client_cpu = CLIENT_CPU; /* usual case */
  tslice = _iterations;
  coresel = __core_picker(this, contestid );
  if (coresel < 0 || 
     (coresel > 0 && coresel != selcoreValidateCoreIndex(contestid, coresel)))
    return -1;

  //----------------------------------------------------------------

  switch (contest) 
  {
    case RC5:
    #if defined(HAVE_DES_CORES)
    case DES:
    #endif
    #if defined(HAVE_CSC_CORES)
    case CSC: // HAVE_CSC_CORES
    #endif
    {
      // copy over the state information
      contestwork.crypto.key.hi = ( work->crypto.key.hi );
      contestwork.crypto.key.lo = ( work->crypto.key.lo );
      contestwork.crypto.iv.hi = ( work->crypto.iv.hi );
      contestwork.crypto.iv.lo = ( work->crypto.iv.lo );
      contestwork.crypto.plain.hi = ( work->crypto.plain.hi );
      contestwork.crypto.plain.lo = ( work->crypto.plain.lo );
      contestwork.crypto.cypher.hi = ( work->crypto.cypher.hi );
      contestwork.crypto.cypher.lo = ( work->crypto.cypher.lo );
      contestwork.crypto.keysdone.hi = ( work->crypto.keysdone.hi );
      contestwork.crypto.keysdone.lo = ( work->crypto.keysdone.lo );
      contestwork.crypto.iterations.hi = ( work->crypto.iterations.hi );
      contestwork.crypto.iterations.lo = ( work->crypto.iterations.lo );

      //determine starting key number. accounts for carryover & highend of keysdone
      u64 key;
      key.hi = contestwork.crypto.key.hi + contestwork.crypto.keysdone.hi + 
         ((((contestwork.crypto.key.lo & 0xffff) + (contestwork.crypto.keysdone.lo & 0xffff)) + 
           ((contestwork.crypto.key.lo >> 16) + (contestwork.crypto.keysdone.lo >> 16))) >> 16);
      key.lo = contestwork.crypto.key.lo + contestwork.crypto.keysdone.lo;

      // set up the unitwork structure
      rc5unitwork.plain.hi = contestwork.crypto.plain.hi ^ contestwork.crypto.iv.hi;
      rc5unitwork.plain.lo = contestwork.crypto.plain.lo ^ contestwork.crypto.iv.lo;
      rc5unitwork.cypher.hi = contestwork.crypto.cypher.hi;
      rc5unitwork.cypher.lo = contestwork.crypto.cypher.lo;

      rc5unitwork.L0.lo = key.lo;
      rc5unitwork.L0.hi = key.hi;
      if (contest == RC5)
        __SwitchRC5Format (&(rc5unitwork.L0));

      refL0 = rc5unitwork.L0;

      if (contestwork.crypto.keysdone.lo!=0 || contestwork.crypto.keysdone.hi!=0 )
      {
        startpermille = (u32)( ((double)(1000.0)) *
        (((((double)(contestwork.crypto.keysdone.hi))*((double)(4294967296.0)))+
                           ((double)(contestwork.crypto.keysdone.lo))) /
        ((((double)(contestwork.crypto.iterations.hi))*((double)(4294967296.0)))+
                        ((double)(contestwork.crypto.iterations.lo)))) );
      }     
      break;
    }
    #if defined(HAVE_OGR_CORES)
    case OGR:
    {
      contestwork.ogr = work->ogr;
      contestwork.ogr.nodes.lo = 0;
      contestwork.ogr.nodes.hi = 0;
      ogr = ogr_get_dispatch_table();
      int r = ogr->init();
      if (r != CORE_S_OK)
        return -1;
      r = ogr->create(&contestwork.ogr.workstub, 
                      sizeof(WorkStub), core_membuffer, sizeof(core_membuffer));
      if (r != CORE_S_OK)
        return -1;
      if (contestwork.ogr.workstub.worklength > contestwork.ogr.workstub.stub.length)
      {
        // This is just a quick&dirty calculation that resembles progress.
        startpermille = contestwork.ogr.workstub.stub.diffs[contestwork.ogr.workstub.stub.length]*10
                      + contestwork.ogr.workstub.stub.diffs[contestwork.ogr.workstub.stub.length+1]/10;
      }
      break;
    }  
    #endif
    default:  
      return -1;
  }

  //---------------------------------------------------------------
#if (CLIENT_OS == OS_RISCOS) && defined(HAVE_X86_CARD_SUPPORT)
  if (threadindex == 1 &&                  /* reserved for x86 thread*/
      GetNumberOfDetectedProcessors() > 1) /* have x86 card */
  {
    RC5PCstruct rc5pc;
    _kernel_oserror *err;
    _kernel_swi_regs r;
  
    rc5pc.key.hi = contestwork.key.hi;
    rc5pc.key.lo = contestwork.key.lo;
    rc5pc.iv.hi = contestwork.iv.hi;
    rc5pc.iv.lo = contestwork.iv.lo;
    rc5pc.plain.hi = contestwork.plain.hi;
    rc5pc.plain.lo = contestwork.plain.lo;
    rc5pc.cypher.hi = contestwork.cypher.hi;
    rc5pc.cypher.lo = contestwork.cypher.lo;
    rc5pc.keysdone.hi = contestwork.keysdone.hi;
    rc5pc.keysdone.lo = contestwork.keysdone.lo;
    rc5pc.iterations.hi = contestwork.iterations.hi;
    rc5pc.iterations.lo = contestwork.iterations.lo;
    rc5pc.timeslice = tslice;

    client_cpu = CPU_X86;
 
    err = _kernel_swi(RC5PC_On,&r,&r);
    if (err)
      LogScreen("Failed to start x86 card");
    else
    {
      r.r[1] = (int)&rc5pc;
      err = _kernel_swi(RC5PC_AddBlock,&r,&r);
      if ((err) || (r.r[0] == -1))
      {
        LogScreen("Failed to add block to x86 cruncher\n");
      }
    }
  }
#endif

  last_resultcode = RESULT_WORKING;
  initialized = 1;

  return( 0 );
}

/* ------------------------------------------------------------------- */

int Problem::RetrieveState( ContestWork * work, unsigned int *contestid, int dopurge )
{
  if (!initialized)
    return -1;
  if (work) // store back the state information
  {
    switch (contest) {
      case RC5:
      case DES:
      case CSC:
        // nothing special needs to be done here
        break;
      case OGR:
        ogr->getresult(core_membuffer, &contestwork.ogr.workstub, sizeof(WorkStub));
        break;
    }
    memcpy( (void *)work, (void *)&contestwork, sizeof(ContestWork));
  }
  if (contestid)
    *contestid = contest;
  if (dopurge)
    initialized = 0;
  if (last_resultcode < 0)
    return -1;
  return ( last_resultcode );
}

/* ------------------------------------------------------------- */

int Problem::Run_RC5(u32 *iterationsP, int *resultcode)
{
  u32 kiter = 0;
  u32 iterations = *iterationsP;

  // align the iterations to an even-multiple of pipeline_count and 2 
  u32 alignfact = pipeline_count + (pipeline_count & 1);
  iterations = ((iterations + (alignfact - 1)) & ~(alignfact - 1));

  // don't allow a too large of a iterations be used ie (>(iter-keysdone)) 
  // (technically not necessary, but may save some wasted time)
  if (contestwork.crypto.keysdone.hi == contestwork.crypto.iterations.hi)
  {
    u32 todo = contestwork.crypto.iterations.lo-contestwork.crypto.keysdone.lo;
    if (todo < iterations)
    {
      iterations = todo;
      iterations = ((iterations + (alignfact - 1)) & ~(alignfact - 1));
    }
  }

#if 0
LogScreen("align iterations: effective iterations: %lu (0x%lx),\n"
          "suggested iterations: %lu (0x%lx)\n"
          "pipeline_count = %lu, iterations%%pipeline_count = %lu\n", 
          (unsigned long)iterations, (unsigned long)(*iterationsP),
          (unsigned long)tslice, (unsigned long)tslice,
          pipeline_count, iterations%pipeline_count );
#endif

  kiter = (*rc5_unit_func)(&rc5unitwork, iterations/pipeline_count );
  *iterationsP = iterations;

  __IncrementKey (&refL0, iterations, contest);
    // Increment reference key count

  if (((refL0.hi != rc5unitwork.L0.hi) ||  // Compare ref to core
      (refL0.lo != rc5unitwork.L0.lo)) &&  // key incrementation
      (kiter == iterations))
  {
    #if 0 /* can you spell "thread safe"? */
    Log("Internal Client Error #23: Please contact help@distributed.net\n"
        "Debug Information: %08x:%08x - %08x:%08x\n",
        rc5unitwork.L0.hi, rc5unitwork.L0.lo, refL0.hi, refL0.lo);
    #endif
    *resultcode = -1;
    return -1;
  };

  contestwork.crypto.keysdone.lo += kiter;
  if (contestwork.crypto.keysdone.lo < kiter)
    contestwork.crypto.keysdone.hi++;
    // Checks passed, increment keys done count.

  if (kiter < iterations)
  {
    // found it!
    u32 keylo = contestwork.crypto.key.lo;
    contestwork.crypto.key.lo += contestwork.crypto.keysdone.lo;
    contestwork.crypto.key.hi += contestwork.crypto.keysdone.hi;
    if (contestwork.crypto.key.lo < keylo) 
      contestwork.crypto.key.hi++; // wrap occured ?
    *resultcode = RESULT_FOUND;
    return RESULT_FOUND;
  }
  else if (kiter != iterations)
  {
    #if 0 /* can you spell "thread safe"? */
    Log("Internal Client Error #24: Please contact help@distributed.net\n"
        "Debug Information: k: %x t: %x\n"
        "Debug Information: %08x:%08x - %08x:%08x\n", kiter, iterations,
        rc5unitwork.L0.lo, rc5unitwork.L0.hi, refL0.lo, refL0.hi);
    #endif
    *resultcode = -1;
    return -1;
  };

  if ( ( contestwork.crypto.keysdone.hi > contestwork.crypto.iterations.hi ) ||
       ( ( contestwork.crypto.keysdone.hi == contestwork.crypto.iterations.hi ) &&
       ( contestwork.crypto.keysdone.lo >= contestwork.crypto.iterations.lo ) ) )
  {
    // done with this block and nothing found
    *resultcode = RESULT_NOTHING;
    return RESULT_NOTHING;
  }

  // more to do, come back later.
  *resultcode = RESULT_WORKING;
  return RESULT_WORKING;    // Done with this round
}  

/* ------------------------------------------------------------- */

int Problem::Run_CSC(u32 *iterationsP, int *resultcode)
{
#ifndef HAVE_CSC_CORES
  *iterationsP = 0;
  *resultcode = -1;
  return -1;
#else  
  s32 rescode = (*unit_func)( &rc5unitwork, iterationsP, core_membuffer );

  if (rescode < 0) /* "kiter" error */
  {
    *resultcode = -1;
    return -1;
  }
  *resultcode = (int)rescode;

  // Increment reference key count
  __IncrementKey (&refL0, *iterationsP, contest);

  // Compare ref to core key incrementation
  if ((refL0.hi != rc5unitwork.L0.hi) || (refL0.lo != rc5unitwork.L0.lo))
  { 
    #ifdef DEBUG_CSC_CORE /* can you spell "thread safe"? */
    Log("CSC incrementation mismatch:\n"
        "expected %08x:%08x, got %08x:%08x\n",
        refL0.lo, refL0.hi, rc5unitwork.L0.lo, rc5unitwork.L0.hi );
    #endif
    *resultcode = -1;
    return -1;
  }

  // Checks passed, increment keys done count.
  contestwork.crypto.keysdone.lo += *iterationsP;
  if (contestwork.crypto.keysdone.lo < *iterationsP)
    contestwork.crypto.keysdone.hi++;

  // Update data returned to caller
  if (*resultcode == RESULT_FOUND)
  {
    u32 keylo = contestwork.crypto.key.lo;
    contestwork.crypto.key.lo += contestwork.crypto.keysdone.lo;
    contestwork.crypto.key.hi += contestwork.crypto.keysdone.hi;
    if (contestwork.crypto.key.lo < keylo) 
      contestwork.crypto.key.hi++; // wrap occured ?
    return RESULT_FOUND;
  }

  if ( ( contestwork.crypto.keysdone.hi > contestwork.crypto.iterations.hi ) ||
       ( ( contestwork.crypto.keysdone.hi == contestwork.crypto.iterations.hi ) &&
       ( contestwork.crypto.keysdone.lo >= contestwork.crypto.iterations.lo ) ) )
  {
    *resultcode = RESULT_NOTHING;
    return RESULT_NOTHING;
  }
  // more to do, come back later.
  *resultcode = RESULT_WORKING;
  return RESULT_WORKING; // Done with this round
#endif  
}

/* ------------------------------------------------------------- */

int Problem::Run_DES(u32 *iterationsP, int *resultcode)
{
#ifndef HAVE_DES_CORES
  *iterationsP = 0;  /* no keys done */
  *resultcode = -1; /* core error */
  return -1;
#else

  //iterationsP == in: suggested iterations, out: effective iterations
  u32 kiter = (*des_unit_func)( &rc5unitwork, iterationsP, core_membuffer );

  __IncrementKey (&refL0, *iterationsP, contest);
  // Increment reference key count

  if (((refL0.hi != rc5unitwork.L0.hi) ||  // Compare ref to core
      (refL0.lo != rc5unitwork.L0.lo)) &&  // key incrementation
      (kiter == *iterationsP))
  {
    #if 0 /* can you spell "thread safe"? */
    Log("Internal Client Error #23: Please contact help@distributed.net\n"
        "Debug Information: %08x:%08x - %08x:%08x\n",
        rc5unitwork.L0.lo, rc5unitwork.L0.hi, refL0.lo, refL0.hi);
    #endif
    *resultcode = -1;
    return -1;
  };

  contestwork.crypto.keysdone.lo += kiter;
  if (contestwork.crypto.keysdone.lo < kiter)
    contestwork.crypto.keysdone.hi++;
    // Checks passed, increment keys done count.

  // Update data returned to caller
  if (kiter < *iterationsP)
  {
    // found it!
    u32 keylo = contestwork.crypto.key.lo;
    contestwork.crypto.key.lo += contestwork.crypto.keysdone.lo;
    contestwork.crypto.key.hi += contestwork.crypto.keysdone.hi;
    if (contestwork.crypto.key.lo < keylo) 
      contestwork.crypto.key.hi++; // wrap occured ?
    *resultcode = RESULT_FOUND;
    return RESULT_FOUND;
  }
  else if (kiter != *iterationsP)
  {
    #if 0 /* can you spell "thread safe"? */
    Log("Internal Client Error #24: Please contact help@distributed.net\n"
        "Debug Information: k: %x t: %x\n"
        "Debug Information: %08x:%08x - %08x:%08x\n", kiter, *iterationsP,
        rc5unitwork.L0.lo, rc5unitwork.L0.hi, refL0.lo, refL0.hi);
    #endif
    *resultcode = -1; /* core error */
    return -1;
  };

  if ( ( contestwork.crypto.keysdone.hi > contestwork.crypto.iterations.hi ) ||
     ( ( contestwork.crypto.keysdone.hi == contestwork.crypto.iterations.hi ) &&
     ( contestwork.crypto.keysdone.lo >= contestwork.crypto.iterations.lo ) ) )
  {
    // done with this block and nothing found
    *resultcode = RESULT_NOTHING;
    return RESULT_NOTHING;
  }

  // more to do, come back later.
  *resultcode = RESULT_WORKING;
  return RESULT_WORKING; // Done with this round
#endif /* #ifdef HAVE_DES_CORES */
}

/* ------------------------------------------------------------- */

int Problem::Run_OGR(u32 *iterationsP, int *resultcode)
{
#if !defined(HAVE_OGR_CORES)
  iterationsP = iterationsP;
#else
  int r, nodes;

  if (*iterationsP > 0x100000UL)
    *iterationsP = 0x100000UL;

  nodes = (int)(*iterationsP);
  r = ogr->cycle(core_membuffer, &nodes);
  *iterationsP = (u32)nodes;

  u32 newnodeslo = contestwork.ogr.nodes.lo + nodes;
  if (newnodeslo < contestwork.ogr.nodes.lo) {
    contestwork.ogr.nodes.hi++;
  }
  contestwork.ogr.nodes.lo = newnodeslo;

  switch (r) 
  {
    case CORE_S_OK:
    {
      r = ogr->destroy(core_membuffer);
      if (r == CORE_S_OK) 
      {
        *resultcode = RESULT_NOTHING;
        return RESULT_NOTHING;
      }
      break;
    }
    case CORE_S_CONTINUE:
    {
      *resultcode = RESULT_WORKING;
      return RESULT_WORKING;
    }
    case CORE_S_SUCCESS:
    {
      if (ogr->getresult(core_membuffer, &contestwork.ogr.workstub, sizeof(WorkStub)) == CORE_S_OK)
      {
        //Log("OGR Success!\n");
        contestwork.ogr.workstub.stub.length = 
                  (u16)(contestwork.ogr.workstub.worklength);
        *resultcode = RESULT_FOUND;
        return RESULT_FOUND;
      }
      break;
    }
  }
  /* Something bad happened */
#endif
 *resultcode = -1; /* this will cause the problem to be discarded */
 return -1;
}

/* ------------------------------------------------------------- */

int Problem::Run(void) /* returns RESULT_*  or -1 */
{
  int using_ptime;
  struct timeval stop, start, pstart;
  int retcode, core_resultcode;
  u32 iterations;

  if ( !initialized )
    return ( -1 );

  if ( last_resultcode != RESULT_WORKING ) /* _FOUND, _NOTHING or -1 */
    return ( last_resultcode );

  CliClock(&start);
  using_ptime = 1;
  if (CliGetProcessTime(&pstart) < 0)
    using_ptime = 0;
    
  if (!started)
  {
    timehi = start.tv_sec; timelo = start.tv_usec;
    runtime_sec = runtime_usec = 0;
    memset((void *)&profiling, 0, sizeof(profiling));
    started=1;

#ifdef STRESS_THREADS_AND_BUFFERS 
    contest = RC5;
    contestwork.crypto.key.hi = contestwork.crypto.key.lo = 0;
    contestwork.crypto.keysdone.hi = contestwork.crypto.iterations.hi;
    contestwork.crypto.keysdone.lo = contestwork.crypto.iterations.lo;
    runtime_usec = 1; /* ~1Tkeys for a 2^20 packet */
    last_resultcode = RESULT_NOTHING;
    return RESULT_NOTHING;
#endif    
  }

  /* 
    On return from the Run_XXX contestwork must be in a state that we
    can put away to disk - that is, do not expect the loader (probfill 
    et al) to fiddle with iterations or key or whatever.
    
    The Run_XXX functions do *not* update problem.last_resultcode, they use
    core_resultcode instead. This is so that members of the problem object
    that are updated after the resultcode has been set will not be out of
    sync when the main thread gets it with RetrieveState(). 
    
    note: although the value returned by Run_XXX is usually the same as 
    the core_resultcode it is not always so. For instance, if 
    post-LoadState() initialization  failed, but can be deferred, Run_XXX 
    may choose to return -1, but keep core_resultcode at RESULT_WORKING.
  */

  iterations = tslice;
  last_runtime_usec = last_runtime_sec = 0;
  core_resultcode = last_resultcode;
  retcode = -1;

  switch (contest)
  {
    case RC5: retcode = Run_RC5( &iterations, &core_resultcode );
              break;
    case DES: retcode = Run_DES( &iterations, &core_resultcode );
              break;
    case OGR: retcode = Run_OGR( &iterations, &core_resultcode );
              break;
    case CSC: retcode = Run_CSC( &iterations, &core_resultcode );
              break;
    default: retcode = core_resultcode = last_resultcode = -1;
       break;
  }

  
  if (retcode < 0) /* don't touch tslice or runtime as long as < 0!!! */
  {
    return -1;
  }
  
  core_run_count++;
  if (using_ptime)
  {
    if (CliGetProcessTime(&stop) < 0)
      using_ptime = 0;
    else 
    {
      start.tv_sec = pstart.tv_sec;
      start.tv_usec = pstart.tv_usec;
    }
  }
  if (!using_ptime)
  {
    CliClock(&stop);
    if ( core_resultcode != RESULT_WORKING ) /* _FOUND, _NOTHING */
    {
      if (((u32)(stop.tv_sec)) > ((u32)(timehi)))
      {
        u32 tmpdif = timehi - stop.tv_sec;
        tmpdif = (((tmpdif >= runtime_sec) ?
          (tmpdif - runtime_sec) : (runtime_sec - tmpdif)));
        if ( tmpdif < core_run_count )
        {
          runtime_sec = runtime_usec = 0;
          start.tv_sec = timehi;
          start.tv_usec = timelo;
        }
      }
    }
  }
  if (stop.tv_sec < start.tv_sec || 
     (stop.tv_sec == start.tv_sec && stop.tv_usec <= start.tv_usec))
  {
    //AIEEE! clock is whacky (or unusably inaccurate if ==)
  }
  else
  {
    if (stop.tv_usec < start.tv_usec)
    {
      stop.tv_sec--;
      stop.tv_usec+=1000000L;
    }
    runtime_usec += (last_runtime_usec = (stop.tv_usec - start.tv_usec));
    runtime_sec  += (last_runtime_sec = (stop.tv_sec - start.tv_sec));
    if (runtime_usec > 1000000L)
    {
      runtime_sec++;
      runtime_usec-=1000000L;
    }
  }

  tslice = iterations;

  last_resultcode = core_resultcode;
  return last_resultcode;
}

/* ----------------------------------------------------------------------- */

int IsProblemLoadPermitted(long prob_index, unsigned int contest_i)
{
  prob_index = prob_index; /* possibly unused */

  #if (CLIENT_OS == OS_RISCOS) && defined(HAVE_X86_CARD_SUPPORT)
  if (prob_index == 1 && /* thread number reserved for x86 card */
     contest_i != RC5 && /* RISC OS x86 thread only supports RC5 */
     GetNumberOfDetectedProcessors() > 1) /* have x86 card */
    return 0;
  #endif
  switch (contest_i)
  {
    case RC5: 
    {
      return 1;
    }
    case DES:
    {
      #ifdef HAVE_DES_CORES
      return 1;
      #else
      return 0;
      #endif
    }
    case OGR:
    {
      #ifdef HAVE_OGR_CORES
      return 1;
      #else
      return 0;
      #endif
    }
    case CSC:
    {
      #ifdef HAVE_CSC_CORES
      return 1;
      #else
      return 0;
      #endif
    }
  }
  return 0;
}
