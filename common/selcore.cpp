/* 
 * Copyright distributed.net 1998-2001 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * Written August 1998 by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * -------------------------------------------------------------------
 * program (pro'-gram) [vi]: To engage in a pastime similar to banging
 * one's head against a wall but with fewer opportunities for reward.
 * -------------------------------------------------------------------
 */
const char *selcore_cpp(void) {
return "@(#)$Id: selcore.cpp,v 1.47.2.131 2002/03/27 22:46:52 andreasb Exp $"; }

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
#if (CLIENT_OS == OS_AIX) // needs signal handler
  #include <sys/signal.h>
  #include <setjmp.h>
#endif
#if (CLIENT_CPU == CPU_X86) && defined(SMC)
  #if defined(__unix__)
    #include <sys/types.h>
    #include <sys/mman.h>
  #elif defined(USE_DPMI)
    extern "C" smc_dpmi_ds_alias_alloc(void);
    extern "C" smc_dpmi_ds_alias_free(void);
  #endif
  extern "C" u32 rc5_unit_func_486_smc( RC5UnitWork * , u32 iterations );
  static int x86_smc_initialized = -1;
#endif

/* ------------------------------------------------------------------------ */

static const char **__corenames_for_contest( unsigned int cont_i )
{
  /* 
   When selecting corenames, use names that describe how (what optimization)
   they are different from their predecessor(s). If only one core,
   use the obvious "MIPS optimized" or similar.
  */
  #define LARGEST_SUBLIST 11 /* including the terminating null */
  static const char *corenames_table[CONTEST_COUNT][LARGEST_SUBLIST]= 
  #undef LARGEST_SUBLIST
  /* ================================================================== */
  {
  #if (CLIENT_CPU == CPU_X86)
    { /* RC5 */
      /* we should be using names that tell us how the cores are different
         (just like "bryd" and "movzx bryd")
      */
      "RG/BRF class 5",        /* 0. P5/Am486 */
      "RG class 3/4",          /* 1. 386/486 */
      "RG class 6",            /* 2. PPro/II/III */
      "RG re-pair I",          /* 3. Cyrix 486/6x86[MX]/MI */
      "RG RISC-rotate I",      /* 4. K5 */
      "RG RISC-rotate II",     /* 5. K6 - may become mmx-k6-2 core at runtime */
      "RG/DG re-pair III",     /* 6. K7 Athlon and Cx-MII, based on Cx re-pair */
      "RG/BRF self-mod",       /* 7. SMC */
      "AK class 7",            /* 8. P4 */
      "jasonp P5/MMX",         /* 9. P5/MMX *only* - slower on PPro+ */
      NULL
    },
    { /* DES */
      "byte Bryd",
      "movzx Bryd",
      "Kwan/Bitslice",
      "MMX/Bitslice",
      NULL
    },
    { /* OGR */
      "GARSP 5.13-A",
      "GARSP 5.13-B",
      NULL
    },
  #elif (CLIENT_CPU == CPU_ARM)
    { /* RC5 */
      "Series A", /* (autofor for ARM 3/6xx/7xxx) "ARM 3, 610, 700, 7500, 7500FE" */
      "Series B", /* (autofor ARM 8xx/StrongARM) "ARM 810, StrongARM 110" */
      "Series C", /* (autofor ARM 2xx) "ARM 2, 250" */
      NULL             /* "ARM 710" */
    },
    { /* DES */
      "Standard ARM core", /* "ARM 3, 610, 700, 7500, 7500FE" or  "ARM 710" */
      "StrongARM core", /* "ARM 810, StrongARM 110" or "ARM 2, 250" */
      NULL
    },
    { /* OGR */
      "GARSP 5.13",
      NULL
    },
  #elif (CLIENT_CPU == CPU_68K)
    { /* RC5 */
      #if defined(__GCC__) || defined(__GNUC__) || \
          (CLIENT_OS == OS_AMIGAOS)// || (CLIENT_OS == OS_MACOS)
      "68000/010", /* 68000/010 */
      "68020/030", /* 68020/030 */
      "68040/060", /* 68040/060 */
      #else
      "Generic",
      #endif
      NULL
    },
    { /* DES */
      "Generic", 
      NULL
    },
    { /* OGR */
      "GARSP 5.13 68000",
      "GARSP 5.13 68020",
      "GARSP 5.13 68030",
      "GARSP 5.13 68040",
      "GARSP 5.13 68060",
      NULL
    },
  #elif (CLIENT_CPU == CPU_ALPHA) 
    { /* RC5 */
      #if (CLIENT_OS == OS_WIN32)
      "Marcelais",
      #else
      "axp bmeyer",
      #endif
      NULL
    },
    { /* DES */
      "dworz/amazing",
      NULL
    },
    { /* OGR */
      "GARSP 5.13",
      NULL
    },
  #elif (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_POWER)
    { /* RC5 */
      /* lintilla depends on allitnil, and since we need both even on OS's 
         that don't support the 601, we may as well "support" them visually.
      */
      "allitnil",
      "lintilla",
      "lintilla-604",    /* Roberto Ragusa's core optimized for PPC 604e */
      "Power RS",        /* _AIXALL only */
      "crunch-vec",      /* altivec only */
      "crunch-vec-7450", /* altivec only */
      NULL
    },
    { /* DES */
      "Generic DES core", 
      NULL
    },
    { /* OGR */
      "GARSP 5.13 PPC-scalar",
      "GARSP 5.13 PowerRS",    /* _AIXALL only */
      "GARSP 5.13 PPC-vector", /* altivec only */
      NULL
    },
  #elif (CLIENT_CPU == CPU_SPARC)
    { /* RC5 */
      #if ((CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS) || (CLIENT_OS == OS_LINUX))
        "Ultrasparc RC5 core",
        "Generic RC5 core",
      #else
        "Generic RC5 core",
      #endif
      NULL
    },
    { /* DES */
      "Generic DES core",
      NULL
    },
    { /* OGR */
      "GARSP 5.13",
      NULL
    },
  #else
    { /* RC5 */
      "Generic RC5 core",
      NULL
    },
    { /* DES */
      "Generic DES core",
      NULL
    },
    { /* OGR */
      "GARSP 5.13",
      NULL
    },
  #endif  
    { /* CSC */
      "6 bit - inline", 
      "6 bit - called",
      "1 key - inline", 
      "1 key - called",
      NULL, /* room */
      NULL
    }
  };
  /* ================================================================== */

  if (cont_i < CONTEST_COUNT)
  {
    return (const char **)(&(corenames_table[cont_i][0]));
  }
  return ((const char **)0);
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
static int __apply_selcore_substitution_rules(unsigned int contestid, 
                                              int cindex)
{
  #if (CLIENT_CPU == CPU_ARM)
  if (contestid == CSC)
  {
    if (cindex != 0) /* "1 key - called" */
      return 0;      /* the only supported core */
  }
  #elif (CLIENT_CPU == CPU_68K)
  if (contestid == OGR)
  {
    long det = GetProcessorType(1);
    if (det == 68000) cindex = 0;
  }
  #elif (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_POWER)
  {
    /* AIX note:
    ** A power-only client running on PPC will never get here. So, at this 
    ** point its either a power-only client running on power, or a ppc-only
    ** client (no power core) running on PPC or power, or _AIXALL client 
    ** running on either power or PPC.
    */
    int have_vec = 0;
    int have_pwr = 0;

    #if defined(_AIXALL) || defined(__VEC__) /* only these two need detection*/
    long det = GetProcessorType(1);
    #endif
    #if defined(_AIXALL)                /* is a power/PPC hybrid client */
    have_pwr = (det >= 0 && (det & 1L<<24)!=0);
    #elif (CLIENT_CPU == CPU_POWER)     /* power only */
    have_pwr = 1;                       /* see note above */  
    #endif
    #if defined(__VEC__)                /* OS+compiler support altivec */ 
    have_vec = (det >= 0 && (det & 1L<<25)!=0); /* have altivec */
    #endif

    if (contestid == RC5)
    {
      if (have_pwr)
        cindex = 3;                     /* "PowerRS" */
      if (!have_pwr && cindex == 3)     /* "PowerRS" */
        cindex = 1;                     /* "lintilla" */
      if (!have_vec && cindex == 4)     /* "crunch-vec" */
        cindex = 1;                     /* "lintilla" */
      if (!have_vec && cindex == 5)     /* "crunch-vec-7450" */
        cindex = 1;                     /* "lintilla" */
    }
    else if (contestid == OGR)
    {
      if (have_pwr)
        cindex = 1;                     /* "PowerRS" */
      if (!have_pwr && cindex == 1)     /* "PowerRS" */
        cindex = 0;                     /* "PPC-scalar" */
      if (!have_vec && cindex == 2)     /* "PPC-vector" */
        cindex = 0;                     /* "PPC-scalar" */
    }
  }  
  #elif (CLIENT_CPU == CPU_X86)
  {
    long det = GetProcessorType(1);
    int have_mmx = (det >= 0 && (det & 0x100)!=0);
    int have_3486 = (det >= 0 && (det & 0xff)==1);
    int have_smc = 0;
    int have_nasm = 0;

    #if !defined(HAVE_NO_NASM)
    have_nasm = 1;
    #endif
    #if defined(SMC)
    have_smc = (x86_smc_initialized > 0);
    #endif

    if (contestid == RC5)
    {
      if (!have_nasm && cindex == 6)    /* "RG/DG re-pair III" */
        cindex = ((have_3486 && have_smc)?(7):(3)); /* "RG self-mod" or
                                                       "RG/HB re-pair I" */
      if (!have_smc && cindex == 7)     /* "RG self-modifying" */
        cindex = 1;                     /* "RG class 3/4" */
      if (!have_nasm && cindex == 8)    /* "AK Class 7" */
        cindex = 2;                     /* "RG Class 6" */
      if (!have_mmx && cindex == 9)     /* "jasonp P5/MMX" */
        cindex = 0;                     /* "RG Class 5" */
      if (!have_nasm && cindex == 9)    /* "jasonp P5/MMX" */
        cindex = 0;                     /* "RG Class 5" */
    }
    else if (contestid == DES)
    {
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
    }
  }
  #endif         
  contestid = contestid; /* possibly unused */
  return cindex;
}

/* -------------------------------------------------------------------- */

static unsigned int __corecount_for_contest( unsigned int cont_i )
{
  const char **cnames = __corenames_for_contest( cont_i );
  if (cnames)
  {
    cont_i = 0;
    while (cnames[cont_i])
      cont_i++;
    return cont_i;
  }
  return 0;  
}

/* ===================================================================== */

void selcoreEnumerateWide( int (*enumcoresproc)(
                            const char **corenames, int idx, void *udata ),
                       void *userdata )
{
  if (enumcoresproc)
  {
    unsigned int corenum;
    for (corenum = 0;;corenum++)
    {
      const char *carray[CONTEST_COUNT];
      int have_one = 0;
      unsigned int cont_i;
      for (cont_i = 0; cont_i < CONTEST_COUNT;cont_i++)
      {
        carray[cont_i] = (const char *)0;
        if (corenum < __corecount_for_contest( cont_i ))
        {
          const char **names = __corenames_for_contest( cont_i );
          carray[cont_i] = names[corenum];
          have_one++;
        }
      }
      if (!have_one)
        break;
      if (! ((*enumcoresproc)( &carray[0], (int)corenum, userdata )) )
        break;
    }
  }  
  return;
}
  
/* ---------------------------------------------------------------------- */

void selcoreEnumerate( int (*enumcoresproc)(unsigned int cont, 
                            const char *corename, int idx, void *udata ),
                       void *userdata )
{
  if (enumcoresproc)
  {
    int stoploop = 0;
    unsigned int cont_i;
    for (cont_i = 0; !stoploop && cont_i < CONTEST_COUNT; cont_i++)
    {
      unsigned int corecount = __corecount_for_contest( cont_i );
      if (corecount)
      {
        unsigned int coreindex;
        const char **corenames = __corenames_for_contest(cont_i);
        for (coreindex = 0; !stoploop && coreindex < corecount; coreindex++)
          stoploop = (! ((*enumcoresproc)(cont_i, 
                      corenames[coreindex], (int)coreindex, userdata )) );
      }
    }
  }
  return;
}  

/* --------------------------------------------------------------------- */

int selcoreValidateCoreIndex( unsigned int cont_i, int idx )
{
  if (idx >= 0 && idx < ((int)__corecount_for_contest( cont_i )))
  {
    if (idx == __apply_selcore_substitution_rules(cont_i, idx))
      return idx;
  }  
  return -1;
}

/* --------------------------------------------------------------------- */

const char *selcoreGetDisplayName( unsigned int cont_i, int idx )
{
  if (idx >= 0 && idx < ((int)__corecount_for_contest( cont_i )))
  {
     const char **names = __corenames_for_contest( cont_i );
     return names[idx];
  }
  return "";
}

/* ===================================================================== */

static struct
{
  int user_cputype[CONTEST_COUNT]; /* what the user has in the ini */
  int corenum[CONTEST_COUNT]; /* what we map it to */
} selcorestatics;
static int selcore_initlev = -1; /* not initialized yet */


int DeinitializeCoreTable( void )  /* ClientMain calls this */
{ 
  if (selcore_initlev <= 0)
  {
    Log("ACK! DeinitializeCoreTable() called for uninitialized table\n");
    return -1;
  }

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

  selcore_initlev--;
  return 0; 
}

/* ---------------------------------------------------------------------- */

int InitializeCoreTable( int *coretypes ) /* ClientMain calls this */
{
  int first_time = 0;
  unsigned int cont_i;

  if (selcore_initlev > 0)
  {
    Log("ACK! InitializeCoreTable() called more than once!\n");
    return -1;
  }
  if (selcore_initlev < 0)
  {
    first_time = 1;
    selcore_initlev = 0;
  }

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

  #if (CLIENT_OS == OS_AIX) && (!defined(_AIXALL)) //not a PPC/POWER hybrid client?
  if (first_time) /* we only want to do this once */
  {
    long detected_type = GetProcessorType(1);
    if (detected_type > 0)
    {
      #if (CLIENT_CPU == CPU_POWER)
      if ((detected_type & (1L<<24)) == 0 ) //not power?
      {
        Log("PANIC::Can't run a Power client on PowerPC architecture.\n");
        return -1;
      }
      #else /* PPC */
      if ((detected_type & (1L<<24)) != 0 ) //is power?
      {
        Log("WARNING::Running a PowerPC client on Power\n"
            "architecture will result in bad performance.\n");
      }
      #endif
    }
  }  
  #endif

  if (first_time) /* we only want to do this once */
  {
    for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
    {
      selcorestatics.user_cputype[cont_i] = -1;
      selcorestatics.corenum[cont_i] = -1;
    }
  }
  if (coretypes)
  {
    for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
    {
      int idx = 0;
      if (__corecount_for_contest( cont_i ) > 0)
      {
        idx = coretypes[cont_i];
        if (idx < 0 || idx >= ((int)__corecount_for_contest( cont_i )))
          idx = -1;
      }      
      if (first_time || idx != selcorestatics.user_cputype[cont_i])
        selcorestatics.corenum[cont_i] = -1; /* got change */
      selcorestatics.user_cputype[cont_i] = idx;
    }
  }

  selcore_initlev++;
  return 0;
}  

/* ---------------------------------------------------------------------- */

#if (CLIENT_OS == OS_AIX )
jmp_buf context;

void sig_invop( int nothing ) {
  longjmp (context, 1);
}
#endif

static long __bench_or_test( int which, 
                            unsigned int cont_i, unsigned int benchsecs, int in_corenum )
{
  long rc = -1;
  #if (CLIENT_OS == OS_AIX)            /* need a signal handler to be able */
  struct sigaction invop, old_handler; /* to test all cores on all systems */
  memset ((void *)&invop, 0, sizeof(invop) );
  invop.sa_handler = sig_invop;
  sigaction(SIGILL, &invop, &old_handler);
  #endif
 
  if (selcore_initlev > 0                  /* core table is initialized? */
      && cont_i < CONTEST_COUNT)           /* valid contest id? */
  {
    /* save current state */
    int user_cputype = selcorestatics.user_cputype[cont_i]; 
    int corenum = selcorestatics.corenum[cont_i];
    int coreidx, corecount = __corecount_for_contest( cont_i );

    rc = 0; /* assume nothing done */
    for (coreidx = 0; coreidx < corecount; coreidx++)
    {
      /* only bench/test cores that won't be automatically substituted */
      if (__apply_selcore_substitution_rules(cont_i, coreidx) == coreidx)
      {
        if (in_corenum < 0)
            selcorestatics.user_cputype[cont_i] = coreidx; /* as if user set it */
        else
        {
          if( in_corenum < corecount )
          {
            selcorestatics.user_cputype[cont_i] = in_corenum;
            coreidx = corecount;
          }
          else  /* invalid core selection, test them all */
          {
            selcorestatics.user_cputype[cont_i] = coreidx;
            in_corenum = -1;
          }
        }   
        selcorestatics.corenum[cont_i] = -1; /* reset to show name */

        #if (CLIENT_OS == OS_AIX)
        if (setjmp(context)) /* setjump will return true if coming from the handler */
        { 
          LogScreen("Error: Core #%i does not work for this system.\n", coreidx);
        } 
        else 
        #endif
        {
          if (which == 's') /* selftest */
            rc = SelfTest( cont_i );
          else 
            rc = TBenchmark( cont_i, benchsecs, 0 );
          #if (CLIENT_OS != OS_WIN32 || !defined(SMC))
          if (rc <= 0) /* failed (<0) or not supported (0) */
            break; /* stop */
          #else
          // HACK! to ignore failed benchmark for x86 rc5 smc core #7 if
          // started from menu and another cruncher is active in background.
          if (rc <= 0) /* failed (<0) or not supported (0) */
          {
            if ( which == 'b' &&  cont_i == RC5 && coreidx == 7 )
              ; /* continue */
            else
              break; /* stop */
          }
          #endif
        }
      }  
    } /* for (coreidx = 0; coreidx < corecount; coreidx++) */

    selcorestatics.user_cputype[cont_i] = user_cputype; 
    selcorestatics.corenum[cont_i] = corenum;

    #if (CLIENT_OS == OS_RISCOS) && defined(HAVE_X86_CARD_SUPPORT)
    if (rc > 0 && cont_i == RC5 && 
          GetNumberOfDetectedProcessors() > 1) /* have x86 card */
    {
      Problem *prob = ProblemAlloc(); /* so bench/test gets threadnum+1 */
      rc = -1; /* assume alloc failed */
      if (prob)
      {
        Log("RC5: using x86 core.\n" );
        if (which != 's') /* bench */
          rc = TBenchmark( cont_i, benchsecs, 0 );
        else
          rc = SelfTest( cont_i );
        ProblemFree(prob);
      }
    }      
    #endif 

  } /* if (cont_i < CONTEST_COUNT) */

  #if (CLIENT_OS == OS_AIX)
  sigaction (SIGILL, &old_handler, NULL);       /* reset handler */
  #endif
  return rc;
}

long selcoreBenchmark( unsigned int cont_i, unsigned int secs, int corenum )
{
  return __bench_or_test( 'b', cont_i, secs, corenum );
}

long selcoreSelfTest( unsigned int cont_i, int corenum)
{
  return __bench_or_test( 's', cont_i, 0, corenum );
}

/* ---------------------------------------------------------------------- */

/* this is called from Problem::LoadState() */
int selcoreGetSelectedCoreForContest( unsigned int contestid )
{
  int corename_printed = 0;
  static long detected_type = -123;
  const char *contname = CliGetContestNameFromID(contestid);

  if (!contname) /* no such contest */
    return -1;
  if (!IsProblemLoadPermitted(-1 /*any thread*/, contestid))
    return -1; /* no cores available */
  if (selcore_initlev <= 0)                /* ACK! selcoreInitialize() */
    return -1;                             /* hasn't been called */

  if (__corecount_for_contest(contestid) == 1) /* only one core? */
    return 0;
  if (selcorestatics.corenum[contestid] >= 0) /* already selected one? */
    return selcorestatics.corenum[contestid];

  if (detected_type == -123) /* haven't autodetected yet? */
  {
    int quietly = 1;
    unsigned int cont_i;
    for (cont_i = 0; quietly && cont_i < CONTEST_COUNT; cont_i++)
    {
      if (__corecount_for_contest(cont_i) < 2)
        ; /* nothing */
      else if (selcorestatics.user_cputype[cont_i] < 0)
        quietly = 0;
    }
    detected_type = GetProcessorType(quietly);
    if (detected_type < 0)
      detected_type = -1;
  }

  #if (CLIENT_CPU == CPU_ALPHA)
  if (contestid == RC5 || contestid == DES) /* old style */
  {
    selcorestatics.corenum[contestid] = selcorestatics.user_cputype[contestid];
    if (selcorestatics.corenum[contestid] < 0)
    {
      /* this is only useful if more than one core, which is currently
         only OSF/DEC-UNIX. If only one core, then that will have been 
         handled in the generic code above, but we play it safe anyway.
      */
      if (detected_type == 5 /*EV5*/ || detected_type == 7 /*EV56*/ ||
          detected_type == 8 /*EV6*/ || detected_type == 9 /*PCA56*/)
        selcorestatics.corenum[contestid] = 1;
      else /* ev3 and ev4 (EV4, EV4, LCA4, EV45) */
        selcorestatics.corenum[contestid] = 0;
    }    
    if (selcorestatics.corenum[contestid] < 0 || 
      selcorestatics.corenum[contestid] >= (int)__corecount_for_contest(contestid))
    {
      selcorestatics.corenum[contestid] = 0;
    }  
  }
  else if (contestid == CSC)
  {
    selcorestatics.corenum[CSC] = selcorestatics.user_cputype[CSC];
    if (selcorestatics.corenum[CSC] < 0)
    {
      ; // Who knows?  Just benchmark them all below
    }
  }
  #elif (CLIENT_CPU == CPU_68K)
  if (contestid == RC5)
  {
    selcorestatics.corenum[RC5] = selcorestatics.user_cputype[RC5];
    if (selcorestatics.corenum[RC5] < 0 && detected_type >= 0)
    {
      selcorestatics.corenum[RC5] = 0;
      #if defined(__GCC__) || defined(__GNUC__) || \
          (CLIENT_OS == OS_AMIGAOS)// || (CLIENT_OS == OS_MACOS)
      if (detected_type >= 68040)
        selcorestatics.corenum[RC5] = 2; /* rc5-060-re-jg.s */
      else if (detected_type >= 68020)
        selcorestatics.corenum[RC5] = 1; /* rc5-020_030-jg.s */
      #endif
    }
  }
  else if (contestid == DES)
  {
    selcorestatics.corenum[DES] = 0; //only one core
  }
  else if (contestid == OGR)
  {
    selcorestatics.corenum[OGR] = selcorestatics.user_cputype[OGR];
    if (selcorestatics.corenum[OGR] < 0 && detected_type > 0)
    {
      int cindex = 0;
      if (detected_type >= 68060)
        cindex = 4;
      else if (detected_type == 68040)
        cindex = 3;
      else if (detected_type == 68030)
        cindex = 2;
      else if (detected_type == 68020)
        cindex = 1;
      selcorestatics.corenum[OGR] = cindex;
    }
  }
  #elif (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_POWER)
  if (contestid == DES)
  {
    selcorestatics.corenum[DES] = 0; /* only one DES core */
  }
  else if (contestid == RC5)
  {
    selcorestatics.corenum[RC5] = selcorestatics.user_cputype[RC5];
    if (selcorestatics.corenum[RC5] < 0 && detected_type > 0)
    {
      int cindex = -1;
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
        case 0x000C: cindex = 1; break; // 7400/7410 (G4) == lintilla
        case 0x8000: cindex = 1; break; // 7450 (G4+)     == lintilla
        default:     cindex =-1; break; // no default (used to be lintilla)
      }
      #if defined(_AIXALL)             /* Power/PPC hybrid */
      if (( detected_type & (1L<<24) ) != 0) //ARCH_IS_POWER?
        cindex = 3;                    /* "PowerRS" */
      #elif (CLIENT_CPU == CPU_POWER)  /* Power only */
        cindex = 3;                    /* "PowerRS" */
      #endif
      #if defined(__VEC__)             /* OS+compiler support altivec */
      if (( detected_type & (1L<<25) ) != 0) //altivec?
        {
          switch ( detected_type & 0xffff) // only compare the low PVR bits
          {
            case 0x000C: cindex = 4; break; // 7400/7410 (G4) == crunch-vec
            case 0x8000: cindex = 5; break; // 7450 (G4+)     == crunch-vec-7450
            default:     cindex =-1; break; // no default
          }
        }
      #endif
      selcorestatics.corenum[RC5] = cindex;
    }
  }
  else if (contestid == CSC)
  {
    selcorestatics.corenum[CSC] = selcorestatics.user_cputype[CSC];
    if (selcorestatics.corenum[CSC] < 0 && detected_type > 0)
    {
      int cindex = -1;/* micro benchmark */
      selcorestatics.corenum[CSC] = cindex;
    }
  }
  else if (contestid == OGR)
  {
    selcorestatics.corenum[OGR] = selcorestatics.user_cputype[OGR];
    if (selcorestatics.corenum[OGR] < 0 && detected_type > 0)
    {
      int cindex = 0;                   /* PPC-scalar */

      #if defined (_AIXALL)             /* Power/PPC hybrid */
      if (( detected_type & (1L<<24) ) != 0) /* ARCH_IS_POWER? */
        cindex = 1;                     /* PowerRS */ 
      #elif (CLIENT_CPU == CPU_POWER)   /* Power only */
        cindex = 1;                     /* "PowerRS" */
      #endif
      #if defined(__VEC__)              /* OS+compiler support altivec */
      if (( detected_type & (1L<<25) ) != 0) /* proc supports altivec? */
        cindex = 2;                     /* PPC-vector */
      #endif

      selcorestatics.corenum[OGR] = cindex;
    }
  }
  #elif (CLIENT_CPU == CPU_X86)
  {
    if (contestid == RC5)
    {
      selcorestatics.corenum[RC5] = selcorestatics.user_cputype[RC5];
      if (selcorestatics.corenum[RC5] < 0)
      {
        if (detected_type >= 0)
        {
          int cindex = -1; 
          int have_smc = 0;
          int have_mmx = ((detected_type & 0x100)!=0);
          #if defined(SMC)
          have_smc = (x86_smc_initialized > 0);
          #endif
          switch (detected_type & 0xff)
          {
            case 0x00: cindex = ((have_mmx)?(9):(0)); break; // P5 == RG/BRF class 5
            case 0x01: cindex = ((have_smc)?(7 /*#99*/):(6 /*#1939*/)); break; // 386/486
            case 0x02: cindex = 2; break; // PII/PIII   == RG class 6
            case 0x03: cindex = 3; break; // Cx6x86/MII == RG re-pair I (#1913)
            case 0x04: cindex = 4; break; // K5         == RG RISC-rotate I
            case 0x05: cindex = 5; break; // K6-1/2/3   == RG RISC-rotate II
            case 0x06: cindex = ((have_smc)?(7 /*#99*/):(6 /*#804*/)); break; // cx486
            case 0x07: cindex = 2; break; // Celeron    == RG class 6
            case 0x08: cindex = 2; break; // PPro       == RG class 6
            case 0x09: cindex = 6; break; // AMD>=K7/Cx>MII == RG/DG re-pair III
            case 0x0A: cindex = 6; break; // Centaur C6 == RG/DG re-pair III (#2082)
            case 0x0B: cindex = 8; break; // P4         == ak/P4 
            default:   cindex =-1; break; // no default
          }
          #if defined(HAVE_NO_NASM)
          if (cindex == 6)   /* ("RG/DG re-pair III") */
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
          selcorestatics.corenum[RC5] = cindex;
        }
      }
    }     
    else if (contestid == DES)
    {  
      selcorestatics.corenum[DES] = selcorestatics.user_cputype[DES];
      if (selcorestatics.corenum[DES] < 0)
      {
        if (detected_type >= 0)
        {
          int cindex = -1;
          switch ( detected_type & 0xff )
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
          if ((detected_type & 0x100) != 0) /* have mmx */
            cindex = 2; /* mmx bitslicer */
          #endif
          selcorestatics.corenum[DES] = cindex;
        }
      }
    }
    else if (contestid == CSC)
    {
      selcorestatics.corenum[CSC] = selcorestatics.user_cputype[CSC];
      if (selcorestatics.corenum[CSC] < 0)
      {
        if (detected_type >= 0)
        {
          int cindex = -1; 
          // this is only valid for nasm'd cores or GCC 2.95 and up
          switch ( detected_type & 0xff )
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
          if ((detected_type & 0x100) != 0) /* have mmx */
            cindex = 1; /* == 6bit - called - MMX */
          #endif
          selcorestatics.corenum[CSC] = cindex;
        }
      }
    }
    else if (contestid == OGR)
    {
      selcorestatics.corenum[OGR] = selcorestatics.user_cputype[OGR];
      if (selcorestatics.corenum[OGR] < 0)
      {
        if (detected_type >= 0)
        {
          int cindex = -1; 
          switch ( detected_type & 0xff )
          {
            case 0x00: cindex = 1; break; // P5           == without BSR (B)
            case 0x01: cindex = 1; break; // 386/486      == without BSR (B)
            case 0x02: cindex = 0; break; // PII/PIII     == with BSR (A)
            case 0x03: cindex = 0; break; // Cx6x86       == with BSR (A)
            case 0x04: cindex = 1; break; // K5           == without BSR (B)
            case 0x05: cindex = 0; break; // K6/K6-2/K6-3 == without BSR (B)
            case 0x06: cindex = 1; break; // Cyrix 486    == without BSR (B)
            case 0x07: cindex = 0; break; // orig Celeron == with BSR (A)
            case 0x08: cindex = 0; break; // PPro         == with BSR (A)
            case 0x09: cindex = 0; break; // AMD K7       == with BSR (A)
            case 0x0A: cindex = 1; break; // Centaur C6   == without BSR (B)
            case 0x0B: cindex = 1; break; // Pentium 4    == without BSR (B)
            default:   cindex =-1; break; // no default
          }
          selcorestatics.corenum[OGR] = cindex;
        }
      }
    }
  }
  #elif (CLIENT_CPU == CPU_ARM)
  if (contestid == RC5)
  {
    selcorestatics.corenum[RC5] = selcorestatics.user_cputype[RC5];
    if (selcorestatics.corenum[RC5] < 0 && detected_type > 0)
    {
      int cindex = -1;
      if (detected_type == 0x300  || /* ARM 3 */ 
          detected_type == 0x600  || /* ARM 600 */
          detected_type == 0x610  || /* ARM 610 */
          detected_type == 0x700  || /* ARM 700 */
          detected_type == 0x710  || /* ARM 710 */
          detected_type == 0x7500 || /* ARM 7500 */ 
          detected_type == 0x7500FE) /* ARM 7500FE */
        cindex = 0;
      else if (detected_type == 0x810 || /* ARM 810 */
               detected_type == 0xA10)   /* StrongARM 110 */
        cindex = 1;
      else if (detected_type == 0x200 || /* ARM 2 */
               detected_type == 0x250)   /* ARM 250 */
        cindex = 2;
      selcorestatics.corenum[RC5] = cindex;
    }
  }
  else if (contestid == DES)
  {
    selcorestatics.corenum[DES] = selcorestatics.user_cputype[DES];
    if (selcorestatics.corenum[DES] < 0 && detected_type > 0)
    {
      int cindex = -1;
      if (detected_type == 0x810 ||  /* ARM 810 */
          detected_type == 0xA10 ||  /* StrongARM 110 */
          detected_type == 0x200 ||  /* ARM 2 */
          detected_type == 0x250)    /* ARM 250 */
        cindex = 1;
      else /* "ARM 3, 610, 700, 7500, 7500FE" or  "ARM 710" */
        cindex = 0;
      selcorestatics.corenum[DES] = cindex;  
    }  
  }
  #elif (CLIENT_CPU == CPU_SPARC)
  if (contestid == RC5)
  {
    selcorestatics.corenum[RC5] = selcorestatics.user_cputype[RC5];
    #if (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS)
    if (selcorestatics.corenum[RC5] < 0)
      selcorestatics.corenum[RC5] = 0; // ultra-crunch is faster on everything I found ...
    #elif (CLIENT_OS == OS_LINUX)
    // run micro benchmark
    #endif
  }
  #endif

  if (selcorestatics.corenum[contestid] < 0)
    selcorestatics.corenum[contestid] = selcorestatics.user_cputype[contestid];

  if (selcorestatics.corenum[contestid] < 0) /* ok, bench it then */
  {
    int corecount = (int)__corecount_for_contest(contestid);
    selcorestatics.corenum[contestid] = 0;
    if (corecount > 0)
    {
      int whichcrunch, saidmsg = 0, fastestcrunch = -1;
      unsigned long fasttime = 0;

      for (whichcrunch = 0; whichcrunch < corecount; whichcrunch++)
      {
        /* test only if not substituted */
        if (whichcrunch == __apply_selcore_substitution_rules(contestid, 
                                                              whichcrunch))
        {                                                              
          long rate;
          selcorestatics.corenum[contestid] = whichcrunch;
          if (!saidmsg)
          {
            LogScreen("%s: Running micro-bench to select fastest core...\n", 
                      contname);
            saidmsg = 1;
          }                                
          if ((rate = TBenchmark( contestid, 2, TBENCHMARK_QUIET | TBENCHMARK_IGNBRK )) > 0)
          {
#ifdef DEBUG
            LogScreen("%s Core %d: %d keys/sec\n", contname,whichcrunch,rate);
#endif
            if (fastestcrunch < 0 || ((unsigned long)rate) > fasttime)
            {
              fasttime = rate;
              fastestcrunch = whichcrunch; 
            }
          }  
        }
      }

      if (fastestcrunch < 0) /* all failed */
        fastestcrunch = 0; /* don't bench again */
      selcorestatics.corenum[contestid] = fastestcrunch;
    }
  }

  if (selcorestatics.corenum[contestid] >= 0) /* didn't fail */
  { 
    /* 
    ** substitution according to real selcoreSelectCore() rules
    ** Returns original if no substitution occurred.
    */ 
    int override = __apply_selcore_substitution_rules(contestid, 
                                       selcorestatics.corenum[contestid]);
    if (!corename_printed)
    {
      if (override != selcorestatics.corenum[contestid])
      {
        Log("%s: selected core #%d (%s)\n"
            "     is not supported by this client/OS/architecture.\n"
            "     Using core #%d (%s) instead.\n", contname, 
                 selcorestatics.corenum[contestid], 
                 selcoreGetDisplayName(contestid, selcorestatics.corenum[contestid]),
                 override, selcoreGetDisplayName(contestid, override) );
      }
      else               
      {
       Log("%s: using core #%d (%s).\n", contname, 
           selcorestatics.corenum[contestid], 
           selcoreGetDisplayName(contestid, selcorestatics.corenum[contestid]) );
      }      
    }  
    selcorestatics.corenum[contestid] = override;
  }  

  return selcorestatics.corenum[contestid];
}

/* ---------------------------------------------------------------------- */

#if 0
// available ANSI cores:
// 2 pipeline: rc5/ansi/rc5ansi_2-rg.cpp
//   extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
//   extern "C" s32 rc5_ansi_rg_unified_form( RC5UnitWork *work,
//                                    u32 *iterations, void *scratch_area );
// 1 pipeline: rc5/ansi/rc5ansi1-b2.cpp
//   extern "C" u32 rc5_ansi_1_b2_rg_unit_func( RC5UnitWork *, u32 iterations );
#endif                                                                

#if (CLIENT_CPU == CPU_UNKNOWN)
  // rc5/ansi/rc5ansi_2-rg.cpp
  extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
#elif (CLIENT_CPU == CPU_X86)
  extern "C" u32 rc5_unit_func_486( RC5UnitWork * , u32 iterations );
  extern "C" u32 rc5_unit_func_p5( RC5UnitWork * , u32 iterations );
  extern "C" u32 rc5_unit_func_p6( RC5UnitWork * , u32 iterations );
  extern "C" u32 rc5_unit_func_6x86( RC5UnitWork * , u32 iterations );
  extern "C" u32 rc5_unit_func_k5( RC5UnitWork * , u32 iterations );
  extern "C" u32 rc5_unit_func_k6( RC5UnitWork * , u32 iterations );
  extern "C" u32 rc5_unit_func_p5_mmx( RC5UnitWork * , u32 iterations );
  extern "C" u32 rc5_unit_func_k6_mmx( RC5UnitWork * , u32 iterations );
  //extern "C" u32 rc5_unit_func_486_smc( RC5UnitWork * , u32 iterations );
  extern "C" u32 rc5_unit_func_k7( RC5UnitWork * , u32 iterations );
  extern "C" u32 rc5_unit_func_p7( RC5UnitWork *, u32 iterations );
#elif (CLIENT_CPU == CPU_ARM)
  extern "C" u32 rc5_unit_func_arm_1( RC5UnitWork * , u32 );
  extern "C" u32 rc5_unit_func_arm_2( RC5UnitWork * , u32 );
  extern "C" u32 rc5_unit_func_arm_3( RC5UnitWork * , u32 );
  #if (CLIENT_OS == OS_RISCOS) && defined(HAVE_X86_CARD_SUPPORT)
  extern "C" u32 rc5_unit_func_x86( RC5UnitWork * , u32 );
  #endif
#elif (CLIENT_CPU == CPU_S390)
  // rc5/ansi/rc5ansi_2-rg.cpp
  extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
#elif (CLIENT_CPU == CPU_IA64)
  // rc5/ansi/rc5ansi_4-rg.cpp
  extern "C" u32 rc5_unit_func_ansi_4_rg( RC5UnitWork *, u32 iterations );
#elif (CLIENT_CPU == CPU_PA_RISC)
  // rc5/parisc/parisc.cpp encapulates parisc.s, 2 pipelines
  // extern "C" u32 rc5_parisc_unit_func( RC5UnitWork *, u32 );
  extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 );
#elif (CLIENT_CPU == CPU_SH4)
  extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 );
#elif (CLIENT_CPU == CPU_88K) //OS_DGUX
  // rc5/ansi/rc5ansi_2-rg.cpp
  extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
#elif (CLIENT_CPU == CPU_MIPS)
  #if (CLIENT_OS == OS_ULTRIX) || (CLIENT_OS == OS_IRIX) || \
      (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_NETBSD)
    // rc5/ansi/rc5ansi_2-rg.cpp
    extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
  #elif (CLIENT_OS == OS_SINIX)
    //rc5/mips/mips-crunch.cpp or rc5/mips/mips-irix.S
    extern "C" u32 rc5_unit_func_mips_crunch( RC5UnitWork *, u32 );
  #else
    #error "What's up, Doc?"
  #endif
#elif (CLIENT_CPU == CPU_SPARC)
  #if ((CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS) || (CLIENT_OS == OS_LINUX))
    //rc5/ultra/rc5-ultra-crunch.cpp
    extern "C" u32 rc5_unit_func_ultrasparc_crunch( RC5UnitWork * , u32 );
    // rc5/ansi/2-rg.cpp
    extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
  #else
    // rc5/ansi/2-rg.cpp
    extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
  #endif
#elif (CLIENT_CPU == CPU_68K)
  #if defined(__GCC__) || defined(__GNUC__) || \
      (CLIENT_OS == OS_AMIGAOS)// || (CLIENT_OS == OS_MACOS)
    extern "C" u32 rc5_unit_func_000_010re( RC5UnitWork *, u32 );
    extern "C" u32 rc5_unit_func_020_030( RC5UnitWork *, u32 );
    extern "C" u32 rc5_unit_func_060re( RC5UnitWork *, u32 );
  #elif (CLIENT_OS == OS_MACOS)
    // rc5/68k/crunch.68k.a.o
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
    // rc5/ansi/rc5ansi_2-rg.cpp
    extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
  #endif
  #if (CLIENT_CPU == CPU_POWERPC) || defined(_AIXALL)
    #if (CLIENT_OS == OS_WIN32) //NT has poor PPC assembly
      // rc5/ansi/rc5ansi_2-rg.cpp
      extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
    #else
      // rc5/ppc/rc5_*.cpp
      // although Be OS isn't supported on 601 machines and there is
      // is no 601 PPC board for the Amiga, lintilla depends on allitnil,
      // so we have both anyway, we may as well support both.
      // rc5ansi_2-rg.cpp for AIX is allready prototyped above
      extern "C" u32 rc5_unit_func_allitnil_compat( RC5UnitWork *, u32 );
      extern "C" u32 rc5_unit_func_lintilla_compat( RC5UnitWork *, u32 );
      extern "C" u32 rc5_unit_func_lintilla_604_compat( RC5UnitWork *, u32 );
      #if defined(__VEC__) /* OS+compiler support altivec */
        extern "C" u32 rc5_unit_func_vec_compat( RC5UnitWork *, u32 );
        extern "C" u32 rc5_unit_func_vec_7450_compat( RC5UnitWork *, u32 );
      #endif
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
#endif

/* ------------------------------------------------------------- */

#if defined(HAVE_CSC_CORES)
  extern "C" s32 csc_unit_func_1k  ( RC5UnitWork *, u32 *iterations, void *membuff );
  #if (CLIENT_CPU != CPU_ARM) // ARM only has one CSC core
  extern "C" s32 csc_unit_func_1k_i( RC5UnitWork *, u32 *iterations, void *membuff );
  extern "C" s32 csc_unit_func_6b  ( RC5UnitWork *, u32 *iterations, void *membuff );
  extern "C" s32 csc_unit_func_6b_i( RC5UnitWork *, u32 *iterations, void *membuff );
  #endif
  #if (CLIENT_CPU == CPU_X86) && !defined(HAVE_NO_NASM)
  extern "C" s32 csc_unit_func_6b_mmx ( RC5UnitWork *, u32 *iterations, void *membuff );
  #endif
#endif

/* ------------------------------------------------------------- */

#if defined(HAVE_OGR_CORES)
  #if (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_POWER)
      extern "C" CoreDispatchTable *ogr_get_dispatch_table(void);
      #if defined(_AIXALL)      /* AIX hybrid client */
      extern "C" CoreDispatchTable *ogr_get_dispatch_table_power();
      #endif
      #if defined(__VEC__)      /* compilor supports AltiVec */
      extern "C" CoreDispatchTable *vec_ogr_get_dispatch_table(void);
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
  #else
      extern "C" CoreDispatchTable *ogr_get_dispatch_table(void);
  #endif
#endif    

/* ------------------------------------------------------------- */

int selcoreSelectCore( unsigned int contestid, unsigned int threadindex,
                       int *client_cpuP, struct selcore *selinfo )
{                               
  int use_generic_proto = 0; /* if rc5/des unit_func proto is generic */
  unit_func_union unit_func; /* declared in problem.h */
  int cruncher_is_asynchronous = 0; /* on a co-processor or similar */
  int pipeline_count = 2; /* most cases */
  int client_cpu = CLIENT_CPU; /* usual case */
  int coresel = selcoreGetSelectedCoreForContest( contestid );
  if (coresel < 0)
    return -1;
  memset( &unit_func, 0, sizeof(unit_func));

  /* -------------------------------------------------------------- */

  if (contestid == RC5) /* avoid switch */
  {
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
          (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_NETBSD)
      {
        // rc5/ansi/rc5ansi_2-rg.cpp
        //xtern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 );
        unit_func.rc5 = rc5_unit_func_ansi_2_rg;
        pipeline_count = 2;
        coresel = 0;
      }
      #elif (CLIENT_OS == OS_SINIX)
      {
        //rc5/mips/mips-crunch.cpp or rc5/mips/mips-irix.S
        //xtern "C" u32 rc5_unit_func_mips_crunch( RC5UnitWork *, u32 );
        unit_func.rc5 = rc5_unit_func_mips_crunch;
        pipeline_count = 2;
        coresel = 0;
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
      #if defined(__GCC__) || defined(__GNUC__) || \
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
    #elif (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_POWER)
    {
      #if (CLIENT_CPU == CPU_POWER) && !defined(_AIXALL) //not hybrid
      {
        // rc5/ansi/rc5ansi_2-rg.cpp
        //xtern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 );
        unit_func.rc5 = rc5_unit_func_ansi_2_rg; //POWER CPU
        pipeline_count = 2;
        coresel = 3;
      }
      #elif (CLIENT_OS == OS_WIN32) 
      {
        //  rc5/ansi/rc5ansi_2-rg.cpp
        //  xtern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32  );
        unit_func.rc5 = rc5_unit_func_ansi_2_rg;
        pipeline_count = 2;
        coresel = 3;
      }
      #else
      { 
        client_cpu = CPU_POWERPC;
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
        #if defined(_AIXALL) // POWER/POWERPC hybrid client
        else if (coresel == 3)    // PowerRS
        {
          client_cpu = CPU_POWER;
          unit_func.rc5 = rc5_unit_func_ansi_2_rg; //rc5/ansi/rc5ansi_2-rg.cpp
          pipeline_count = 2;
        }
        #endif
        #if defined(__VEC__)
        else if (coresel == 4)    // G4 (PPC 7400/7410)
        {
          unit_func.rc5 = rc5_unit_func_vec_compat;
          pipeline_count = 1;
        }
        else if (coresel == 5)    // G4 (PPC 7450)
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
  } /* if (contestid == RC5) */
  
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
  } /* if (contestid == DES) */
  #endif /* #ifdef HAVE_DES_CORES */

  /* ================================================================== */

  #if defined(HAVE_OGR_CORES)
  if (contestid == OGR)
  {
    #if (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_POWER)
      //extern "C" CoreDispatchTable *ogr_get_dispatch_table(void);
      //extern "C" CoreDispatchTable *vec_ogr_get_dispatch_table(void);
      //extern "C" CoreDispatchTable *ogr_get_dispatch_power(void);
      #if defined(_AIXALL)                       /* maybe ARCH_IS_POWER */
      if (coresel == 1)                          /* "PowerRS" */
        unit_func.ogr = ogr_get_dispatch_table_power();
      #endif
      #if defined(__VEC__)          /* compiler+OS supports AltiVec */
      if (coresel == 2)                           /* "PPC-vector" */
        unit_func.ogr = vec_ogr_get_dispatch_table();
      #endif
      if (!unit_func.ogr)
      {
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
    #elif (CLIENT_CPU == CPU_X86)
      if (coresel == 0) //A
        unit_func.ogr = ogr_get_dispatch_table(); //A
      else 
      {
        unit_func.ogr = ogr_get_dispatch_table_nobsr(); //B
        coresel = 1;
      }  
    #else
      //extern "C" CoreDispatchTable *ogr_get_dispatch_table(void);
      unit_func.ogr = ogr_get_dispatch_table();
      coresel = 0;
    #endif
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
  }
  #endif /* #ifdef HAVE_CSC_CORES */

  /* ================================================================== */

  if (coresel >= 0 && unit_func.gen && 
     coresel < ((int)__corecount_for_contest( contestid )))
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
