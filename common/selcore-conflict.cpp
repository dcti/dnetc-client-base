/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * -------------------------------------------------------------------
 * program (pro'-gram) [vi]: To engage in a pastime similar to banging
 * one's head against a wall but with fewer opportunities for reward.
 * -------------------------------------------------------------------
 */
const char *selcore_cpp(void) {
return "@(#)$Id: selcore-conflict.cpp,v 1.47.2.41 2000/01/08 12:57:21 snake Exp $"; }


#include "cputypes.h"
#include "client.h"    // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "problem.h"   // problem class
#include "cpucheck.h"  // cpu selection, GetTimesliceBaseline()
#include "logstuff.h"  // Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "clicdata.h"  // GetContestNameFromID()
#include "bench.h"     // TBenchmark()
#include "selftest.h"  // SelfTest()
#include "selcore.h"   // keep prototypes in sync

/* ------------------------------------------------------------------------ */

static const char **__corenames_for_contest( unsigned int cont_i )
{
  /* 
   When selecting corenames, use names that describe how (what optimization)
   they are different from their predecessor(s). If only one core,
   use the obvious "MIPS optimized" or similar.
  */
  #define LARGEST_SUBLIST 7 /* including the terminating null */
  static const char *corenames_table[CONTEST_COUNT][LARGEST_SUBLIST]= 
  #undef LARGEST_SUBLIST
  {
  #if (CLIENT_CPU == CPU_X86)
    { /* RC5 */
      /* we should be using names that tell us how the cores are different
         (just like "bryd" and "movzx bryd")
      */
      "RG/BRF class 5", /* P5/Am486/Cx486 - may become P5MMX at runtime*/
      "RG class 3/4",   /* 386/486 - may become SMC at runtime */
      "RG class 6",     /* PPro/II/III */
      "RG Cx re-pair",  /* Cyrix 6x86[MX]/M2, AMD K7 */
      "RG RISC-rotate I", /* K5 */
      "RG RISC-rotate II", /* K6 - may become mmx-k6-2 core at runtime */
      NULL
    },
    { /* DES */
      "byte Bryd",
      "movzx Bryd",
      #if defined(MMX_BITSLICER) || defined(CLIENT_SUPPORTS_SMP) 
      "Kwan/Bitslice", /* may become MMX bitslice at runtime */
      #endif
      NULL
    },
  #elif (CLIENT_CPU == CPU_ARM)
    { /* RC5 */
      "Series A core", /* (autofor for ARM 3/6xx/7xxx) "ARM 3, 610, 700, 7500, 7500FE" */
      "Series B core", /* (autofor ARM 8xx/StrongARM) "ARM 810, StrongARM 110" */
      "Series C core", /* (autofor ARM 2xx) "ARM 2, 250" */
      NULL             /* "ARM 710" */
    },
    { /* DES */
      "Standard ARM core", /* "ARM 3, 610, 700, 7500, 7500FE" or  "ARM 710" */
      "StrongARM optimized core", /* "ARM 810, StrongARM 110" or "ARM 2, 250" */
      NULL
    },
  #elif (CLIENT_CPU == CPU_68K)
    { /* RC5 */
    #if (CLIENT_OS == OS_AMIGAOS)
      "loopy",    /* 68000/10/20/30 */
      "unrolled", /* 40/60 */
    #elif defined(__GCC__) || defined(__GNUC__)
      "68k asm cruncher",
    #else
      "Generic RC5 core",
    #endif
      NULL
    },
    { /* DES */
      "Generic DES core", 
      NULL
    },
  #elif (CLIENT_CPU == CPU_ALPHA) 
    { /* RC5 */
      #if (CLIENT_OS == OS_DEC_UNIX)
      "ev3 and ev4 optimized",
      "ev5 and ev6 optimized",
      #elif (CLIENT_OS == OS_WIN32)
      "Marcelais",
      #else
      "axp bmeyer",
      #endif
      NULL
    },
    { /* DES */
      #if (CLIENT_OS == OS_DEC_UNIX)
      "ev3 and ev4 optimized",
      "ev5 and ev6 optimized",
      #else
      "dworz/amazing",
      #endif
      NULL
    },
  #elif (CLIENT_CPU == CPU_POWERPC) || (CLIENT_OS == OS_POWER)
    { /* RC5 */
      /* lintilla depends on allitnil, and since we need both even on OS's 
         that don't support the 601, we may as well "support" them visually.
         On POWER/PowerPC hybrid clients ("_AIXALL"), running on a POWER
         CPU, core #0 becomes "RG AIXALL", and core #1 disappears.
       */
      "allitnil",
      "lintilla",
      NULL, /* this may become the G4 vector core at runtime */
      NULL
    },
    { /* DES */
      "Generic DES core", 
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
  #endif  
    { /* OGR */
      "Standard OGR core",
      NULL
    },
    { /* CSC */
#if (CLIENT_CPU != CPU_ARM)
      "6 bit - inline", 
      "6 bit - called",
      "1 key - inline", 
#endif
      "1 key - called",
      NULL, /* room */
      NULL
    }
  };  
  static int fixed_up = -1;
  if (fixed_up < 0)
  {
    #if (CLIENT_CPU == CPU_X86)
    {
      long det = GetProcessorType(1);
      #ifdef SMC /* actually only for the first thread */
      corenames_table[RC5][1] = "RG self-modifying";
      #endif
      if (det >= 0 && (det & 0x100)!=0) /* ismmx */
      {
        #if defined(MMX_RC5)
        corenames_table[RC5][0] = "jasonp P5/MMX"; /* slower on a PII/MMX */
        #endif
        #if defined(MMX_RC5_AMD)
        corenames_table[RC5][5] = "BRF Kx/MMX"; /* is this k6-2 only? */
        #endif
        #ifdef MMX_BITSLICER
        corenames_table[DES][2] = "BRF MMX bitslice";
        #endif
        #if defined(MMX_CSC)
        corenames_table[CSC][1] = "6 bit - bitslice";//replaces '6 bit - called'
        #endif
      }
    }
    #endif
    #if (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_POWER)
    {
      long det = GetProcessorType(1);
      if (det < 0) 
        ; /* error, Power never errors though */
      else if (( det & (1L<<24) ) != 0) //ARCH_IS_POWER
      {                               //only one core - (ansi)
        corenames_table[RC5][0] = "RG AIXALL (Power CPU)",
        corenames_table[RC5][1] = NULL;
      }
      #if (CLIENT_OS == OS_MACOS)
      else if ( det == 12) //PPC 7500
      {
        corenames_table[RC5][2] = "crunch-vec"; /* aka rc5_unit_func_vec() wrapper */
        corenames_table[RC5][3] = NULL;
      }
      #endif
    }
    #endif
    fixed_up = 1;  
  }
  if (cont_i < CONTEST_COUNT)
  {
    return (const char **)(&(corenames_table[cont_i][0]));
  }
  return ((const char **)0);
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

void selcoreEnumerateWide( int (*proc)(
                            const char **corenames, int idx, void *udata ),
                       void *userdata )
{
  if (proc)
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
      if (! ((*proc)( &carray[0], (int)corenum, userdata )) )
        break;
    }
  }  
  return;
}
  
/* ---------------------------------------------------------------------- */

void selcoreEnumerate( int (*proc)(unsigned int cont, 
                            const char *corename, int idx, void *udata ),
                       void *userdata )
{
  if (proc)
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
          stoploop = (! ((*proc)(cont_i, 
                      corenames[coreindex], (int)coreindex, userdata )) );
      }
    }
  }
  return;
}  

/* --------------------------------------------------------------------- */

int selcoreValidateCoreIndex( unsigned int cont_i, int index )
{
  if (index >= 0 && index < ((int)__corecount_for_contest( cont_i )))
    return index;
  return -1;
}

/* --------------------------------------------------------------------- */

const char *selcoreGetDisplayName( unsigned int cont_i, int index )
{
  if (index >= 0 && index < ((int)__corecount_for_contest( cont_i )))
  {
     const char **names = __corenames_for_contest( cont_i );
     return names[index];
  }
  return "";
}

/* ===================================================================== */

static struct
{
  int user_cputype[CONTEST_COUNT]; /* what the user has in the ini */
  int corenum[CONTEST_COUNT]; /* what we map it to */
} selcorestatics;

/* ---------------------------------------------------------------------- */

int DeinitializeCoreTable( void ) { return 0; }

int InitializeCoreTable( int *coretypes ) /* ClientMain calls this */
{
  static int initialized = -1;
  unsigned int cont_i;
  if (initialized < 0)
  {
    for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
    {
      selcorestatics.user_cputype[cont_i] = -1;
      selcorestatics.corenum[cont_i] = -1;
    }
    initialized = 0;
  }
  if (coretypes)
  {
    for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
    {
      int index = 0;
      if (__corecount_for_contest( cont_i ) > 1)
        index = selcoreValidateCoreIndex( cont_i, coretypes[cont_i] );
      if (!initialized || index != selcorestatics.user_cputype[cont_i])
        selcorestatics.corenum[cont_i] = -1; /* got change */
      selcorestatics.user_cputype[cont_i] = index;
    }
    initialized = 1;
  }
  if (initialized > 0)
    return 0;
  return -1;
}  

/* ---------------------------------------------------------------------- */

static long __bench_or_test( int which, 
                            unsigned int cont_i, unsigned int benchsecs )
{
  long rc = -1;
  
  if (InitializeCoreTable(((int *)0)) < 0) /* ACK! selcoreInitialize() */
    return -1;                             /* hasn't been called */

  if (cont_i < CONTEST_COUNT)
  {
    /* save current state */
    int user_cputype = selcorestatics.user_cputype[cont_i]; 
    int corenum = selcorestatics.corenum[cont_i];
    unsigned int coreidx, corecount = __corecount_for_contest( cont_i );
    rc = 0;
    for (coreidx = 0; coreidx < corecount; coreidx++)
    {
      selcorestatics.user_cputype[cont_i] = coreidx; /* as if user set it */
      selcorestatics.corenum[cont_i] = -1; /* reset to show name */
      if (which == 's') /* selftest */
      {
        int irc = SelfTest( cont_i );
        if (irc <= 0) /* failed or not supported */
        {
          rc = (long)irc;
          break; /* test failed. stop */
        }
      }
      else if ((rc = TBenchmark( cont_i, benchsecs, 0 )) <= 0)
        break; /* failed/not supported for this contest */

      #if (CLIENT_OS == OS_RISCOS) && defined(HAVE_X86_CARD_SUPPORT)
      if (cont_i == RC5 && coreidx == (corecount-1) &&
          GetNumberOfDetectedProcessors() > 1) /* have x86 card */
      {
        Problem *prob = new Problem(); /* so bench/test gets threadnum+1 */
        rc = 1;
        Log("RC5: using x86 core.\n" );
        if (which != 's') /* bench */
          rc = TBenchmark( cont_i, benchsecs, 0 );
        else
        {
          int irc = SelfTest( cont_i );
          if (irc <= 0) /* failed or not supported */
            rc = (long)irc;
        }
        delete prob;
        if (rc <= 0) 
          break; /* failed/not supported for this contest */
      }      
      #endif 
    }
    selcorestatics.user_cputype[cont_i] = user_cputype; 
    selcorestatics.corenum[cont_i] = corenum;
  }
  return rc;
}

int selcoreBenchmark( unsigned int cont_i, unsigned int secs )
{
  return __bench_or_test( 'b', cont_i, secs );
}

int selcoreSelfTest( unsigned int cont_i )
{
  return (int)__bench_or_test( 's', cont_i, 0 );
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

  if (InitializeCoreTable(((int *)0)) < 0) /* ACK! selcoreInitialize() */
    return -1;                             /* hasn't been called */

  if (__corecount_for_contest(contestid) == 1) /* only one core? */
    return 0;
  if (selcorestatics.corenum[contestid] >= 0) /* already selected one? */
    return selcorestatics.corenum[contestid];

  if (detected_type == -123) /* haven't autodetected yet? */
  {
    detected_type = GetProcessorType(1 /* quietly */);
    if (detected_type < 0)
      detected_type = -1;
    else
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
      if (!quietly)
        GetProcessorType(0);
    }
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
      selcorestatics.corenum[RC5] = 0; /* rc5-000_030-jg.s */
      if (detected_type >= 68040)
        selcorestatics.corenum[RC5] = 1; /* rc5-040_060-jg.s */
    }
  }
  else if (contestid == DES)
  {
    selcorestatics.corenum[DES] = 0; //only one core
  }
  #elif (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_POWER)
  #if (!defined(_AIXALL)) //not a PPC/POWER hybrid client?
  if (detected_type >= 0)
  {
    #if (CLIENT_CPU == CPU_POWER)
    if ((detected_type & (1L<<24)) == 0 ) //not power?
    {
      Log("PANIC::Can't run a PowerPC client on Power architecture\n");
      return -1; //this is a good place to abort()
    }
    #else /* PPC */
    if ((detected_type & (1L<<24)) != 0 ) //is power?
    {
      Log("PANIC::Can't run a Power client on PowerPC architecture\n");
      return -1; //this is a good place to abort()
    }
    #endif
  }  
  #endif
  if (contestid == DES)
  {
    selcorestatics.corenum[DES] = 0; /* only one DES core */
  }
  else if (contestid == RC5)
  {
    /* lintilla depends on allitnil, and since we need both even on OS's 
       that don't support the 601, we may as well "support" them visually.
    */
    selcorestatics.corenum[RC5] = selcorestatics.user_cputype[RC5];
    if (selcorestatics.corenum[RC5] < 0 && detected_type >= 0)
    {
      int cindex = -1;
      if (( detected_type & (1L<<24) ) != 0) //ARCH_IS_POWER
        cindex = 0;                 //only one core - (ansi)
      else if (detected_type == 1 )    //PPC 601
        cindex = 0;               // lintilla
      #if (CLIENT_OS == OS_MACOS) /* vec core is currently macos only */
      else if (detected_type == 12) //PPC 7500
        cindex = 2;               // vector
      #endif
      else                        //the rest
        cindex = 1;               // allitnil
      selcorestatics.corenum[RC5] = cindex;
    }
  }
  else if (contestid == CSC)
  {
    selcorestatics.corenum[CSC] = selcorestatics.user_cputype[CSC];
    if (selcorestatics.corenum[CSC] < 0 && detected_type > 0)
    {
      int cindex = -1;
      if ((detected_type & (1L<<24) ) != 0) //ARCH_IS_POWER
        ; //don't know yet
      else 
      {
       /*
        long det = (detected_type & 0x00ffffffL);
        if (det == 1)       //PPC 601
          cindex = 2;       // G1: 16k L1 cache - 1 key inline
        else if (det == 12) //PPC 7400
          cindex = 1;       // G4: 64k L1 cache - 6 bit called
        //don't know about the rest

        Uh, whats this? I disable this for now and thus let the client
        do a mini bench at startup - a wrong core was selected for G4 CPUs
        
        */
      }
      selcorestatics.corenum[CSC] = cindex;
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
          switch ( detected_type & 0xff )
          {
            case 0: cindex = 0; break; // P5
            case 1: cindex = 1; break; // 386/486
            case 2: cindex = 2; break; // PII/PIII
            case 3: cindex = 3; break; // Cx6x86
            case 4: cindex = 4; break; // K5
            case 5: cindex = 5; break; // K6/K6-2/K6-3
            #if defined(SMC)    
            case 6: cindex = 1; break; // cyrix 486 uses SMC if available
            #else 
            case 6: cindex = 0; break; // else default to P5 (see /bugs/ #99)
            #endif
            case 7: cindex = 2; break; // castrated Celeron
            case 8: cindex = 2; break; // PPro
            case 9: cindex = 3; break; // AMD K7
            //no default
          }
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
          #ifdef MMX_BITSLICER
          if ((detected_type & 0x100) != 0) /* have mmx */
            cindex = 2; /* mmx bitslicer */
          else 
          #endif
          {
            switch ( detected_type & 0xff )
            {
              case 0: cindex = 0; break; // P5             == standard Bryd
              case 1: cindex = 0; break; // 386/486        == standard Bryd
              case 2: cindex = 1; break; // PII/PIII       == movzx Bryd
              case 3: cindex = 1; break; // Cx6x86         == movzx Bryd
              case 4: cindex = 0; break; // K5             == standard Bryd
              case 5: cindex = 1; break; // K6             == movzx Bryd
              case 6: cindex = 0; break; // Cx486          == movzx Bryd
              case 7: cindex = 1; break; // orig Celeron   == movzx Bryd
              case 8: cindex = 1; break; // PPro           == movzx Bryd
              case 9: cindex = 1; break; // AMD K7         == movzx Bryd
              //no default
            }
          }
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
          #if defined(MMX_CSC)
          if ((detected_type & 0x100) != 0) /* have mmx */
            cindex = 1; /* == 6bit - called - MMX */
          else
          #endif
          {
            // this is only valid for nasm'd cores or GCC 2.95 and up
            switch ( detected_type & 0xff )
            {
              case 0: cindex = 3; break; // P5             == 1key - called
              case 1: cindex = 3; break; // 386/486        == 1key - called
              case 2: cindex = 2; break; // PII/PIII       == 1key - inline
              case 3: cindex = 3; break; // Cx6x86         == 1key - called
              case 4: cindex = 2; break; // K5             == 1key - inline
              case 5: cindex = 0; break; // K6/K6-2/K6-3   == 6bit - inline
              case 6: cindex = 3; break; // Cyrix 486      == 1key - called
              case 7: cindex = 3; break; // orig Celeron   == 1key - called
              case 8: cindex = 3; break; // PPro           == 1key - called
              case 9: cindex = 0; break; // AMD K7         == 6bit - inline
              //no default
            }
          }
          selcorestatics.corenum[CSC] = cindex;
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
      if (detected_type == 0x3    || /* ARM 3 */ 
          detected_type == 0x600  || /* ARM 600 */
          detected_type == 0x610  || /* ARM 610 */
          detected_type == 0x700  || /* ARM 700 */
          detected_type == 0x710  || /* ARM 710 */
          detected_type == 0x7500 || /* ARM 7500 */ 
          detected_type == 0x7500FE) /* ARM 7500FE */
        cindex = 0;
      else if (detected_type == 0x200) /* ARM 2, 250 */
        cindex = 1;
      else if (detected_type == 0x810 || /* ARM 810 */
	       detected_type == 0xA10)    /* StrongARM 110 */
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
          detected_type == 0x200)    /* ARM 2, 250 */
        cindex = 1;
      else /* "ARM 3, 610, 700, 7500, 7500FE" or  "ARM 710" */
        cindex = 0;
      selcorestatics.corenum[DES] = cindex;  
    }  
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
      int whichcrunch;
      int saidmsg = 0, fastestcrunch = -1;
      unsigned long fasttime = 0;
      const u32 benchsize = 100000;
      Problem *problem = new Problem();

      for (whichcrunch = 0; whichcrunch < corecount; whichcrunch++)
      {
        ContestWork contestwork;
        unsigned long elapsed;
        selcorestatics.corenum[contestid] = whichcrunch;
        memset( (void *)&contestwork, 0, sizeof(contestwork));
        contestwork.crypto.iterations.lo = benchsize;
        if (problem->LoadState( &contestwork, contestid, 
                                benchsize, 0, 0, 0, 0 ) == 0)
        {
          if (!saidmsg)
          {
            LogScreen("%s: Running micro-bench to select fastest core...\n", 
                      contname);
            saidmsg = 1;
          }                                
          problem->Run();
   
          elapsed = (((unsigned long)problem->runtime_sec) * 1000000UL)+
                    (((unsigned long)problem->runtime_usec));
          //printf("%s Core %d: %lu usec\n", contname,whichcrunch,elapsed);
    
          if (fastestcrunch < 0 || elapsed < fasttime)
          {
            fastestcrunch = whichcrunch; 
            fasttime = elapsed;
          }
        }
      }
      delete problem;

      if (fastestcrunch < 0) /* all failed */
        fastestcrunch = 0; /* don't bench again */
      selcorestatics.corenum[contestid] = fastestcrunch;
    }
  }

  if (selcorestatics.corenum[contestid] >= 0 && !corename_printed)
  { 
    Log("%s: using core #%d (%s).\n", contname, 
         selcorestatics.corenum[contestid], 
         selcoreGetDisplayName(contestid, selcorestatics.corenum[contestid]) );
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
  #if (CLIENT_OS == OS_RISCOS) && defined(HAVE_X86_CARD_SUPPORT)
  extern "C" u32 rc5_unit_func_x86( RC5UnitWork * , u32 );
  #endif
#elif (CLIENT_CPU == CPU_S390)
  // rc5/ansi/rc5ansi_2-rg.cpp
  extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
#elif (CLIENT_CPU == CPU_PA_RISC)
  // rc5/parisc/parisc.cpp encapulates parisc.s, 2 pipelines
  extern "C" u32 rc5_parisc_unit_func( RC5UnitWork *, u32 );
#elif (CLIENT_CPU == CPU_88K) //OS_DGUX
  // rc5/ansi/rc5ansi_2-rg.cpp
  extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
#elif (CLIENT_CPU == CPU_MIPS)
  #if (CLIENT_OS == OS_ULTRIX) || (CLIENT_OS == OS_IRIX)
    // rc5/ansi/rc5ansi_2-rg.cpp
    extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
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
    // rc5/ansi/2-rg.cpp
    extern "C" u32 rc5_ansi_2_rg_unit_func( RC5UnitWork *, u32 iterations );
  #endif
#elif (CLIENT_CPU == CPU_68K)
  #if (CLIENT_OS == OS_MACOS)
    // rc5/ansi/rc5ansi_2-rg.cpp
    extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
  #elif(CLIENT_OS == OS_AMIGAOS)
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
    // rc5/ansi/rc5ansi_2-rg.cpp
    extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
  #endif
  #if (CLIENT_CPU == CPU_POWERPC) || defined(_AIXALL)
    #if (CLIENT_OS == OS_WIN32) //NT has poor PPC assembly
      // rc5/ansi/rc5ansi_2-rg.cpp
      extern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 iterations );
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

#if defined(HAVE_CSC_CORES)
  extern "C" s32 csc_unit_func_1k  ( RC5UnitWork *, u32 *iterations, void *membuff );
  #if (CLIENT_CPU != CPU_ARM) // ARM only has one CSC core
  extern "C" s32 csc_unit_func_1k_i( RC5UnitWork *, u32 *iterations, void *membuff );
  extern "C" s32 csc_unit_func_6b  ( RC5UnitWork *, u32 *iterations, void *membuff );
  extern "C" s32 csc_unit_func_6b_i( RC5UnitWork *, u32 *iterations, void *membuff );
  #endif
  #if (CLIENT_CPU == CPU_X86) && defined(MMX_CSC)
  extern "C" s32 csc_unit_func_6b_mmx ( RC5UnitWork *, u32 *iterations, void *membuff );
  #endif
#endif

/* ------------------------------------------------------------- */

int selcoreSelectCore( unsigned int contestid, unsigned int threadindex,
                       int *client_cpuP, Problem *problem )
{                               
  #if (CLIENT_CPU == CPU_X86) //most projects have an mmx core
  static int ismmx = -1; 
  if (ismmx == -1) 
  { 
    long det = GetProcessorType(1 /* quietly */);
    ismmx = (det >= 0) ? (det & 0x100) : 0;
  }    
  #endif
  int use_generic_proto = 0; /* if rc5/des unit_func proto is generic */
  unit_func_union unit_func; /* declared in problem.h */
  int cruncher_is_asynchronous = 0; /* on a co-processor or similar */
  int pipeline_count = 2; /* most cases */
  int client_cpu = CLIENT_CPU; /* usual case */
  int coresel = selcoreGetSelectedCoreForContest( contestid );
  if (coresel < 0)
    return -1;

  /* -------------------------------------------------------------- */

  if (contestid == RC5) /* avoid switch */
  {
    #if (CLIENT_CPU == CPU_ARM)
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
    #elif (CLIENT_CPU == CPU_PA_RISC)
    {
      // /rc5/parisc/parisc.cpp encapulates parisc.s, 2 pipelines
      //xtern "C" u32 rc5_parisc_unit_func( RC5UnitWork *, u32 );
      unit_func.rc5 = rc5_parisc_unit_func;
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
      #if (CLIENT_OS == OS_ULTRIX) || (CLIENT_OS == OS_IRIX)
      {
        // rc5/ansi/rc5ansi_2-rg.cpp
        //xtern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32 );
        unit_func.rc5 = rc5_unit_func_ansi_2_rg;
        pipeline_count = 2;
        coresel = 0;
      }
      #elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_SINIX)
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
      #if (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS)
      {
        //rc5/ultra/rc5-ultra-crunch.cpp
        //xtern "C" u32 rc5_unit_func_ultrasparc_crunch( RC5UnitWork * , u32 );
        unit_func.rc5 = rc5_unit_func_ultrasparc_crunch;
        pipeline_count = 2;
        coresel = 0;
      }
      #else
      {
        // rc5/ansi/2-rg.cpp
        //xtern "C" u32 rc5_ansi_2_rg_unit_func( RC5UnitWork *, u32 iterations );
        unit_func.rc5 = rc5_ansi_2_rg_unit_func;
        pipeline_count = 2;
        coresel = 0;
      }
      #endif
    }
    #elif (CLIENT_CPU == CPU_68K)
    {
      #if (CLIENT_OS == OS_MACOS) /* Take 68K Macs back to ANSI core *for now*
                                     due to the fact that the old assembly core 
                                     won't compile under Codewarrior */
      {
        // rc5/ansi/rc5ansi_2-rg.cpp
        //xtern u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32);
        unit_func.rc5 = rc5_unit_func_ansi_2_rg;
        pipeline_count = 2;
      }

      #elif (CLIENT_OS == OS_AMIGAOS)
      {
        // rc5/68k/rc5_68k_crunch.c around rc5/68k/rc5-0x0_0y0-jg.s
        //xtern "C" u32 rc5_unit_func_000_030( RC5UnitWork *, u32 );
        //xtern "C" u32 rc5_unit_func_040_060( RC5UnitWork *, u32 );
        if (coresel == 1 )
        {
          pipeline_count = 2;
          unit_func.rc5 = rc5_unit_func_040_060;
          coresel = 1;
        }
        else
        {
          pipeline_count = 2;
          unit_func.rc5 = rc5_unit_func_000_030;
          coresel = 0;
        }
      }
      #elif defined(__GCC__) || defined(__GNUC__) /* hpux, next, linux, sun3 */
      {
        // rc5/68k/rc5_68k_gcc_crunch.c around rc5/68k/crunch.68k.gcc.s
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
      }
      #else //((CLIENT_CPU == CPU_POWERPC) || defined(_AIXALL))
      { 
        //#if (CLIENT_OS == OS_WIN32) //NT has poor PPC assembly
        //  rc5/ansi/rc5ansi_2-rg.cpp
        //  xtern "C" u32 rc5_unit_func_ansi_2_rg( RC5UnitWork *, u32  );
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

        client_cpu = CPU_POWERPC;
        #if defined(_AIXALL) //ie POWER/POWERPC hybrid client
        if ((GetProcessorType(1) & (1L<<24)) != 0) //ARCH_IS_POWER
        {
          client_cpu = CPU_POWER;
          unit_func.rc5 = rc5_unit_func_ansi_2_rg; //rc5/ansi/rc5ansi_2-rg.cpp
          pipeline_count = 2;
          coresel = 0; //core #0 is "RG AIXALL" on POWER, and allitnil on PPC
          gotcore = 1;
        }
        #endif
        if (!gotcore && coresel == 0)     // G1 (PPC 601)
        {  
          unit_func.rc5 = rc5_unit_func_allitnil_compat;
          pipeline_count = 1;
          gotcore = 1;
        }  
        else if (!gotcore && coresel == 2) // G4 (PPC 7500)
        {
          unit_func.rc5 = rc5_unit_func_vec_compat;
          pipeline_count = 1;
          gotcore = 1;
        }
        if (!gotcore)                     // the rest (G2/G3)
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
      if (coresel < 0 || coresel > 5)
        coresel = 0;
      pipeline_count = 2; /* most cases */
      if (coresel == 1)   // Intel 386/486
      {
        unit_func.rc5 = rc5_unit_func_486;
        #if defined(SMC) 
        if (threadindex == 0) /* first thread or benchmark/test */
          unit_func.rc5 =  rc5_unit_func_486_smc;
        #endif
      }
      else if (coresel == 2) // Ppro/PII
        unit_func.rc5 = rc5_unit_func_p6;
      else if (coresel == 3) // 6x86(mx)
        unit_func.rc5 = rc5_unit_func_6x86;
      else if (coresel == 4) // K5
        unit_func.rc5 = rc5_unit_func_k5;
      else if (coresel == 5) // K6/K6-2/K7
      {
        unit_func.rc5 = rc5_unit_func_k6;
        #if defined(MMX_RC5_AMD)
        if (ismmx)
        { 
          unit_func.rc5 = rc5_unit_func_k6_mmx;
          pipeline_count = 4;
        }
        #endif
      }
      else // Pentium (0/6) + others
      {
        unit_func.rc5 = rc5_unit_func_p5;
        #if defined(MMX_RC5)
        if (ismmx)
        { 
          unit_func.rc5 = rc5_unit_func_p5_mmx;
          pipeline_count = 4; // RC5 MMX core is 4 pipelines
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
          pipeline_count = 2;
          unit_func.rc5 = rc5_alpha_osf_ev5;
        }
        else // EV3_CPU, EV4_CPU, LCA4_CPU, EV45_CPU and default
        {
          pipeline_count = 2;
          #if defined(DEC_UNIX_CPU_SELECT)
          unit_func.rc5 = rc5_alpha_osf_ev4; 
          #else
          unit_func.rc5 = rc5_alpha_osf_ev5; 
          #endif
          coresel = 0;
        }
      }
      #elif (CLIENT_OS == OS_WIN32) /* little-endian asm */
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
      #if (CLIENT_OS == OS_DEC_UNIX) && defined(DEC_UNIX_CPU_SELECT)
      {
        //xtern u32 des_alpha_osf_ev4( RC5UnitWork * , u32 *, char * );
        //xtern u32 des_alpha_osf_ev5( RC5UnitWork * , u32 *, char * );
        if (coresel == 1) /* EV5, EV56, PCA56, EV6 */
          unit_func.des = des_alpha_osf_ev5;
        else // EV3_CPU, EV4_CPU, LCA4_CPU, EV45_CPU and default
          unit_func.des = des_alpha_osf_ev4;
      }
      #else
      {
        //des/alpha/des-slice-dworz.cpp
        //xtern u32 des_unit_func_slice_dworz( RC5UnitWork * , u32 *, char * );
        unit_func.des = des_unit_func_slice_dworz;
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
        if (ismmx) 
          slicit = des_unit_func_mmx;
      }
      #endif  
      if (slicit && coresel > 1) /* not standard bryd and not ppro bryd */
      {                /* coresel=2 is valid only if we have a slice core */
        coresel = 2;
        unit_func.des = slicit;
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
          else                    /* fifth...nth thread */
            unit_func.des = slicit;
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
          if (threadindex == 1)  /* second thread */
            unit_func.des = p2des_unit_func_p5;
          else if (threadindex == 2) /* third thread */
            unit_func.des = p1des_unit_func_pro;
          else if (threadindex == 3) /* fourth thread */
            unit_func.des = p2des_unit_func_pro;
          else                    /* fifth...nth thread */
            unit_func.des = slicit;
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
    coresel = 0;
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
               #if (CLIENT_CPU == CPU_X86) && defined(MMX_CSC)
               if (ismmx) //6b-non-mmx isn't used (by default) on x86
                 unit_func.gen = csc_unit_func_6b_mmx;
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

  if (coresel >= 0 && coresel < ((int)__corecount_for_contest( contestid )))
  {
    if (client_cpuP)
      *client_cpuP = client_cpu;
    if (problem)
    {
      problem->client_cpu = client_cpu;
      problem->pipeline_count = pipeline_count;
      problem->use_generic_proto = use_generic_proto;
      problem->cruncher_is_asynchronous = cruncher_is_asynchronous;
      memcpy( (void *)&(problem->unit_func), &unit_func, sizeof(unit_func));
    }
    return coresel;
  }

  threadindex = threadindex; /* possibly unused. shaddup compiler */
  return -1; /* core selection failed */
}

/* ------------------------------------------------------------- */
