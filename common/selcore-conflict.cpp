/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * program (pro'-gram) [vi]: To engage in a pastime similar to banging
 * one's head against a wall but with fewer opportunities for reward.
 * 
 */
const char *selcore_cpp(void) {
return "@(#)$Id: selcore-conflict.cpp,v 1.47.2.6 1999/10/08 01:41:31 cyp Exp $"; }


#include "cputypes.h"
#include "client.h"    // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "problem.h"   // problem class
#include "cpucheck.h"  // cpu selection, GetTimesliceBaseline()
#include "logstuff.h"  // Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "clicdata.h"  // GetContestNameFromID()
#include "selcore.h"   // keep prototypes in sync

/* ------------------------------------------------------------------------ */

//************************************
//"--------- max width = 34 ---------" (35 including terminating '\0')
//************************************

#if (CLIENT_CPU == CPU_X86)
static const char *rc5anddes_table[]=
{
  "Pentium, Am486, Cx486/5x86/MediaGX",
  "80386 & 80486",
  "Pentium Pro/II/III",
  "Cyrix 6x86/6x86MX/M2",
  "AMD K5",
  "AMD K6/K7"
  //core 6 is "reserved" (was Pentium MMX)
};
#elif ((CLIENT_CPU == CPU_ALPHA) && ((CLIENT_OS == OS_DEC_UNIX) || \
     (CLIENT_OS == OS_OPENBSD) || (CLIENT_OS == OS_LINUX)))
static const char *rc5anddes_table[]=
{
  "unknown",
  "EV3",
  "EV4 (21064)",
  "unknown",
  "LCA4 (21066/21068)",
  "EV5 (21164)",
  "EV4.5 (21064)",
  "EV5.6 (21164A)",
  "EV6 (21264)",
  "EV5.6 (21164PC)"
};
#elif (CLIENT_CPU == CPU_ARM)
static const char *rc5anddes_table[]=
{
  "ARM 3, 610, 700, 7500, 7500FE",
  "ARM 810, StrongARM 110",
  "ARM 2, 250",
  "ARM 710"
};
#elif (CLIENT_OS == OS_AIX) || ((CLIENT_CPU == CPU_POWERPC) && \
      ((CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_MACOS)))
static const char *rc5anddes_table[]=
{
  #if (CLIENT_OS == OS_AIX)
  "POWER CPU",
  #endif
  "PowerPC 601",
  "PowerPC 603/604/750"
};
#elif (CLIENT_CPU == CPU_68K)
static const char *rc5anddes_table[]=
{
  "Motorola 68000", "Motorola 68010", "Motorola 68020", "Motorola 68030",
  "Motorola 68040", "Motorola 68060"
};
#else
  #define NO_CPUTYPE_TABLE
#endif

static const char *csc_table[]=
{
  "6 bit - inline", 
  "6 bit - called", 
  "1 key - inline", 
  "1 key - called"
};

/* ====================================================================== */

static const char **__corenames_for_contest( unsigned int cont_i )
{
  static const char *ansi_core[] = { "Generic ANSI core" };
  if (cont_i == RC5 || cont_i == DES)
  {
    #if defined(NO_CPUTYPE_TABLE)
    return &ansi_core[0];
    #else
    return &rc5anddes_table[0];
    #endif
  }
  else if (cont_i == OGR)
    return &ansi_core[0];
  else if (cont_i == CSC)
    return &csc_table[0];
  return ((const char **)0);
}

static unsigned int __corecount_for_contest( unsigned int cont_i )
{
  if (cont_i == RC5 || cont_i == DES)
  {
    #if defined(NO_CPUTYPE_TABLE)
    return 1;
    #else 
    return (sizeof(rc5anddes_table)/sizeof(rc5anddes_table[0]));
    #endif
  }
  else if (cont_i == OGR)
    return 1;
  else if (cont_i == CSC)
    return (sizeof(csc_table)/sizeof(csc_table[0]));
  return 0;
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

int selcoreValidateCoreIndex( unsigned int cont_i, int index )
{
  /* To enable future expansion, we return -1 if there is only one core */
  if (index >= 0 && index < ((int)__corecount_for_contest( cont_i )))
    return index;
  return -1;
}

const char *selcoreGetDisplayName( unsigned int cont_i, int index )
{
  if (index >= 0 && index < ((int)__corecount_for_contest( cont_i )))
  {
     const char **names = __corenames_for_contest( cont_i );
     return names[index];
  }
  return "";
}

/* ---------------------------------------------------------------------- */

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
    int verbosedetect = 0;
    for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
    {
      int gotchange = 0, index = 0;
      if (__corecount_for_contest( cont_i ) > 1)
        index = selcoreValidateCoreIndex( cont_i, coretypes[cont_i] );
      gotchange = (!initialized ||
                   index != selcorestatics.user_cputype[cont_i]);
      if (gotchange)
        selcorestatics.corenum[cont_i] = -1;
      if (!verbosedetect)
        verbosedetect = (index == -1);
      selcorestatics.user_cputype[cont_i] = index;
    }
    initialized = 1;
  }
  if (initialized > 0)
    return 0;
  return -1;
}  

/* ---------------------------------------------------------------------- */

/* this is called from Problem::LoadState() */
int selcoreGetSelectedCoreForContest( unsigned int contestid )
{
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

    
  #if (CLIENT_CPU == CPU_68K)
  if (contestid == RC5 || contestid == DES) /* old style */
  {
    const char *corename = NULL;
    selcorestatics.corenum[DES] = 0;  /* only one DES core */
    selcorestatics.corenum[RC5] = selcorestatics.user_cputype[RC5];
    if (selcorestatics.corenum[RC5] < 0)
      selcorestatics.corenum[RC5] = detected_type;
    if (selcorestatics.corenum[RC5] < 0)
      selcorestatics.corenum[RC5] = 0;
    if (selcorestatics.corenum[RC5] == 4 || selcorestatics.corenum[RC5] == 5 ) 
      corename = "040/060";  // there is no 68050, so type5=060
    else //if (cputype == 0 || cputype == 1 || cputype == 2 || cputype == 3)
      corename = "000/010/020/030";
    LogScreen( "Selected code optimized for the Motorola 68%s.\n", corename );
  }
  #elif (CLIENT_CPU == CPU_POWERPC)
  if (contestid == RC5 || contestid == DES) /* old style */
  {
    selcorestatics.corenum[DES] = 0; /* only one DES core */
    #if ((CLIENT_OS == OS_BEOS) || (CLIENT_OS == OS_AMIGAOS))
      // Be OS isn't supported on 601 machines
      // There is no 601 PPC board for the Amiga
      selcorestatics.corenum[RC5] = 1; //"PowerPC 603/604/750"
    #elif (CLIENT_OS == OS_WIN32)
      //actually not supported, but just in case
      selcorestatics.corenum[RC5] = 1;
    #endif
  }
  #elif (CLIENT_CPU == CPU_X86)
  if (contestid == RC5)
  {
    const char *selmsg = NULL;
    selcorestatics.corenum[RC5] = selcorestatics.user_cputype[RC5];
    if (selcorestatics.corenum[RC5] < 0)
      selcorestatics.corenum[RC5] = (int)(detected_type & 0xff);
    
    if (selcorestatics.corenum[RC5] == 1) // Intel 386/486
    {
      #if defined(SMC) /* actually only for the first thread */
      selmsg = "80386 & 80486 self modifying";
      #endif
    }
    else if (selcorestatics.corenum[RC5] <= 0 || 
             selcorestatics.corenum[RC5] >= 6)
    {         
      selcorestatics.corenum[RC5] = 0;
      #if defined(MMX_RC5)
      if (detected_type == 0x106) /* Pentium MMX only! */
        selmsg = "Pentium MMX";
      #endif
    }
    if (!selmsg)
      selmsg = selcoreGetDisplayName( RC5, selcorestatics.corenum[RC5] );
    if (selmsg)
      LogScreen( "%s: selecting %s code.\n", contname, selmsg );
  }     
  else if (contestid == DES)
  {  
    const char *selmsg = NULL;
    int selppro_des = 0;
    selcorestatics.corenum[DES] = selcorestatics.user_cputype[DES];
    if (selcorestatics.corenum[DES] < 0)
      selcorestatics.corenum[DES] = detected_type & 0xff;
    if (selcorestatics.corenum[DES] == 1) // 386/486
      selppro_des = 0;
    else if (selcorestatics.corenum[DES] == 2) // Ppro/PII
      selppro_des = 1;
    else if (selcorestatics.corenum[DES] == 3) // 6x86(mx)
      selppro_des = 1;
    else if (selcorestatics.corenum[DES] == 4) // K5
      selppro_des = 0;
    else if (selcorestatics.corenum[DES] == 5) // K6/K6-2
      selppro_des = 1;
    else // Pentium (0/6) + others
        selcorestatics.corenum[DES] = 0;
    #if defined(MMX_BITSLICER)
    if ((detected_type & 0x100) != 0)//use the MMX DES core ?
      selmsg = "MMX bitslice";
    #endif
    if (!selmsg)
      selmsg = ((selppro_des)?("PentiumPro optimized BrydDES"):("BrydDES"));
    if (selmsg)
      LogScreen( "%s: selecting %s core.\n", contname, selmsg );
  }
  else if (contestid == CSC)
  {
    selcorestatics.corenum[CSC] = selcorestatics.user_cputype[CSC];
    int user_selected = 1;
#if 0 /* we don't support hardware detection yet, but this is how we would */
    if (selcorestatics.corenum[CSC] < 0)
    {
      int cpu2core = detected_type & 0xff;
      if (cpu2core == 2) // Ppro/PII/PIII
        selcorestatics.corenum[CSC] = 1; //6bit - called
      /*
      else if (cpu2core == ....
        ...
      */
      user_selected = 0;
    }
#endif
    if (selcorestatics.corenum[CSC] >= 0)
    {
      LogScreen( "%s: %s core #%d (%s)\n", contname, 
                 ((user_selected)?("using"):("auto-selected")),
  	             selcorestatics.corenum[CSC],
                 selcoreGetDisplayName( CSC, selcorestatics.corenum[CSC] ) );
    }
  }
  #elif (CLIENT_CPU == CPU_ARM)
  if (contestid == RC5 || contestid == DES)
  {
    selcorestatics.corenum[contestid] = selcorestatics.user_cputype[contestid];
    if (selcorestatics.corenum[contestid] < 0)
    {
      if (detected_type >= 0)
        selcorestatics.corenum[contestid] = (int)detected_type;
    }
    if (selcorestatics.corenum[contestid] >= 0)
    {
      LogScreen( "%s: selecting %s optimized code.\n", contname, 
             selcoreGetDisplayName( RC5, selcorestatics.corenum[RC5] ) );
    }
    /* otherwise fall into bench */
  }
  #endif


  
  if (selcorestatics.corenum[contestid] < 0) /* ok, bench it then */
  {
    int corecount = (int)__corecount_for_contest(contestid);
    selcorestatics.corenum[contestid] = 0;
    if (corecount > 0)
    {
      int whichcrunch;
      int fastestcrunch = -1;
      unsigned long fasttime = 0;
      Problem *problem = new Problem();
      const u32 benchsize = 100000;

      LogScreen("%s: Manually selecting fastest core...\n", contname);
      for (whichcrunch = 0; whichcrunch < corecount; whichcrunch++)
      {
        ContestWork contestwork;
        unsigned long elapsed;
        selcorestatics.corenum[contestid] = whichcrunch;
        memset( (void *)&contestwork, 0, sizeof(contestwork));
        contestwork.crypto.iterations.lo = benchsize;
        problem->LoadState( &contestwork, contestid, benchsize, whichcrunch );
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
      selcorestatics.corenum[contestid] = fastestcrunch;
      LogScreen("%s: selected core #%d (%s).\n", contname, fastestcrunch, 
                     selcoreGetDisplayName( contestid, fastestcrunch ) );
    }
  }
  
  return selcorestatics.corenum[contestid];
}

/* ---------------------------------------------------------------------- */

