/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * PORTER NOTE: when adding support for a new processor family, add each
 * major processor type individually - *even_if_one_core_covers_more_than_
 * one_processor*. This is to avoid having obsolete cputype entries
 * in inis when more cores become available.                   - cyp
 * ----------------------------------------------------------------------
 */
const char *selcore_cpp(void) {
return "@(#)$Id: selcore-conflict.cpp,v 1.47.2.4 1999/09/19 16:04:37 cyp Exp $"; }


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
static const char *cputypetable[]=
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
static const char *cputypetable[]=
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
static const char *cputypetable[]=
{
  "ARM 3, 610, 700, 7500, 7500FE",
  "ARM 810, StrongARM 110",
  "ARM 2, 250",
  "ARM 710"
};
#elif (CLIENT_OS == OS_AIX) || ((CLIENT_CPU == CPU_POWERPC) && \
      ((CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_MACOS)))
static const char *cputypetable[]=
{
  #if (CLIENT_OS == OS_AIX)
  "POWER CPU",
  #endif
  "PowerPC 601",
  "PowerPC 603/604/750"
};
#elif (CLIENT_CPU == CPU_68K)
static const char *cputypetable[]=
{
  "Motorola 68000", "Motorola 68010", "Motorola 68020", "Motorola 68030",
  "Motorola 68040", "Motorola 68060"
};
#else
  #define NO_CPUTYPE_TABLE
#endif

/* ----------------------------------------------------------------------
 * PORTER NOTE: when adding support for a new processor family, add each
 * major processor type individually - *even_if_one_core_covers_more_than_
 * one_processor*. This is to avoid having obsolete cputype entries
 * in inis when more cores become available.                   - cyp
 * ----------------------------------------------------------------------
*/ 

/* returns name for cputype (from .ini), 0... or "" if no such cputype */
const char *selcoreUserGetCPUNameFromCPUType( int user_cputype )
{
  user_cputype = user_cputype; /* shaddup compiler */
  #if !(defined(NO_CPUTYPE_TABLE))
  if ( user_cputype >= 0 &&
       user_cputype<((int)(sizeof(cputypetable)/sizeof(cputypetable[0]))) )
    return cputypetable[user_cputype];
  #endif
  return "";  
}

/* ---------------------------------------------------------------------- */

static struct
{
  int user_cputype;   /* what the user has in the ini */
  long detected_type; /* what the hardware found */
  int corenum[CONTEST_COUNT]; /* what we map it to */
} selcorestatics = { -123, -123, {-1} };

/* ---------------------------------------------------------------------- */

int selcoreInitialize( int user_cputype ) /* Client::Main() calls this */
{
  int i, corecount;

  #ifdef NO_CPUTYPE_TABLE
  user_cputype = 0;
  corecount = 1; /* only one core per contest */
  #else
  corecount = (int)(sizeof(cputypetable)/sizeof(cputypetable[0]));
  if (user_cputype<0 || user_cputype>=corecount)
    user_cputype = -1;
  #endif

  if (selcorestatics.user_cputype == user_cputype) //no change
    return 0;                      //(cputype can change when restarted)

  for (i=0; i < CONTEST_COUNT; i++)
    selcorestatics.corenum[i] = -1;
  
  if (selcorestatics.detected_type == -123 || (user_cputype < 0))
  {
    //returns -1 if unable to detect
    selcorestatics.detected_type = GetProcessorType( (user_cputype >= 0) );
    if (selcorestatics.detected_type < 0)
      selcorestatics.detected_type = -1;
    else if (((int)(selcorestatics.detected_type & 0xff)) >= corecount)
      selcorestatics.detected_type = -1;
    else if (user_cputype < 0)
      user_cputype = (int)(selcorestatics.detected_type & 0xff);
  }

  selcorestatics.user_cputype = user_cputype;
  
  return 0;
}    
  
/* ---------------------------------------------------------------------- */

/* this is called from Problem::LoadState() */
int selcoreGetSelectedCoreForContest( unsigned int contestid )
{
  const char *contname = CliGetContestNameFromID(contestid);
  if (!contname) /* no such contest */
    return -1;
  
  if (selcorestatics.user_cputype == -123)   /* ACK! selcoreInitialize() */
    return -1;                               /* hasn't been called */

  if (contestid == OGR)                      /* OGR only has one core */
    return 0;

  if (selcorestatics.corenum[contestid] >= 0) /* already selected one? */
    return selcorestatics.corenum[contestid];
    
  #if (CLIENT_CPU == CPU_68K)
  if (selcorestatics.corenum[RC5] < 0) /* haven't done that yet */
  {
    const char *corename = NULL;
    selcorestatics.corenum[DES] = 0;  /* only one DES core */
    selcorestatics.corenum[RC5] = selcorestatics.user_cputype;
    if (selcorestatics.corenum[RC5] < 0)
      selcorestatics.corenum[RC5] = 0;
    if (selcorestatics.corenum[RC5] == 4 || selcorestatics.corenum[RC5] == 5 ) 
      corename = "040/060";  // there is no 68050, so type5=060
    else //if (cputype == 0 || cputype == 1 || cputype == 2 || cputype == 3)
      corename = "000/010/020/030";
    LogScreen( "Selected code optimized for the Motorola 68%s.\n", corename );
  }
  #elif (CLIENT_CPU == CPU_POWERPC)
  {
    selcorestatics.corenum[DES] = 0; /* only one DES core */
    #if ((CLIENT_OS == OS_BEOS) || (CLIENT_OS == OS_AMIGAOS))
      // Be OS isn't supported on 601 machines
      // There is no 601 PPC board for the Amiga
      selcorestatics.corenum[RC5] = 1; //"PowerPC 603/604/750"
    #elif (CLIENT_OS == OS_WIN32)
      //actually not supported, but just in case
      selcorestatics.corenum[DES] = 1;
    #endif
  }
  #elif (CLIENT_CPU == CPU_X86)
  if (selcorestatics.corenum[contestid] < 0) /* haven't done this yet */
  {
    const char *selmsg = NULL;
    if (contestid == RC5)
    {
      selcorestatics.corenum[contestid] = selcorestatics.user_cputype;
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
        if (selcorestatics.detected_type == 0x106) /* Pentium MMX only! */
          selmsg = "Pentium MMX";
        #endif
      }
      if (!selmsg)
        selmsg = selcoreUserGetCPUNameFromCPUType(selcorestatics.corenum[RC5]);
    }     
    else if (contestid == DES)
    {  
      int selppro_des = 0;
      selcorestatics.corenum[contestid] = selcorestatics.user_cputype;
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
      if ((selcorestatics.detected_type & 0x100) != 0)//use the MMX DES core ?
        selmsg = "MMX bitslice";
      #endif
      if (!selmsg)
        selmsg = ((selppro_des)?("PentiumPro optimized BrydDES"):("BrydDES"));
    }
    if (selmsg)
      LogScreen( "%s: selecting %s core.\n", contname, selmsg );
  }
  #elif (CLIENT_CPU == CPU_ARM)
  if (contestid == RC5 || contestid == DES)
  {
    if (selcorestatics.user_cputype != -1) /* otherwise fall into bench */
    {
      selcorestatics.corenum[contestid] = selcorestatics.user_cputype;
      LogScreen( "%s: selecting %s core.\n", contname, 
        selcoreUserGetCPUNameFromCPUType(selcorestatics.corenum[contestid]);
    }
  }
  #endif
  
  if (selcorestatics.corenum[contestid] < 0) /* ok, bench it then */
  {
    int corecount = 0; /* number of cores available */
    if (contestid == CSC)
      corecount = 4;
    #ifndef NO_CPUTYPE_TABLE
    else
      corecount = (int)(sizeof(cputypetable)/sizeof(cputypetable[0]));
    #endif
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
      LogScreen("%s: selected core #%d.\n", contname, fastestcrunch );
    }
  }
  
  return selcorestatics.corenum[contestid];
}

/* ---------------------------------------------------------------------- */

#if 0
const char *GetCoreNameFromCoreType( unsigned int user_cputype ) /* COMPAT */
{ return selcoreUserGetCPUNameFromCPUType( user_cputype ); }  
#endif

int Client::SelectCore(int quietly)  /* COMPAT */
{
  /* what this function will eventually do is copy the 
     per-contest cputypes into the static structure */
  quietly = quietly;
  return selcoreInitialize( this->cputype );
}

/* ---------------------------------------------------------------------- */

