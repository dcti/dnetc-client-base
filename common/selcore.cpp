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
return "@(#)$Id: selcore.cpp,v 1.47.2.2 1999/09/17 17:12:08 cyp Exp $"; }


#include "cputypes.h"
#include "client.h"    // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes
//#include "version.h"
#include "problem.h"   // problem class
#include "cpucheck.h"  // cpu selection, GetTimesliceBaseline()
#include "logstuff.h"  // Log()/LogScreen()/LogScreenPercent()/LogFlush()
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

/* ---------------------------------------------------------------------- */

const char *GetCoreNameFromCoreType( unsigned int coretype )
{
  #if (defined(NO_CPUTYPE_TABLE))
    if (coretype) return ""; //dummy to suppress unused variable warnings
  #else
    if (coretype<(sizeof(cputypetable)/sizeof(cputypetable[0])))
      return cputypetable[coretype];
  #endif
  return "";
}

/* ---------------------------------------------------------------------- */

/* *REPEATED WARNING* - cputype *must* be valid on return from SelectCore() */

int Client::SelectCore(int quietly)
{
  static s32 last_cputype = -123;
  static int detectedtype = -123;
  unsigned int corecount = 0; /* number of cores available */

  numcpu = ValidateProcessorCount( numcpu, quietly ); //in cpucheck.cpp

  if (cputype == last_cputype) //no change, so don't bother reselecting
    return 0;                  //(cputype can change when restarted)

  #ifdef NO_CPUTYPE_TABLE
  cputype = 0;
  #else
  corecount = (sizeof(cputypetable)/sizeof(cputypetable[0]));
  #endif
  
  if (cputype<0 || cputype>=((int)corecount))
    cputype = -1;
  if (cputype == -1)
  {
    if (detectedtype == -123) 
      detectedtype = GetProcessorType(quietly);//returns -1 if unable to detect
    if (detectedtype < 0)
      detectedtype = -1;
    else if ((cputype = ((s32)(detectedtype & 0xff))) >=((s32)corecount))
    { 
      detectedtype = -1; 
      cputype = -1; 
    }
  }
    
#if (CLIENT_CPU == CPU_68K)
  if (cputype == -1)
    cputype = 0;
  if (!quietly)
  {
    const char *corename = NULL;
    if (cputype == 4 || cputype == 5 ) // there is no 68050, so type5=060
      corename = "040/060";
    else //if (cputype == 0 || cputype == 1 || cputype == 2 || cputype == 3)
      corename = "000/010/020/030";
    LogScreen( "Selected code optimized for the Motorola 68%s.\n", corename ); 
  }
#elif (CLIENT_CPU == CPU_X86)
  int selppro_des = 0;
  const char *selmsg_rc5 = NULL, *selmsg_des = NULL;
    
  if (detectedtype < 0) /* user provided a #, but we need to detect for mmx */
    detectedtype = GetProcessorType(1); /* but do it quietly */
  
  if (cputype == 6) /* Pentium MMX */
    cputype = 0;    /* but we need backwards compatability */

  if (cputype == 1) // Intel 386/486
  {
    #if defined(SMC) 
    {
      #if defined(CLIENT_SUPPORTS_SMP)
      if (numcpu < 2)
      #endif
        selmsg_rc5 = "80386 & 80486 self modifying";
    }
    #endif
  }
  else if (cputype == 2) // Ppro/PII
    selppro_des = 1;
  else if (cputype == 3) // 6x86(mx)
    selppro_des = 1;
  else if (cputype == 4) // K5
    ;
  else if (cputype == 5) // K6/K6-2
    selppro_des = 1;
  else // Pentium (0/6) + others
  {
    cputype = 0;
    #if defined(MMX_RC5)
    if (detectedtype == 0x106) /* Pentium MMX only! */
      selmsg_rc5 = "Pentium MMX";
    #endif
  }

  #if defined(MMX_BITSLICER)
  if ((detectedtype & 0x100) != 0)   // use the MMX DES core ?
    selmsg_des = "MMX bitslice";
  #endif

  if (!selmsg_des)
    selmsg_des = ((selppro_des)?("PentiumPro optimized BrydDES"):("BrydDES"));
  if (!selmsg_rc5)
    selmsg_rc5 = GetCoreNameFromCoreType(cputype);
    
  if (!quietly)
    LogScreen( "DES: selecting %s core.\n"
               "RC5: selecting %s core.\n", selmsg_des, selmsg_rc5 );
      
#elif (CLIENT_CPU == CPU_ARM)
  if (cputype == -1)
  {
    const u32 benchsize = 100000;
    unsigned long fasttime[2] = { 0, 0 };
    unsigned int contestid;
    int whichcrunch;
    int fastcoretest[2] = { -1, -1 };
  
    if (!quietly)
      LogScreen("Manually selecting fastest core...\n");
    for ( contestid = 0; contestid < CONTEST_COUNT; contestid++)
    {
      for ( whichcrunch = 0; whichcrunch < 3; whichcrunch++)
      {
        Problem *problem = new Problem();
        ContestWork contestwork;
	unsigned long elapsed;
        //!! This should probably be addressed for OGR, zero data is not valid input.
	memset( (void *)&contestwork, 0, sizeof(contestwork));
        contestwork.crypto.iterations.lo = benchsize;
        problem->LoadState( &contestwork , contestid, benchsize, whichcrunch );
        problem->Run();
    
        elapsed = (((unsigned long)problem->runtime_sec) * 1000000UL)+
	          (((unsigned long)problem->runtime_usec));
        //printf("%s Core %d: %lu usec\n", CliGetContestName(contestid),whichcrunch,elapsed);
    
        if (fastcoretest[contestid] < 0 || elapsed < fasttime[contestid])
        {
          fastcoretest[contestid] = whichcrunch; 
	  fasttime[contestid] = elapsed;
        }
	delete problem;
      }
    }
    cputype = (fastcoretest[0] + ((fastcoretest[1]&1)<<2));
    if (cputype == 6)
      cputype = 1;
    else if (cputype == 5)
      cputype = 2;
    else if (cputype == 2)
      cputype = 3;
    else
      cputype = 0;
    detectedtype = cputype;
  }
  if (!quietly)
    LogScreen("Selecting %s code.\n",
              GetCoreNameFromCoreType(cputype));
#elif (CLIENT_CPU == CPU_POWERPC)
  #if ((CLIENT_OS == OS_BEOS) || (CLIENT_OS == OS_AMIGAOS))
    // Be OS isn't supported on 601 machines
    // There is no 601 PPC board for the Amiga
    cputype = 1; //"PowerPC 603/604/750"
  #elif (CLIENT_OS == OS_WIN32)
    //actually not supported, but just in case
    cputype = 1;
  #endif
#endif

  if (cputype == -1)
  {
    unsigned long fasttime = 0;
    int whichcrunch;
    if (!quietly)
        LogScreen("Manually selecting fastest core...\n");
    for (whichcrunch = 0; whichcrunch < ((int)corecount); whichcrunch++ )
    {
      const u32 benchsize = 500000L;
      Problem *problem = new Problem();
      ContestWork contestwork;
      unsigned long elapsed;
      memset( (void *)&contestwork, 0, sizeof(contestwork));
      contestwork.crypto.iterations.lo = benchsize;
      problem->LoadState( &contestwork , 0 /* RC5 */, benchsize, whichcrunch );
      problem->Run();  //threadnum
      elapsed = (((unsigned long)problem->runtime_sec) * 1000000UL)+
                 (((unsigned long)problem->runtime_usec));
      if (cputype < 0 || elapsed < fasttime)
        {cputype = whichcrunch; fasttime = elapsed;}
      delete problem;
    }
    detectedtype = cputype;
    if (!quietly)
      LogScreen( "Selected %s code.\n", GetCoreNameFromCoreType(cputype) ); 
  }

  if (cputype == -1)
    cputype = 0;
  last_cputype = cputype;
  return 0;
}

/* ---------------------------------------------------------------------- */

