// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: selcore.cpp,v $
// Revision 1.6  1998/09/25 11:31:23  chrisb
// Added stuff to support 3 cores in the ARM clients.
//
// Revision 1.5  1998/09/23 22:02:42  blast
// Added multi-core support for all 68k platforms.
// AmigaOS now has autodetection (-1) or manual (0, 1) of cores.
// Other 68k porters will have to add autodetection if their OS can handle
// autodetection.
//
// Revision 1.4  1998/09/04 06:05:46  silby
// Made selcore more verbose on x86 so that people would not confused rc5 and des core selections.
//
// Revision 1.3  1998/09/01 22:32:22  remi
// Allow a P5-MMX to use the RC5 MMX core.
//
// Revision 1.2  1998/08/22 08:01:46  silby
// Rewrote x86 core selection.
//
// Revision 1.1  1998/08/21 23:34:54  cyruspatel
// Created from code in cliconfig.cpp. x86 SelectCore() works again - clients
// on P5s are back up to speed. Validation of cputype is now done in SelectCore.
// All CLIENT_CPU types should be setting cputype correctly now. Coders! please
// verify this! Calls to Problem::LoadState() expects cputype to be valid.
//
//

#if (!defined(lint) && defined(__showids__))
const char *selcore_cpp(void) {
return "@(#)$Id: selcore.cpp,v 1.6 1998/09/25 11:31:23 chrisb Exp $"; }
#endif

#include "cputypes.h"
#include "client.h"   // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h" // basic (even if port-specific) #includes
#include "version.h"
#include "problem.h"  // ___unit_func()
#include "cpucheck.h" // cpu selection, GetTimesliceBaseline()
#include "clirate.h"
#include "logstuff.h"  //Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "clirate.h" //for PPC CliGetKeyrateForProblemNoSave() in SelectCore
#include "selcore.h"  //keep prototypes in sync

// --------------------------------------------------------------------------

#if (CLIENT_CPU == CPU_X86)
static const char *cputypetable[]=
  {
  "Pentium Classic, Cyrix 486/5x86/MediaGX, AMD 486",
  "Intel 80386 & 80486",
  "Pentium Pro & Pentium II",
  "Cyrix 6x86/6x86MX/M2",
  "AMD K5",
  "AMD K6",
  "Pentium MMX"
  };
#elif (CLIENT_CPU == CPU_ARM)
static const char *cputypetable[]=
  {
  "ARM 3, 610, 700, 7500, 7500FE",
  "ARM 810, StrongARM 110",
  "ARM 2, 250",
  "ARM 710"
  };
#elif (CLIENT_CPU == CPU_POWERPC && (CLIENT_OS == OS_LINUX || CLIENT_OS == OS_AIX))
static const char *cputypetable[]=
  {
  "PowerPC 601",
  "PowerPC 603/604/750"
  };
#elif (CLIENT_CPU == CPU_68K)
static const char *cputypetable[]=
  {
  "Motorola 68040/060",
  "Motorola 68000/020/030"
  };
#else
  #define NO_CPUTYPE_TABLE
#endif

// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------

s32 Client::SelectCore(void)
{
  ValidateProcessorCount(); //in cpucheck.cpp

#if (CLIENT_CPU == CPU_POWERPC)
  LogScreenRaw( "| RC5 PowerPC assembly by Dan Oetting at USGS\n");

  #if ((CLIENT_OS == OS_BEOS) || (CLIENT_OS == OS_AMIGAOS))
    // Be OS isn't supported on 601 machines
    // There is no 601 PPC board for the Amiga
    cputype = 1; //"PowerPC 603/604/750"
  #elif (CLIENT_OS == OS_WIN32)
    //actually not supported, but just in case
    cputype = 1;
  #else
    {
    if (cputype<0 || cputype>=(int)(sizeof(cputypetable)/sizeof(cputypetable[0])))
      cputype = -1;
        
    if (cputype == -1)
      {
      double fasttime = 0;

      LogScreenRaw("\n| Automatically selecting fastest core...\n"
      "| This is just a guess based on a small test of each core. If you know what\n"
      "| processor this machine has, then please set it in the Performance section\n"
      "| of the client configuration.\n\n");

      for (whichcrunch = 0; whichcrunch < 2; whichcrunch++)
        {
        const s32 benchsize = 500000L;
        Problem problem;
        ContestWork contestwork;

	contestwork.key.lo = contestwork.key.hi = htonl( 0 );
        contestwork.iv.lo = contestwork.iv.hi = htonl( 0 );
        contestwork.plain.lo = contestwork.plain.hi = htonl( 0 );
        contestwork.cypher.lo = contestwork.cypher.hi = htonl( 0 );
        contestwork.keysdone.lo = contestwork.keysdone.hi = htonl( 0 );
        contestwork.iterations.lo = htonl( benchsize );
        contestwork.iterations.hi = htonl( 0 );
        problem.LoadState( &contestwork, 0, benchsize, whichcrunch ); // RC5 core selection

        LogScreenRaw("| Benchmarking the %s core... ", ((whichcrunch)?("second"):("first")));
        problem.Run( 0 ); //threadnum
        double elapsed = CliGetKeyrateForProblemNoSave( &problem );
        LogScreenRaw( "%.1f kkeys/sec\n", (elapsed / 1000.0) );

        if (cputype < 0 || elapsed < fasttime)
          {cputype = whichcrunch; fasttime = elapsed;}
        }
      }
    }
  #endif
  whichcrunch = cputype;
  
  LogScreenRaw( "| Selected %s code.\n", GetCoreNameFromCoreType(cputype) ); 
#elif (CLIENT_CPU == CPU_68K)
  LogScreenRaw( "| RC5 68K assembly by John Girvin\n");
  #if (CLIENT_OS == OS_AMIGAOS)
//  if (cputype<0 || cputype>=(int)(sizeof(cputypetable)/sizeof(cputypetable[0])))
//    cputype=-1;
  if (cputype==-1)
  {
    if (SysBase->AttnFlags & AFF_68040) // Means we have either 040 or 060
      cputype=0;
    else
      cputype=1;
  }
  #endif
  if (cputype == 1)
    rc5_unit_func = rc5_unit_func_000_030;
  else //if (cputype == 0)
    rc5_unit_func = rc5_unit_func_040_060;

  LogScreenRaw( "| Selected %s code.\n", GetCoreNameFromCoreType(cputype) ); 

#elif (CLIENT_CPU == CPU_X86)

  s32 detectedtype = GetProcessorType(); //was x86id() now in cpucheck.cpp
  s32 requestedtype = cputype;

  if (cputype<0 || cputype>=(int)(sizeof(cputypetable)/sizeof(cputypetable[0])))
    {
    requestedtype = detectedtype; //use autodetect
    cputype = (requestedtype & 0xFF);
    }

  LogScreenRaw("RC5: Selecting %s core.\n", GetCoreNameFromCoreType(cputype));
  
  #if ((defined(KWAN) || defined(MEGGS)) && !defined(MMX_BITSLICER))
    #define DESUNITFUNC51 des_unit_func_slice
    #define DESUNITFUNC52 des_unit_func_slice
    #define DESUNITFUNC61 des_unit_func_slice
    #define DESUNITFUNC62 des_unit_func_slice
  #elif defined(MULTITHREAD)
    #define DESUNITFUNC51 p1des_unit_func_p5
    #define DESUNITFUNC52 p2des_unit_func_p5
    #define DESUNITFUNC61 p1des_unit_func_pro
    #define DESUNITFUNC62 p2des_unit_func_pro
  #else
    #define DESUNITFUNC51 p1des_unit_func_p5
    #define DESUNITFUNC52 p1des_unit_func_p5
    #define DESUNITFUNC61 p1des_unit_func_pro
    #define DESUNITFUNC62 p1des_unit_func_pro
  #endif

  if (cputype == 1) // Intel 386/486
    {
    rc5_unit_func = rc5_unit_func_486;
    des_unit_func = DESUNITFUNC51;  //p1des_unit_func_p5;
    des_unit_func2 = DESUNITFUNC52; //p2des_unit_func_p5;
    }
  else if (cputype == 2) // Ppro/PII
    {
    rc5_unit_func = rc5_unit_func_p6;
    des_unit_func =  DESUNITFUNC61;  //p1des_unit_func_pro;
    des_unit_func2 = DESUNITFUNC62;  //p2des_unit_func_pro;
    }
  else if (cputype == 3) // 6x86(mx)
    {
    rc5_unit_func = rc5_unit_func_6x86;
    des_unit_func =  DESUNITFUNC61;  //p1des_unit_func_pro;
    des_unit_func2 = DESUNITFUNC62;  //p2des_unit_func_pro;
    }
  else if (cputype == 4) // K5
    {
    rc5_unit_func = rc5_unit_func_k5;
    des_unit_func =  DESUNITFUNC51;  //p1des_unit_func_p5;
    des_unit_func2 = DESUNITFUNC52;  //p2des_unit_func_p5;
    }
  else if (cputype == 5) // K6/K6-2
    {
    rc5_unit_func = rc5_unit_func_k6;
    des_unit_func =  DESUNITFUNC61;  //p1des_unit_func_pro;
    des_unit_func2 = DESUNITFUNC62;  //p2des_unit_func_pro;
    }
  #ifdef MMX_BITSLICER
  else if ( (requestedtype & 0xFF) == 6 ) // Pentium MMX ONLY
    {
    if ((detectedtype & 0x100) == 0x100)
      {
      rc5_unit_func = rc5_unit_func_p5_mmx;
      des_unit_func = DESUNITFUNC51;  //p1des_unit_func_p5;
      des_unit_func2 = DESUNITFUNC52; //p2des_unit_func_p5;
      }
    else // not really MMX capable
      {
      LogScreenRaw("The processor is not MMX instruction capable.\n"
            "Reverting to the Pentium Classic core.\n");
      rc5_unit_func = rc5_unit_func_p5;
      des_unit_func =  DESUNITFUNC51;  //p1des_unit_func_p5;
      des_unit_func2 = DESUNITFUNC52;  //p2des_unit_func_p5;
      };     
    } 
  #endif
  else // Pentium (Classic/MMX) + others
    {
    rc5_unit_func = rc5_unit_func_p5;
    des_unit_func =  DESUNITFUNC51;  //p1des_unit_func_p5;
    des_unit_func2 = DESUNITFUNC52;  //p2des_unit_func_p5;
    cputype = 0;
    }

  #if (defined(MMX_BITSLICER) && defined(KWAN) && defined(MEGGS))
  if ((detectedtype & 0x100) && usemmx)   // use the MMX DES core ?
    { 
    des_unit_func = des_unit_func2 = des_unit_func_mmx;
    LogScreen("DES: Selecting MMX Bitslice core.\n");
    }
  else
  #endif
    {
    if (des_unit_func == DESUNITFUNC51)
      LogScreen("DES: Selecting Pentium optimized Bryddes core.\n");
    else LogScreen("DES: Selecting Pentium Pro optimized Bryddes core.\n");
    };
      
  #undef DESUNITFUNC61
  #undef DESUNITFUNC62
  #undef DESUNITFUNC51
  #undef DESUNITFUNC52

#elif (CLIENT_CPU == CPU_ARM)
#if (CLIENT_OS == OS_RISCOS)
    if (cputype<0 || cputype>=(int)(sizeof(cputypetable)/sizeof(cputypetable[0])))
	cputype = -1;
    if (cputype == -1)              //was ArmID(). Now in cpucheck.cpp
	cputype = GetProcessorType(); // will return -1 if unable to identify
#endif
    if (cputype == -1)
    {
	const s32 benchsize = 50000*2; // pipeline count is 2
	double fasttime[2] = { 0, 0 };
	s32 fastcoretest[2] = { -1, -1 };
	
	LogScreenRaw("Automatically selecting fastest core...\n"
		     "This is just a guess based on a small test of each core. If you know what\n"
		     "processor this machine has, then please set it in the Performance section\n"
		     "of the client configuration.\n");
	

	for (int contestid = 0; contestid < 2; contestid++)
	{
	    for (int whichcrunch = 0; whichcrunch < 3; whichcrunch++)
	    {
		Problem problem;
		ContestWork contestwork;
		contestwork.key.lo = contestwork.key.hi = htonl( 0 );
		contestwork.iv.lo = contestwork.iv.hi = htonl( 0 );
		contestwork.plain.lo = contestwork.plain.hi = htonl( 0 );
		contestwork.cypher.lo = contestwork.cypher.hi = htonl( 0 );
		contestwork.keysdone.lo = contestwork.keysdone.hi = htonl( 0 );
		contestwork.iterations.lo = htonl( benchsize );
		contestwork.iterations.hi = htonl( 0 );
		problem.LoadState( &contestwork , contestid, benchsize, whichcrunch ); 

		if (contestid == 0)
		{
		    // there are now 3 RC5 cores from which to choose
		    // probably be a 4th one soon
		    switch(whichcrunch)
		    {
		    case 1:
			rc5_unit_func = rc5_unit_func_arm_2;
			break;
		    case 2:
			rc5_unit_func = rc5_unit_func_arm_3;
			break;
		    default:
			rc5_unit_func = rc5_unit_func_arm_1;
			break;
		    }
		}
		else
		{
		    // select the correct DES core engine
		    switch(whichcrunch)
		    {
		    case 1:
			des_unit_func = des_unit_func_strongarm;
			break;
		    default:
			des_unit_func = des_unit_func_arm;
			break;
		    }
		}
		problem.Run( 0 ); //threadnum
		
		double elapsed = CliGetKeyrateForProblemNoSave( &problem );
		//printf("%s Core %d: %f\n",contestid ? "DES" : "RC5",whichcrunch,elapsed);
		
		if (fastcoretest[contestid] < 0 || elapsed < fasttime[contestid])
		{
		    fastcoretest[contestid] = whichcrunch; fasttime[contestid] = elapsed;
		}
	    }
	}
	cputype = (8-(fastcoretest[0] + ((fastcoretest[1]&1)<<2)))&7;
	if (cputype == 1)
	{
	    cputype = 3;
	}
	else if (cputype > 3)
	{
	    cputype = 1;
	}
    }
    LogScreenRaw("Selecting %s code.\n",GetCoreNameFromCoreType(cputype));
    
    // select the correct core engine
    switch(cputype)
    {
    case 0:
	rc5_unit_func = rc5_unit_func_arm_1;
	des_unit_func = des_unit_func_arm;
	break;
    default:
    case 1:
	rc5_unit_func = rc5_unit_func_arm_3;
	des_unit_func = des_unit_func_strongarm;
	break;
    case 2:
	rc5_unit_func = rc5_unit_func_arm_2;
	des_unit_func = des_unit_func_strongarm;
	break;
    case 3:
	rc5_unit_func = rc5_unit_func_arm_3;
	des_unit_func = des_unit_func_arm;
	break;
    }
    
#else
  cputype = 0;
#endif
  return 0;
}

// ---------------------------------------------------------------------------
