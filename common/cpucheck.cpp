// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: cpucheck.cpp,v $
// Revision 1.34  1998/10/30 00:07:19  foxyloxy
//
// Rectify some deviations from the standard of "-1" means detection
// failed.
//
// Revision 1.33  1998/10/11 00:43:20  cyp
// Implemented 'quietly' in SelectCore() and ValidateProcessorCount()
//
// Revision 1.32  1998/10/09 12:25:21  cyp
// ValidateProcessorCount() is no longer a client method [is now standalone].
//
// Revision 1.31  1998/10/09 01:39:27  blast
// Added selcore.h to list of includes. Changed 68k print to use 
// GetCoreNameFromCoreType(). Added Celeron-A to the CPU list so that 
// it finally uses the right cores .. :) instead of being unknown :)
//
// Revision 1.30  1998/10/08 21:23:04  blast
// Fixed Automatic CPU detection that cyp had written a little strangely
// for 68K CPU's under AmigaOS. It was good thinking but it would've 
// reported the wrong cpu type, and also, there is no 68050, cyp :)
//
// Revision 1.29  1998/10/08 16:47:17  cyp
// Fixed a missing ||
//
// Revision 1.28  1998/10/08 11:05:26  cyp
// Moved AmigaOS 68k hardware detection code from selcore.cpp to cpucheck.cpp
//
// Revision 1.27  1998/10/08 10:04:21  cyp
// GetProcessorType() is now standalone (no longer a Client::method).
//
// Revision 1.26  1998/09/28 02:44:47  cyp
// removed non-mt limit; removed references to MAXCPUS; turned static function
// __GetProcessorCount() into public GetNumberOfDetectedProcessors(); created
// GetNumberOfSupportedProcessors() ["supported" by the client that is]
//
// Revision 1.24  1998/08/14 00:05:03  silby
// Changes for rc5 mmx core integration.
//
// Revision 1.23  1998/08/10 20:15:04  cyruspatel
// Setting cpunum to zero on multi-threading platforms now forces the client
// to run in non-mt mode.
//
// Revision 1.22  1998/08/05 18:41:12  cyruspatel
// Converted more printf()s to LogScreen()s, changed some Log()/LogScreen()s
// to LogRaw()/LogScreenRaw()s, ensured that DeinitializeLogging() is called,
// and InitializeLogging() is called only once (*before* the banner is shown)
//
// Revision 1.21  1998/08/05 16:40:53  cberry
// fixed typo in ARM part of GetProcessorInformationStrings()
//
// Revision 1.20  1998/08/02 16:17:53  cyruspatel
// Completed support for logging.
//
// Revision 1.19  1998/07/18 17:05:39  cyruspatel
// Lowered the TimesliceBaseline for 486's from 1024 to 512. The high tslice
// was causing problems on 486s with 3com and other poll-driven NICs.
//
// Revision 1.17  1998/07/13 23:39:33  cyruspatel
// Added functions to format and display raw cpu info for better management
// of the processor detection functions and tables. Well, not totally raw,
// but still less cooked than SelectCore(). All platforms are supported, but
// the data may not be meaningful on all. The info is accessible to the user
// though the -cpuinfo switch.
//
// Revision 1.16  1998/07/12 00:40:15  cyruspatel
// Corrected a cosmetic boo-boo. ("... failed to detect an processor")
//
// Revision 1.15  1998/07/11 09:47:18  cramer
// Added support for solaris numcpu auto detection.
//
// Revision 1.14  1998/07/11 02:34:49  cramer
// Added automagic number of cpu detection for linux.  If it cannot detect the
// number of processors, a warning is issued and we assume it's only got one.
// (Note to Linus: add num_cpus to struct sysinfo.)
//
// Revision 1.13  1998/07/09 03:22:54  silby
// Changed so that IDT winchip WILL use mmx now also.
//
// Revision 1.12  1998/07/08 09:50:36  remi
// Added support for the MMX bitslicer.
// GetProcessorType() & 0x100 != 0 if we should use the MMX DES core.
// GetProcessorType() & 0x200 != 0 if we should use the MMX RC5 core.
//
// Revision 1.11  1998/07/07 21:55:37  cyruspatel
// client.h has been split into client.h and baseincs.h 
//
// Revision 1.10  1998/07/06 09:17:23  jlawson
// eliminated unused value assignment warning.
//
// Revision 1.9  1998/07/05 12:42:40  cyruspatel
// Created cpucheck.h to support makefiles that rely on autodependancy info
// to detect file changes.
//
// Revision 1.8  1998/07/05 06:55:06  silby
// Change to the bitmask for intel CPUIDs so that secondary CPUs will be id'd correctly.
//
// Revision 1.7  1998/06/28 19:48:13  silby
// Changed default amd 486 core selection to pentium core and changed strings to reflect that.
//
// Revision 1.6  1998/06/23 20:22:05  cyruspatel
// Added new function: GetTimesliceBaseline() returns a value that the
// ideal RC5 keyrate (kKeys per Mhz) would be IF a machine were running
// at peak efficiency. For non-preemptive systems, it is thus a good
// indicator of how low we can set the timeslice/rate-of-yield without
// losing efficiency. Or inversely, at what point OS responsiveness starts
// to suffer - which also applies to preemptive but non-mt systems handling
// of a break request. - Currently only supports all the x86 OS's.
//
// Revision 1.5  1998/06/22 10:28:22  kbracey
// Just tidying
//
// Revision 1.4  1998/06/22 09:37:47  cyruspatel
// Fixed another cosmetic bug.
//
// Revision 1.3  1998/06/22 03:40:23  cyruspatel
// Fixed a numcputemp/cpu_count variable mixup. 
//
// Revision 1.1  1998/06/21 17:12:02  cyruspatel
// Created
//
//
#if (!defined(lint) && defined(__showids__))
const char *cpucheck_cpp(void) {
return "@(#)$Id: cpucheck.cpp,v 1.34 1998/10/30 00:07:19 foxyloxy Exp $"; }
#endif

#include "cputypes.h"
#include "baseincs.h"  // for platform specific header files
#include "cpucheck.h"  //just to keep the prototypes in sync.
#include "threadcd.h"  //for the OS_SUPPORTS_THREADING define
#include "logstuff.h"  //LogScreen()/LogScreenRaw()

#if (CLIENT_OS == OS_SOLARIS)
#include <unistd.h>    // cramer - sysconf()
#elif (CLIENT_OS == OS_IRIX)
#include <sys/prctl.h>
#endif

// --------------------------------------------------------------------------

unsigned int GetNumberOfSupportedProcessors( void )
{
  return ( 128 ); /* just some arbitrary number */
}

// -------------------------------------------------------------------------

int GetNumberOfDetectedProcessors( void )  //returns -1 if not supported
{
  static int cpucount = -2;

  if (cpucount == -2)
    {
    cpucount = -1;
    #if (CLIENT_OS == OS_BEOS)
      {
      system_info the_info;
      get_system_info(&the_info);
      cpucount = the_info.cpu_count;
      }
    #elif (CLIENT_OS == OS_WIN32)
      {
      SYSTEM_INFO systeminfo;
      GetSystemInfo(&systeminfo);
      cpucount = systeminfo.dwNumberOfProcessors;
      if (cpucount < 1)
        cpucount = 1;
      }
    #elif (CLIENT_OS == OS_NETWARE)
      {
      cpucount = nwCliGetNumberOfProcessors();
      }
    #elif (CLIENT_OS == OS_OS2)
      {
      int rc = (int) DosQuerySysInfo(QSV_NUMPROCESSORS, QSV_NUMPROCESSORS,
                  &cpucount, sizeof(cpucount));
      // check if call is valid if not, default to -1
      if (rc!=0 || cpucount < 1)
        cpucount = -1;
      }
    #elif (CLIENT_OS == OS_LINUX)
      { // cramer -- yes, I'm cheating, but it's the only way...
      char buffer[256];
      cpucount = 0;
      if (FILE *cpuinfo = fopen("/proc/cpuinfo", "r"))
        {
        while(fgets(buffer, 256, cpuinfo))
	  {
          if (strstr(buffer, "processor") == buffer)
            cpucount++;
	  }
	fclose(cpuinfo);
	}
      }
    #elif (CLIENT_OS == OS_IRIX)
      {
      cpucount = (int)prctl( PR_MAXPPROCS, 0, 0);
      }
    #elif (CLIENT_OS == OS_SOLARIS)
      {
      cpucount = sysconf(_SC_NPROCESSORS_ONLN);
      }
    #endif
    if (cpucount < 1)  //not supported
      cpucount = -1;
    }
  return cpucount;
}

// --------------------------------------------------------------------------

unsigned int ValidateProcessorCount( int numcpu, int quietly )
{
  static int detected_count = -2;

  //-------------------------------------------------------------
  // Validate processor/thread count.
  // --------------------
  // The use of -numcpu on the command line is legal for non-mt clients.
  // Simply put, if it ain't MULTITHREAD, there's ain't gonna be a thread,
  // no matter what the user puts on the command line or in the ini file.
  //
  // On MULTITHREAD systems, numcpu with zero implies "force_non-mt"
  //
  // The MAX CPUS limit is no longer the business of the select cpu logic.
  // The thread data and Problem tables are (or rather, will soon be) grown
  // dynamically.
  //
  // Whether a CLIENT_CPU (core) is thread safe or not the responsibility of
  // select cpu logic either.
  // 
  //--------------------

  #ifndef MULTITHREAD           //this is the only place in the config where
  detected_count = 0;           //we check this - implies force non-mt
  #endif
  
  if (numcpu < 0)                //numcpu == 0 implies force non-mt; 
    {                            //numcpu < 0  implies autodetect
    if (detected_count == -2)
      {
      detected_count = GetNumberOfDetectedProcessors();
      // returns -1 if no hardware detection
      if ( detected_count < 1 )
        {
        if (!quietly)
          LogScreen("Automatic processor count detection failed.\n"
                    "A single processor machine is assumed.\n");
        detected_count = 1;
        }
      else
        {
        if (!quietly)
          LogScreen("Automatic processor detection found %d processor%s.\n",
             detected_count, ((detected_count==1)?(""):("s")) );
        }
      }
    numcpu = detected_count;
    if (numcpu < 0) //zero is legal for multithread (implies force non-mt)
      numcpu = 1;
    }
  return (unsigned int)numcpu;
}

// --------------------------------------------------------------------------

#if (!((CLIENT_CPU == CPU_X86) || \
      ((CLIENT_CPU == CPU_68K) && (CLIENT_OS == OS_AMIGAOS)) || \
      ((CLIENT_CPU == CPU_ARM) && (CLIENT_OS == OS_RISCOS)) ))
int GetProcessorType(int quietly)
{ 
  if (!quietly)
    LogScreen("Automatic processor detection is not supported.\n");
  return -1; 
}
#endif

// --------------------------------------------------------------------------

#if ((CLIENT_CPU == CPU_68K) && (CLIENT_OS == OS_AMIGAOS))
int GetProcessorType(int quietly)
{    
  static int detectedtype = -1;
  long flags;
  if (detectedtype == -1)
    {
    flags = (long)(SysBase->AttnFlags);
    if ((flags & AFF_68060) && (flags & AFF_68040) && (flags & AFF_68030))
        detectedtype = 5; // Phase5 060 boards at least report this...
    else if ((flags & AFF_68040) && (flags & AFF_68030))
        detectedtype = 4; // 68040
    else if ((flags & AFF_68030) && (flags & AFF_68020))
        detectedtype = 3; // 68030
    else if ((flags & AFF_68020) && (flags & AFF_68010))
        detectedtype = 2; // 68020
    else if (flags & AFF_68010)
        detectedtype = 1; // 68010
    else
        detectedtype = 0; // Assume a 68000
    }
  if (!quietly)
    {
    int x68k = detectedtype; if (x68k == 5) x68k = 6;
    LogScreen("Automatic processor detection found a Motorola 680%d0\n",x68k);
    }
  return (detectedtype);
}    
#endif

// --------------------------------------------------------------------------

#if (CLIENT_CPU == CPU_X86)

#ifdef __WATCOMC__
  #define x86ident _x86ident
#endif
#if (CLIENT_OS == OS_LINUX) && !defined(__ELF__)
  extern "C" u32 x86ident( void ) asm ("x86ident");
#else
  extern "C" u32 x86ident( void );
#endif

struct _cpuxref { int cpuidb;
                  unsigned int kKeysPerMhz; // numbers are from Alde's tables
                  int coretouse; // & 0x100 == processor is MMX capable
                  char *cpuname; };

struct _cpuxref *__GetProcessorXRef( int *cpuidbP, int *vendoridP,
                               char **articleP, char **vendornameP )
{
  char *article = NULL; //"an" "a"
  char *vendorname = NULL; //"Cyrix", "Centaur", "AMD", "Intel", ""
  static struct _cpuxref *cpuxref = NULL;

  u32 detectedvalue = x86ident(); //must be interpreted
  int vendorid = (int)(detectedvalue >> 16);
  int cpuidb  = (int)(detectedvalue & 0xffff);
  
  if (cpuidbP) *cpuidbP = cpuidb;
  if (vendoridP) *vendoridP = vendorid;

  if ( vendorid == 0x7943) // Cyrix CPU
    {
    article = "a";
    vendorname = "Cyrix";
    cpuidb &= 0xfff0; //strip last 4 bits, don't need stepping info
    static struct _cpuxref __cpuxref[]={
      {    0x40, 0512,     0, "486"     }, // use Pentium core
      {  0x0440, 0512,     0, "MediaGX" },
      {  0x0490, 1185,     0, "5x86"    },
      {  0x0520, 2090,     3, "6x86"    }, // "Cyrix 6x86/6x86MX/M2"
      {  0x0540, 1200,     0, "GXm"     }, // use Pentium core here too
      {  0x0600, 2115, 0x103, "6x86MX"  },
      {  0x0000, 2115,     3, NULL      } //default core == 6x86
      }; cpuxref = &__cpuxref[0];
    }
  else if ( vendorid == 0x6543) //centaur/IDT cpu
    {
    article = "a";
    vendorname = "Centaur/IDT";
    cpuidb &= 0xfff0; //strip last 4 bits, don't need stepping info
    static struct _cpuxref __cpuxref[]={
      {  0x0540, 1200,0x100, "C6"      }, // use Pentium core
      {  0x0000, 1200,    0, NULL      }  // default core == Pentium
      }; cpuxref = &__cpuxref[0];
    }
  else if ( vendorid == 0x7541) // AMD CPU
    {
    article = "an";
    vendorname = "AMD";
    cpuidb &= 0xfff0; //strip last 4 bits, don't need stepping info
    static struct _cpuxref __cpuxref[]={
      {  0x0040, 0512,     0, "486"      },   // "Pentium, Pentium MMX, Cyrix 486/5x86/MediaGX, AMD 486",
      {  0x0430, 0512,     0, "486DX2"   },
      {  0x0470, 0512,     0, "486DX2WB" },
      {  0x0480, 0512,     0, "486DX4"   },
      {  0x0490, 0512,     0, "486DX4WB" },
      {  0x04E0, 1185,     0, "5x86"     },
      {  0x04F0, 1185,     0, "5x86WB"   },
      {  0x0500, 2353,     4, "K5 PR75, PR90, or PR100" }, // use K5 core
      {  0x0510, 2353,     4, "K5 PR120 or PR133" },
      {  0x0520, 2353,     4, "K5 PR166" },
      {  0x0530, 2353,     4, "K5 PR200" },
      {  0x0560, 1611, 0x105, "K6"       },
      {  0x0570, 1611, 0x105, "K6"       },
      {  0x0580, 1611, 0x105, "K6-2"     },
      {  0x0590, 1611, 0x105, "K6-3"     },
      {  0x0000, 1611,     5, NULL       }   // for the future - default core = K6
      }; cpuxref = &__cpuxref[0];
    }
  else if (vendorid == 0x6E49 || vendorid == 0x6547) // Intel CPU
    {
    article = "an";
    vendorname = "Intel";
    if ((cpuidb == 0x30) || (cpuidb == 0x40))
      vendorname = ""; //generic 386/486
    cpuidb &= 0x0ff0; //strip last 4 bits, don't need stepping info
    static struct _cpuxref __cpuxref[]={
      {  0x0030, 0426,     1, "80386"    },   // generic 386/486 core
      {  0x0040, 0512,     1, "80486"    },
      {  0x0400, 0512,     1, "486DX 25 or 33" },
      {  0x0410, 0512,     1, "486DX 50" },
      {  0x0420, 0512,     1, "486SX" },
      {  0x0430, 0512,     1, "486DX2" },
      {  0x0440, 0512,     1, "486SL" },
      {  0x0450, 0512,     1, "486SX2" },
      {  0x0470, 0512,     1, "486DX2WB" },
      {  0x0480, 0512,     1, "486DX4" },
      {  0x0490, 0512,     1, "486DX4WB" },
      {  0x0500, 1416,     0, "Pentium" }, //stepping A
      {  0x0510, 1416,     0, "Pentium" },    
      {  0x0520, 1416,     0, "Pentium" },
      {  0x0530, 1416,     0, "Pentium Overdrive" },
      {  0x0540, 1432, 0x106, "Pentium MMX" },
      {  0x0570, 1416,     0, "Pentium" },
      {  0x0580, 1432, 0x106, "Pentium MMX" },
      {  0x0600, 2785,     2, "Pentium Pro" },
      {  0x0610, 2785,     2, "Pentium Pro" },
      {  0x0630, 2785, 0x102, "Pentium II" },
      {  0x0650, 2785, 0x102, "Pentium II" },
      {  0x0660, 2785, 0x102, "Celeron-A" },
      {  0x0000, 2785,     2, NULL         }  // default core = PPro/PII
      }; cpuxref = &__cpuxref[0];
    }

  if (articleP) *articleP = article;
  if (vendornameP) *vendornameP = vendorname;

  if ( cpuxref != NULL ) // we have a mfg's table
    {
    unsigned int pos;
    for (pos=0 ; ; pos++)
      {
      if (( (cpuxref[pos].cpuname)==NULL ) ||
            ( cpuidb == (cpuxref[pos].cpuidb)) )
        return (&(cpuxref[pos]));
      }
    }
  return NULL;
}  

// ---------------------

int GetProcessorType(int quietly)
{
  int coretouse;
  int vendorid, cpuidb;                           
  char *article, *vendorname;
  struct _cpuxref *cpuxref = 
          __GetProcessorXRef( &cpuidb, &vendorid, &article, &vendorname );

  const char *apd = "Automatic processor detection";
  if ( cpuxref == NULL ) // fell through
    {
    if (!quietly)
      LogScreen( "%s failed. (id: %04X:%04X)\n", apd, vendorid, cpuidb );
    coretouse = 0;
    }
  else if ( cpuxref->cpuname == NULL )  // fell through to last element
    {
    coretouse = (cpuxref->coretouse);
    if (!quietly)
      LogScreen("%s found an unrecognized %s processor. (id: %04X)\n", apd,
                                                    vendorname, cpuidb );
    }
  else // if ( cpuidb == (cpuxref->cpuidb))
    {
    coretouse = (cpuxref->coretouse);      
    if (!quietly)
      {
      if ( !vendorname || !*vendorname )  // generic type - no vendor name
        LogScreen( "%s found %s %s.\n", apd, article, (cpuxref->cpuname));
      else
        LogScreen( "%s found %s %s %s.\n", apd, article,
                                       vendorname, (cpuxref->cpuname) );
      }
    }
  return coretouse;
}
#endif // client_cpu == x86 or not

// --------------------------------------------------------------------------

#if ((CLIENT_CPU == CPU_ARM) && (CLIENT_OS == OS_RISCOS))

#include <setjmp.h>

static jmp_buf ARMident_jmpbuf;

static void ARMident_catcher(int)
{
  longjmp(ARMident_jmpbuf, 1);
}

static u32 GetARMIdentification(void)
{
  static u32 detectedvalue = 0x0;

  if ( detectedvalue != 0x0 )
    return detectedvalue;

  // ARMident() will throw SIGILL on an ARM 2 or ARM 250, because
  // they don't have the system control coprocessor. (We ignore the
  // ARM 1 because I'm not aware of any existing C++ compiler that
  // targets it...)

  signal(SIGILL, ARMident_catcher);

  if (setjmp(ARMident_jmpbuf))
  {
    detectedvalue = 0x2000;
  }
  else
  {
    detectedvalue = ARMident();
  }

  signal(SIGILL, SIG_DFL);

  detectedvalue = (detectedvalue >> 4) & 0xfff; // extract part number field

  if ((detectedvalue & 0xf00) == 0)
  {
    // an old-style ID (ARM 3 or prototype ARM 600) - shift it into the new form
    detectedvalue <<= 4;
  }

  if (detectedvalue == 0x300)
  {
    detectedvalue = 3;
  }
  else if (detectedvalue == 0x710)
  {
    // the ARM 7500 returns an ARM 710 ID - need to look at its
    // integral IOMD unit to spot the difference
    u32 detectediomd = IOMDident();
    detectediomd &= 0xff00; // just want most significant byte

    if (detectediomd == 0x5b00)
      detectedvalue = 0x7500;
    else if (detectediomd == 0xaa00)
      detectedvalue = 0x7500FE;
  }
  return detectedvalue;
}  

int GetProcessorType(int quietly)
{
  u32 detectedvalue = GetARMIdentification(); //must be interpreted
  int coretouse; // the core the client should use
  char apd[40];

  apd[0]=0;
  switch (detectedvalue)
    {
    case 0x200:
      strcpy(apd, "found an ARM 2 or ARM 250.");
      coretouse=2;
      break;
    case 0x3:
    case 0x600:
    case 0x610:
    case 0x700:
    case 0x7500:
    case 0x7500FE:
      coretouse=0;
      break;
    case 0x710:
      coretouse=3;
      break;
    case 0x810:
      coretouse=1;
      break;
    case 0xA10:
      strcpy(apd, "found a StrongARM 110." );
      coretouse=1;
      break;
    default:
      sprintf(apd, "failed. (id: %08X)", detectedvalue);
      coretouse=-1;
      break;
    }
  if (!quietly)
    {
    if (!apd[0]) sprintf(apd, "found an ARM %X.", detectedvalue);
    LogScreen( "Automatic processor detection %s\n", apd );
    }
  return coretouse;
}

#endif //Arm/riscos cpucheck

// --------------------------------------------------------------------------

// GetTimesliceBaseline() returns a value that the ideal RC5 keyrate (kKeys 
// per Mhz) would be IF a machine were running at peak efficiency. For 
// non-preemptive systems, it is thus a good indicator of how low we can 
// set the timeslice/rate-of-yield without losing efficiency. Or inversely, 
// at what point OS responsiveness starts to suffer - which also applies to 
// preemptive but non-mt systems handling of a break request. 
//
// The function can also be used on non-mt systems to check for an excessive
// timeslice - on x86 systems an excessive timeslice is > 4*baseline

unsigned int GetTimesliceBaseline(void) 
{ 
#if (CLIENT_CPU == CPU_X86)    
  struct _cpuxref *cpuxref = __GetProcessorXRef( NULL, NULL, NULL, NULL );
  return ((!cpuxref) ? ( 0512 ) : (cpuxref->kKeysPerMhz) );
   //(cpuxref->kKeysPerMhz * (((cpuxref->coretouse & 0xF00 )!=0)?2:1) ) );
#else 
  return 0;
#endif    
}  

// --------------------------------------------------------------------------

// Originally intended as tool to assist in managing the processor ID table 
// for x86, I now (after writing it) realize that it could also get users on
// non-x86 platforms to send us *their* processor detection code. :) - Cyrus 

void GetProcessorInformationStrings( const char ** scpuid, const char ** smaxscpus, const char ** sfoundcpus )
{
  const char *maxcpu_s, *foundcpu_s, *cpuid_s;
#if (CLIENT_CPU == CPU_X86)    
  static char cpuid_b[12];
  int vendorid, cpuidb;                     
  __GetProcessorXRef( &cpuidb, &vendorid, NULL, NULL );
  sprintf( cpuid_b, "%04X:%04X", vendorid, cpuidb );
  cpuid_s = ((const char *)(&cpuid_b[0]));      
#elif ((CLIENT_CPU == CPU_ARM) && (CLIENT_OS == OS_RISCOS))
  static char cpuid_b[10];
  u32 cpuidb = GetARMIdentification();
  if (cpuidb == 0x0200)
    cpuid_s = "ARM 2 or ARM 250";
  else if (cpuidb == 0x0A10)
    cpuid_s = "StrongARM 110";
  else
    {
    sprintf( cpuid_b, "%X", cpuidb );
    cpuid_s = ((const char *)(&cpuid_b[0]));      
    }
#else
  cpuid_s = "none (client does not support identification)";
#endif    

  #if defined(MULTITHREAD)
    static char maxcpu_b[80]; 
    sprintf( maxcpu_b, "%d (threading may be disabled "
         "by setting -numcpu to zero)", GetNumberOfSupportedProcessors() ); 
    maxcpu_s = ((const char *)(&maxcpu_b[0]));
  #elif defined(OS_SUPPORTS_THREADING) //from threadcd.h
    maxcpu_s = "1 (threading is emulated - client built without thread support)";
  #elif ((CLIENT_CPU != CPU_X86) && (CLIENT_CPU != CPU_88K) && \
        (CLIENT_CPU != CPU_SPARC) && (CLIENT_CPU != CPU_POWERPC))
    maxcpu_s = "1 (threading is emulated - cores are not thread-safe)";
  #else
    maxcpu_s = "1 (threading is emulated - OS does not support threads)";
  #endif  

  int cpucount = GetNumberOfDetectedProcessors();
  if (cpucount < 1)
    foundcpu_s = "1 (OS does not support detection)";
  else
    {  
    static char foundcpu_b[6]; 
    sprintf( foundcpu_b, "%d", cpucount );
    foundcpu_s = ((const char *)(&foundcpu_b[0]));
    }

  if ( scpuid ) *scpuid = cpuid_s;
  if ( smaxscpus ) *smaxscpus = maxcpu_s; 
  if ( sfoundcpus ) *sfoundcpus = foundcpu_s; 
  return;
}  

// --------------------------------------------------------------------------

void DisplayProcessorInformation(void)
{
  const char *scpuid, *smaxscpus, *sfoundcpus;
  GetProcessorInformationStrings( &scpuid, &smaxscpus, &sfoundcpus );
  
  LogScreen("Automatic processor identification tag:\n\t%s\n"
   "Number of processors detected by this client:\n\t%s\n"
   "Number of processors supported by each instance of this client:\n\t%s\n",
   scpuid, sfoundcpus, smaxscpus );
  return;
}     
 
// --------------------------------------------------------------------------
