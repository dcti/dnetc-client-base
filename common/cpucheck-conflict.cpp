// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
/*
   Implementing long __GetRawProcessorID( const char **cpuname ):
   
   if identification failed: 
               return -1L, set cpuname to NULL 
   if identification is not supported:
               return -2L, set cpuname to NULL 
   if we have an ID and a name:
               return ID and fully formatted name (eg "Alpha EV5.6 (21164PC)")
   if we have an ID but no name: 
               return ID, set cpuname to ""
   if we have a name, but no ID: 
               return ID==0, and set cpuname to the raw name (eg "PCA56" )
*/
// $Log: cpucheck-conflict.cpp,v $
// Revision 1.74  1999/04/02 20:56:41  jlawson
// standardized naming of raw cpu type identification functions and cleaned
// many other aspects and formatting of common cpu reporting.
//
// Revision 1.73  1999/04/01 07:16:24  jlawson
// cleaned formatting. alpha linux now defaults to 1 processor instead of
// returning 0 on non-smp kernels.
//
// Revision 1.72  1999/04/01 01:05:37  cyp
// brought (hopefully correctly) Alpha detection into line with all the
// others. This module is not the easiest code to maintain, so would porters
// please look around when adding support? Many thanks.
//
// Revision 1.71  1999/02/23 04:45:35  silby
// Added Pentium III to x86 cpu list.
//
// Revision 1.70  1999/01/31 20:19:08  cyp
// Discarded all 'bool' type wierdness. See cputypes.h for explanation.
//
// Revision 1.69  1999/01/29 18:52:30  jlawson
// fixed formatting.  changed some int vars to bool.
//
// Revision 1.68  1999/01/29 04:15:35  pct
// Updates for the initial attempt at a multithreaded/multicored Digital
// Unix Alpha client.  Sorry if these changes cause anyone any grief.
//
// Revision 1.67  1999/01/21 19:34:52  michmarc
// changes to the #ifdef around GetProcessorType broke the link setp on
// some platforms. ((CLIENT_CPU != CPU_ALPHA) && (CLIENT_OS != OS_DEC_UNIX))
// to  ((CLIENT_CPU != CPU_ALPHA) || (CLIENT_OS != OS_DEC_UNIX))
// It should be using the stub except on Alpha/Dec, and the logical inverse
// of (Alpha && DecUnix) is (!Alpha || !DecUnix)
// [Maybe this time it will actually link on all platforms.]
//
// Revision 1.66  1999/01/19 12:13:17  patrick
// changes to the #ifdef around GetProcessorType broke the link setp on some
// platforms. Changed therefore
// ((CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS != CPU_DIGITAL_UNIX))
// to ((CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_DEC_UNIX))
// I hope this was the intention of the one who originally introduced this.
//
// Revision 1.65  1999/01/19 09:54:35  patrick
// move AIX include <unistd.h> to basincs.h
//
// Revision 1.64  1999/01/18 12:12:34  cramer
// - Added code for ncpu detection for linux/alpha
// - Corrected the alpha RC5 core handling (support "timeslice")
// - Changed the way selftest runs... it will not stop if a test fails,
//     but will terminate at the end of each contest selftest if any test
//     failed.  Interrupting the test is seen as the remaining tests
//     having failed (to be fixed later)
//
// Revision 1.63  1999/01/18 00:33:19  remi
// Arrg!
//
// Revision 1.62  1999/01/18 00:24:42  remi
// Added IDT WinChip 2 to the list.
//
// Revision 1.61  1999/01/15 20:22:49  michmarc
// Fix GetProcessorType for Non-Digital-Unix Alpha platforms
//
// Revision 1.60  1999/01/14 23:02:12  pct
// Updates for Digital Unix alpha client and ev5 related code.  This also
// includes inital code for autodetection of CPU type and SMP.
//
// Revision 1.59  1999/01/13 15:17:02  kbracey
// Fixes to RISC OS processor detection and scheduling
//
// Revision 1.58  1999/01/13 10:46:15  cramer
// Cosmetic update (comments and indenting)
//
// Revision 1.57  1999/01/12 16:36:13  cyp
// Made failed cpu count detection message sound, uh, less severe.
//
// Revision 1.56  1999/01/11 20:55:04  patrick
// numCPU support for AIX enabled
//
// Revision 1.55  1999/01/06 22:17:36  dicamillo
// Support PPC prototype Macs  some developers still have.
//
// Revision 1.54  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.53  1998/12/23 10:54:37  myshkin
// Added code to _GetRawPPCIdentification to read /proc/cpuinfo on linux-ppc.
// Added *ppc-gcc272 entry to configure (until I successfully upgrade to egcs).
//
// Revision 1.52  1998/12/22 15:58:24  jcmichot
// QNX change: _x86ident vs x86ident.
//
// Revision 1.51  1998/12/14 05:15:08  dicamillo
// Mac OS updates to eliminate use of MULTITHREAD and have a singe client
// for MT and non-MT machines.
//
// Revision 1.50  1998/12/09 07:41:56  dicamillo
// fixed log comment.
//
// Revision 1.49  1998/12/09 07:34:15  dicamillo
// Added constant for number of processors Mac client supports; fixed typos
// which prevented compilation.
//
// Revision 1.48  1998/12/04 12:09:01  chrisb
// fixed typo
//
// Revision 1.47  1998/12/04 16:44:30  cyp
// Noticed and fixed MacOS's returning raw cpu type numbers to SelectCore().
// Fixed a long description header in ProcessorIndentification stuff. Tried
// to make cpu detection stuff more cpu- and less os- centric.
//
// Revision 1.46  1998/12/01 19:49:14  cyp
// Cleaned up MULT1THREAD #define. See cputypes.h log entry for details.
//
// Revision 1.45  1998/12/01 11:24:11  chrisb
// more riscos x86 changes
//
// Revision 1.44  1998/11/25 09:23:32  chrisb
// various changes to support x86 coprocessor under RISC OS
//
// Revision 1.43  1998/11/08 00:48:35  sampo
// fix a few typos to enable it to compile.
//
// Revision 1.42  1998/11/06 18:50:39  cyp
// Cleaned up tabs and macos cpu detection logic.
//
// Revision 1.41  1998/11/04 22:55:09  dicamillo
// change Mac cpu_description text to be static
//
// Revision 1.40  1998/11/04 22:35:42  dicamillo
// additional updates to Mac CPU detection code
//
// Revision 1.39  1998/11/04 20:02:42  sampo
// Fix for cpu detection on macs with a 601 processor upgraded to 750.
//
// Revision 1.38  1998/11/03 22:26:46  remi
// Attempt to auto-detect the number of cpus on Linux/Sparc machines.
//
// Revision 1.37  1998/11/01 01:12:54  sampo
// Added MacOS 68k detection stuff
//
// Revision 1.36  1998/10/31 21:53:55  silby
// Fixed a typo from previous commit.
//
// Revision 1.35  1998/10/30 19:43:39  sampo
// Added MacOS PowerPC detection stuff
//
// Revision 1.34  1998/10/30 00:07:19  foxyloxy
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
return "@(#)$Id: cpucheck-conflict.cpp,v 1.74 1999/04/02 20:56:41 jlawson Exp $"; }
#endif

#include "cputypes.h"
#include "baseincs.h"  // for platform specific header files
#include "cpucheck.h"  //just to keep the prototypes in sync.
#include "logstuff.h"  //LogScreen()/LogScreenRaw()

#if (CLIENT_OS == OS_SOLARIS)
#include <unistd.h>    // cramer - sysconf()
#elif (CLIENT_OS == OS_IRIX)
#include <sys/prctl.h>
#elif (CLIENT_OS == OS_DEC_UNIX)
#include <unistd.h>
#include <sys/sysinfo.h>
#include <machine/hal_sysinfo.h>
#include <machine/cpuconf.h>
#endif

// --------------------------------------------------------------------------

unsigned int GetNumberOfSupportedProcessors( void )
{
#if (CLIENT_OS == OS_RISCOS)
  return ( 2 ); /* not just some arbitrary number */
#elif (CLIENT_OS == OS_MACOS)
  if (haveMP)
    return( MAC_MAXCPUS );
  return(1);
#else
  return ( 128 ); /* just some arbitrary number */
#endif
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
        cpucount = -1;
    }
    #elif (CLIENT_OS == OS_NETWARE)
    {
      cpucount = nwCliGetNumberOfProcessors();
    }
    #elif (CLIENT_OS == OS_OS2)
    {
      int rc = (int) DosQuerySysInfo(QSV_NUMPROCESSORS, QSV_NUMPROCESSORS,
                  &cpucount, sizeof(cpucount));
      if (rc != 0 || cpucount < 1)
        cpucount = -1;
    }
    #elif (CLIENT_OS == OS_LINUX)
    {
      FILE *cpuinfo = fopen("/proc/cpuinfo", "r");
      cpucount = 0;
      if ( cpuinfo )
      {
        char buffer[256];
        while(fgets(buffer, sizeof(buffer), cpuinfo))
        {
          buffer[sizeof(buffer) - 1] = '\0';
          #if (CLIENT_CPU == CPU_X86 || CLIENT_CPU == CPU_POWERPC)
          if (strstr(buffer, "processor") == buffer)
            cpucount++;
          #elif (CLIENT_CPU == CPU_SPARC)
          // 2.1.x kernels (at least 2.1.125)
          if (strstr( buffer, "ncpus active\t: " ) == buffer)
            cpucount = atoi( buffer+15 );
          // 2.0.x non-smp kernels (at least 2.0.35)
          else if (strstr( buffer, "BogoMips\t: " ) == buffer)
            cpucount = 1;
          // 2.0.x smp kernels (at least 2.0.35)
          else if ( buffer == strstr( buffer, 
                    "        CPU0\t\tCPU1\t\tCPU2\t\tCPU3\n" ) )
          {
            fgets( buffer, 256, cpuinfo );
            for (char *p = strtok( buffer+7, "\t \n" ); p;
                    p = strtok( NULL, "\t \n" ))
            {
              if (strstr( p, "online" ) || strstr( p, "akp"))
                cpucount++;
            }
          }
          #elif (CLIENT_CPU == CPU_ALPHA)
          cpucount = 1; /* assume this until we know otherwise */
          /* SMP data (2.1+) - "CPUs probed %d active %d map 0x%x AKP %d\n" */
          if (memcmp(buffer, "CPUs probed", 11) == 0 && 
                    (buffer[11] == '\t' || buffer[11] == ' '))
          {
            char *p = strstr( buffer, "active" );
            if (p && (p[6] == '\t' || p[6] == ' '))
            {
              p += 6; while (*p && !isdigit(*p)) p++;
              cpucount = atoi(p);
              break;
            }
          }
          #else
          cpucount = -1;
          break;
          #endif
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
    #elif (CLIENT_OS == OS_AIX)
    {
      cpucount = sysconf(_SC_NPROCESSORS_ONLN);
    }
    #elif (CLIENT_OS == OS_RISCOS)
    {
      cpucount = riscos_count_cpus();
    }
    #elif (CLIENT_OS == OS_QNX)
    {
      cpucount = 1;
    }
    #elif (CLIENT_OS == OS_MACOS)
    {
      cpucount = 1;
      if (haveMP)
        cpucount = MPProcessors();
    }
    #elif ( (CLIENT_OS == OS_DEC_UNIX) && defined(OS_SUPPORTS_SMP))
    {
      // We really only want to do this for multithreaded clients.
      // Earlier versions of the Digital Unix don't support this.
      int status = 0;
      struct cpu_info buf;
      int st = 0;
      status = getsysinfo(GSI_CPU_INFO, (char *) &buf, sizeof(buf), st, NULL,NULL);
      if (status == -1)
        cpucount = -1;
      else
        cpucount = buf.cpus_in_box;
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
  // numcpu with zero implies "force-single-threaded"
  //
  // Whether a CLIENT_CPU (core) is thread safe or not, or whether the client
  // was built with thread support or not, is not the responsibility of
  // select cpu logic.
  //--------------------

  if (detected_count == -2)  // returns -1 if no hardware detection
    detected_count = GetNumberOfDetectedProcessors(); 

  if (numcpu < 0)                //numcpu == 0 implies force non-mt;
  {                              //numcpu < 0  implies autodetect
    if ( detected_count < 1 )
    {
      if (!quietly)
        LogScreen( CLIENT_OS_NAME " does not support SMP or\n"
                  "does not support processor count detection.\n"
                  /* "Automatic processor count detection failed.\n" */
                  "A single processor machine is assumed.\n");
      detected_count = 1;
    }
    else if (!quietly)
    {
      LogScreen("Automatic processor detection found %d processor%s.\n",
                detected_count, ((detected_count==1)?(""):("s")) );
    }
    numcpu = detected_count;
    if (numcpu < 0) //zero is legal (implies force-single-threaded)
      numcpu = 0;
  }

  if (((unsigned int)(numcpu)) > GetNumberOfSupportedProcessors())
    numcpu = (int)GetNumberOfSupportedProcessors();

  return (unsigned int)numcpu;
}

// --------------------------------------------------------------------------

#if (CLIENT_CPU == CPU_68K)
static long __GetRawProcessorID(const char **cpuname)
{
  static long detectedtype = -2; /* -1==failed, -2==not supported */
  static const char *detectedname = NULL;
  static char namebuf[20];
  static struct { const char *name; long rid; } cpuridtable[] = {
                { "68000",           68000L  },
                { "68010",           68010L  },
                { "68020",           68020L  },
                { "68030",           68030L  },
                { "68040",           68040L  },
                { "68060",           68060L  }
                };

  #if (CLIENT_OS == OS_AMIGAOS)
  if (detectedtype == -2)
  {
    long flags = (long)(SysBase->AttnFlags);

    if ((flags & AFF_68060) && (flags & AFF_68040) && (flags & AFF_68030))
      detectedtype = 68060L; // Phase5 060 boards at least report this...
    else if ((flags & AFF_68040) && (flags & AFF_68030))
      detectedtype = 68040L; // 68040
    else if ((flags & AFF_68030) && (flags & AFF_68020))
      detectedtype = 68030L; // 68030
    else if ((flags & AFF_68020) && (flags & AFF_68010))
      detectedtype = 68020L; // 68020
    else if (flags & AFF_68010)
      detectedtype = 68010L; // 68010
    else
      detectedtype = -1;
  }
  #elif (CLIENT_OS == OS_MACOS)
  if (detectedtype == -2)
  {
    long result;
    // Note: gestaltProcessorType is used so that if the 68K client is run on
    // on PPC machine for test purposes, it will get the type of the 68K
    // emulator. Also, gestaltNativeCPUType is not present in early versions of
    // System 7. (For more info, see Gestalt.h)
    detectedtype = -1L;
    if (Gestalt(gestaltProcessorType, &result) == noErr)
    {
      switch(result)
      {
        case gestalt68000: detectedtype = 68000L;  break;
        case gestalt68010: detectedtype = 68010L;  break;
        case gestalt68020: detectedtype = 68020L;  break;
        case gestalt68030: detectedtype = 68030L;  break;
        case gestalt68040: detectedtype = 68040L;  break;
        default:           detectedtype = -1L;     break;
      }
    }
  }
  #elif (CLIENT_OS == OS_LINUX)
  if (detectedtype == -2L)
  {
    FILE *cpuinfo;
    detectedtype = -1L;
    if ((cpuinfo = fopen( "/proc/cpuinfo", "r")) != NULL)
    {
      char buffer[256];
      while(fgets(buffer, sizeof(buffer), cpuinfo)) 
      {
        const char *p = "CPU:\t\t";
        unsigned int n = strlen( p );
        if ( memcmp( buffer, p, n ) == 0 )
        {
          p = &buffer[n]; buffer[sizeof(buffer)-1]='\0';
          for ( n = 0; n < (sizeof(cpuridtable)/sizeof(cpuridtable[0])); n++ )
          {
            unsigned int l = strlen( cpuridtable[n].name );
            if ((!p[l] || isspace(p[l])) && memcmp(p,cpuridtable[n].name,l)==0)
            {
              detectedtype = cpuridtable[n].rid;
              break;
            }
          }
          if (detectedtype == -1L)
          {
            for ( n = 0; *p && *p!='\r' && *p!='\n' && n<(sizeof(namebuf)-1); n++ )
              namebuf[n] = *p++;
            namebuf[n] = '\0';
            detectedname = (const char *)&namebuf[0];
            detectedtype = 0;
          }
          break;
        }
      }
      fclose(cpuinfo);
    }
  }
  #endif

  if (detectedtype > 0 && detectedname == NULL )
  {
    unsigned int n;
    detectedname = "";
    for (n = 0; n < (sizeof(cpuridtable)/sizeof(cpuridtable[0])); n++ )
    {
      if (((long)(cpuridtable[n].rid)) == detectedtype )
      {
        strcpy( namebuf "Motorola " );
        strcat( namebuf, cpuridtable[n].name );
        detectedname = (const char *)&namebuf[0];
        break;
      }
    }
  }

  if (cpuname)
    *cpuname = detectedname;
  return detectedtype;
}

#endif /* (CLIENT_CPU == CPU_68K) */

// --------------------------------------------------------------------------

#if (CLIENT_CPU == CPU_POWERPC)
static long __GetRawProcessorID(const char **cpuname)
{
  /* ******* detected type reference is (PVR value >> 16) *********** */
  static long detectedtype = -2L; /* -1 == failed, -2 == not supported */
  static const char *detectedname = NULL;
  static char namebuf[30];
  static struct { int rid; const char *name; } cpuridtable[] = {
                {       1, "601"             },
                {       3, "603"             },
                {       4, "604"             },
                {       6, "603e"            },
                {       7, "603ev"           },
                {       8, "740/750/G3"      },
                {       9, "604e"            },
                {      10, "604ev/Mach5"     }
                };

  #if (CLIENT_OS == OS_MACOS)
  if (detectedtype == -2L)
  {
    // Note: need to use gestaltNativeCPUtype in order to get the correct
    // value for G3 upgrade cards in a 601 machine.
    long result;
    detectedtype = -1;
    if (Mac_PPC_prototype) 
      detectedtype = 1L;  // old developer machines - 601
    else if (Gestalt(gestaltNativeCPUtype, &result) == noErr)
      detectedtype = result - 0x100L;
  }
  #elif (CLIENT_OS == OS_LINUX)
  if (detectedtype == -2L)
  {
    FILE *cpuinfo;
    detectedtype = -1L;
    if ((cpuinfo = fopen( "/proc/cpuinfo", "r")) != NULL)
    {
      char buffer[256];
      while(fgets(buffer, sizeof(buffer), cpuinfo)) 
      {
        const char *p = "cpu\t\t: ";
        unsigned int n = strlen( p );
        if ( memcmp( buffer, p, n ) == 0 )
        {
          static struct 
           { const char *sig;  int rid; } sigs[] = {
           { "601",                  1  },
           { "603",                  3  },
           { "604",                  4  },
           { "603e",                 6  },
           { "603ev",                7  },
           { "750 (Arthur)",         8  },
           { "604e",                 9  },
           { "604ev5 (MachV)",      10  }
           };
          p = &buffer[n]; buffer[sizeof(buffer)-1]='\0';
          for ( n = 0; n < (sizeof(sigs)/sizeof(sigs[0])); n++ )
          {
            unsigned int l = strlen( sigs[n].sig );
            if ((!p[l] || isspace(p[l])) && memcmp( p, sigs[n].sig, l)==0)
            {
              detectedtype = (long)sigs[n].rid;
              break;
            }
          }
          if (detectedtype == -1L)
          {
            for ( n = 0; *p && *p!='\n' && *p!='\r' && n<(sizeof(namebuf)-1); n++ )
              namebuf[n] = *p++;
            namebuf[n] = '\0';
            detectedname = (const char *)&namebuf[0];
            detectedtype = 0;
          }
          break;
        }
      }
      fclose(cpuinfo);
    }
  }
  #endif
  
  if (detectedtype > 0 && detectedname == NULL )
  {
    unsigned int n;
    detectedname = "";
    for (n = 0; n < (sizeof(cpuridtable)/sizeof(cpuridtable[0])); n++ )
    {
      if (((long)(cpuridtable[n].rid)) == detectedtype )
      {
        strcpy( namebuf "PowerPC " );
        strcat( namebuf, cpuridtable[n].name );
        detectedname = (const char *)&namebuf[0];
        break;
      }
    }
  }
  
  if (cpuname)
    *cpuname = detectedname;
  return detectedtype;
}
#endif /* (CLIENT_CPU == CPU_POWERPC) */

// --------------------------------------------------------------------------

#if (CLIENT_CPU == CPU_X86)

#if (defined(__WATCOMC__) || (CLIENT_OS == OS_QNX)) 
  #define x86ident _x86ident
#endif
#if (CLIENT_OS == OS_LINUX) && !defined(__ELF__)
  extern "C" u32 x86ident( void ) asm ("x86ident");
#else
  extern "C" u32 x86ident( void );
#endif

static long __GetRawProcessorID(const char **cpuname, int whattoret = 0 )
{
  static long detectedtype = -2L;
  static const char *detectedname = NULL;
  static int  kKeysPerMhz = 512; /* default rate if not found */
  static int  coretouse   = 0;   /* default core if not found */
  
  if ( detectedtype == -2L )
  {
    static char namebuf[30];
    const char *vendorname = NULL;
    struct cpuxref { int cpuid, kKeysPerMhz, coretouse; 
                     const char *cpuname; } *internalxref = NULL;
    u32 dettype     = x86ident();
    int cpuidbmask  = 0xfff0; /* mask with this to find it in the table */
    int cpuid       = (((int)dettype) & 0xffff);
    int vendorid    = (((int)(dettype >> 16)) & 0xffff);
  
    sprintf( namebuf, "%04X:%04X", vendorid, cpuid );
    detectedname = (const char *)&namebuf[0];
    detectedtype = 0; /* assume not found */
  
    if ( vendorid == 0x7943 ) // Cyrix CPU
    {
      static struct cpuxref cyrixxref[]={
          {    0x40, 0512,     0, "486"       }, // use Pentium core
          {  0x0440, 0512,     0, "MediaGX" },
          {  0x0490, 1185,     0, "5x86"      },
          {  0x0520, 2090,     3, "6x86"      }, // "Cyrix 6x86/6x86MX/M2"
          {  0x0540, 1200,     0, "GXm"       }, // use Pentium core here too
          {  0x0600, 2115, 0x103, "6x86MX"    },
          {  0x0000, 2115,     3, NULL        } //default core == 6x86
          }; internalxref = &cyrixxref[0];
      vendorname = "Cyrix ";
      cpuidbmask = 0xfff0; //strip last 4 bits, don't need stepping info
    }
    else if ( vendorid == 0x6543 ) //centaur/IDT cpu
    {
      static struct cpuxref centaurxref[]={
          {  0x0540, 1200,0x100, "C6"          }, // use Pentium core
          {  0x0585, 1346,0x102, "WinChip 2"   }, // pentium Pro (I think)
          {  0x0000, 1346,    0, NULL          }  // default core == Pentium
          }; internalxref = &centaurxref[0];
      vendorname = "Centaur/IDT ";
      cpuidbmask = 0xfff0; //strip last 4 bits, don't need stepping info
    }
    else if ( vendorid == 0x7541 ) // AMD CPU
    {
      static struct cpuxref amdxref[]={
          {  0x0040, 0512,     0, "486"        },   // "Pentium, Pentium MMX, Cyrix 486/5x86/MediaGX, AMD 486",
          {  0x0430, 0512,     0, "486DX2"     },
          {  0x0470, 0512,     0, "486DX2WB" },
          {  0x0480, 0512,     0, "486DX4"     },
          {  0x0490, 0512,     0, "486DX4WB" },
          {  0x04E0, 1185,     0, "5x86"       },
          {  0x04F0, 1185,     0, "5x86WB"     },
          {  0x0500, 2353,     4, "K5 PR75, PR90, or PR100" }, // use K5 core
          {  0x0510, 2353,     4, "K5 PR120 or PR133" },
          {  0x0520, 2353,     4, "K5 PR166" },
          {  0x0530, 2353,     4, "K5 PR200" },
          {  0x0560, 1611, 0x105, "K6"         },
          {  0x0570, 1611, 0x105, "K6"         },
          {  0x0580, 1690, 0x105, "K6-2"       },
          {  0x0590, 1690, 0x105, "K6-3"       },
          {  0x0000, 1690,     5, NULL         }   // for the future - default core = K6
          }; internalxref = &amdxref[0];
      vendorname = "AMD ";
      cpuidbmask = 0xfff0; //strip last 4 bits, don't need stepping info
    }
    else if ( vendorid == 0x6E49 || vendorid == 0x6547 ) // Intel CPU
    {
      static struct cpuxref intelxref[]={
          {  0x0030, 0426,     1, "80386"      },   // generic 386/486 core
          {  0x0040, 0512,     1, "80486"      },
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
          {  0x0630, 3092, 0x102, "Pentium II" },
          {  0x0650, 3092, 0x102, "Pentium II" },
          {  0x0660, 3092, 0x102, "Celeron-A" },
          {  0x0670, 4096, 0x102, "Pentium III" },
          {  0x0000, 4096,     2, NULL           }  // default core = PPro/PII
          }; internalxref = &intelxref[0];
      vendorname = "Intel "; 
      if ((cpuid == 0x30) || (cpuid == 0x40))
        vendorname = ""; //not for generic 386/486
      cpuidbmask = 0xfff0; //strip last 4 bits, don't need stepping info
    }
  
    if (internalxref != NULL) /* we know about this vendor */
    {
      unsigned int pos;
      int maskedid = ( cpuid & cpuidbmask );
  
      for (pos = 0; ; pos++ )
      {
        if ((internalxref[pos].cpuname == NULL) || /* bonk! hit bottom */
           (maskedid == (internalxref[pos].cpuid & cpuidbmask))) /* found it */
        {
          kKeysPerMhz  = internalxref[pos].kKeysPerMhz;
          coretouse    = internalxref[pos].coretouse;
          detectedtype = dettype;
          if ( internalxref[pos].cpuname )
          {
            strcpy( namebuf, vendorname );
            strcat( namebuf, internalxref[pos].cpuname );
            detectedname = (const char *)&namebuf[0];
          }
          break;
        }
      }
    }
  }
  
  if (cpuname)
    *cpuname = detectedname;
  if (whattoret == 'c')
    return ((long)coretouse);
  if (whattoret == 'k')
    return ((long)kKeysPerMhz);
  return detectedtype;
}
#endif /* X86 */

// --------------------------------------------------------------------------

#if (CLIENT_OS == OS_RISCOS)
#include <setjmp.h>
static jmp_buf ARMident_jmpbuf;
static void ARMident_catcher(int)
{
  longjmp(ARMident_jmpbuf, 1);
}
#endif

#if (CLIENT_CPU == CPU_ARM)
static long __GetRawProcessorID(const char **cpuname )
{
  static long detectedtype = -2L; /* -1 == failed, -2 == not supported */
  static const char *detectedname = NULL;

  #if (CLIENT_OS == OS_RISCOS)
  if ( detectedvalue == -2L )
  {
    static char namebuf[60];
    // ARMident() will throw SIGILL on an ARM 2 or ARM 250, because
    // they don't have the system control coprocessor. (We ignore the
    // ARM 1 because I'm not aware of any existing C++ compiler that
    // targets it...)
    signal(SIGILL, ARMident_catcher);
    if (setjmp(ARMident_jmpbuf))
      detectedvalue = 0x2000;
    else
      detectedvalue = ARMident();
    signal(SIGILL, SIG_DFL);
    detectedvalue = (detectedvalue >> 4) & 0xfff; // extract part number field

    if ((detectedvalue & 0xf00) == 0) //old-style ID (ARM 3 or prototype ARM 600)
      detectedvalue <<= 4;            // - shift it into the new form
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
        detectedvalue = 0x7500FEL;
    }
    detectedname = ((const char *)&(namebuf[0]));
    switch (detectedvalue)
    {
      case 0x200: strcpy( namebuf, "ARM 2 or 250" ); 
                  break;
      case 0xA10: strcpy( namebuf, "StrongARM 110" ); 
                  break;
      case 0x3:
      case 0x600:
      case 0x610:
      case 0x700:
      case 0x7500:
      case 0x7500FEL:
      case 0x710:
      case 0x810: sprintf( namebuf, "ARM %lX", detectedvalue );
                  break;
      default:    sprintf( namebuf, "%lX", detectedvalue );
                  detectedvalue = 0;
                  break;
    }
  }
  #endif /* RISCOS */

  if ( cpuname )
    *cpuname = detectedname;
  return detectedvalue;
}  
#endif

// --------------------------------------------------------------------------

#if (CLIENT_CPU == CPU_MIPS)
static long __GetRawProcessorID(const char **cpuname)
{
  static int detectedtype = -2L; /* -1 == failed, -2 == not supported */
  static const char *detectedname = NULL;
  static char namebuf[30];
  static struct { const char *name; int rid; } cpuridtable {
                { "R2000"         ,       1  },
                { "R3000"         ,       2  },
                { "R3000A"        ,       3  },
                { "R3041"         ,       4  },
                { "R3051"         ,       5  },
                { "R3052"         ,       6  },
                { "R3081"         ,       7  },
                { "R3081E"        ,       8  },
                { "R4000PC"       ,       9  },
                { "R4000SC"       ,      10  },
                { "R4000MC"       ,      11  },
                { "R4200"         ,      12  },
                { "R4400PC"       ,      13  },
                { "R4400SC"       ,      14  },
                { "R4400MC"       ,      15  },
                { "R4600"         ,      16  },
                { "R6000"         ,      17  },
                { "R6000A"        ,      18  },
                { "R8000"         ,      19  },
                { "R10000"        ,      20  }
                };
  
  #if (CLIENT_OS == OS_LINUX)
  if (detectedtype == -2L)
  {
    FILE *cpuinfo;
    detectedtype = -1L;
    if ((cpuinfo = fopen( "/proc/cpuinfo", "r")) != NULL)
    {
      char buffer[256];
      while(fgets(buffer, sizeof(buffer), cpuinfo)) 
      {
        const char *p = "cpu model\t\t: ";
        unsigned int n = strlen( p );
        if ( memcmp( buffer, p, n ) == 0 )
        {
          p = &buffer[n]; buffer[sizeof(buffer)-1]='\0';
          for ( n = 0; n < (sizeof(cpuridtable)/sizeof(cpuridtable[0])); n++ )
          {
            unsigned int l = strlen( cpuridtable[n].sig );
            if ((!p[l] || isspace(p[l])) && memcmp( p, cpuridtable[n].sig, l)==0)
            {
              detectedtype = (long)cpuridtable[n].rid;
              break;
            }
          }
          if (detectedtype == -1L)
          {
            for ( n = 0; *p && *p!='\r' && *p!='\n' && n<(sizeof(namebuf)-1); n++ ) 
              namebuf[n] = *p++;
            namebuf[n] = '\0';
            detectedname = (const char *)&namebuf[0];
            detectedtype = 0;
          }
          break;
        }
      }
      fclose(cpuinfo);
    }
  }
  #endif

  if (detectedtype > 0 && detectedname == NULL)
  {
    unsigned int n;
    detectedname = "";
    for ( n = 0; n < (sizeof(mips_chips)/sizeof(mips_chips[0])); n++ )
    {
      if (detectedtype == mips_chips[n].rid )
      {
        strcpy( namebuf, "MIPS " );
        strcat( namebuf, mips_chips[n].name );
        detectedname = (const char *)&namebuf[0];
        break;
      }
    }
  }
  if (cpuname)
    *cpuname = detectedname;
  return detectedtype;
}
#endif

// --------------------------------------------------------------------------

#if (CLIENT_CPU == CPU_SPARC)
static long __GetRawProcessorID(const char **cpuname)
{
  /* detectedtype reference is (0x100 + ((get_psr() >> 24) & 0xff)) */
  static long detectedtype = -2L; /* -1 == failed, -2 == not supported */
  static const char *detectedname = NULL;
  
  /* from linux/arch/sparc/kernel/cpu.c */
  static struct { int psr_impl, psr_vers; const char *name; } cpuridtable[] = {
  { 0, 0, "Fujitsu  MB86900/1A or LSI L64831 SparcKIT-40"}, /* Sun4/100, 4/200, SLC */
  { 0, 4, "Fujitsu  MB86904"}, /* born STP1012PGA */
  { 1, 0, "LSI Logic Corporation - L64811"}, /* SparcStation2, SparcServer 490 & 690 */
  { 1, 1, "Cypress/ROSS CY7C601"}, /* SparcStation2 */
  { 1, 3, "Cypress/ROSS CY7C611"}, /* Embedded controller */
  { 1, 0xf, "ROSS HyperSparc RT620"}, /* Ross Technologies HyperSparc */
  { 1, 0xe, "ROSS HyperSparc RT625"},
  /* ECL Implementation, CRAY S-MP Supercomputer... AIEEE! */
  /* Someone please write the code to support this beast! ;) */
  { 2, 0, "Bipolar Integrated Technology - B5010"}, /* Cray S-MP */
  { 3, 0, "LSI Logic Corporation - unknown-type"},
  { 4, 0, "Texas Instruments, Inc. - SuperSparc 50"},
  { 4, 1, "Texas Instruments, Inc. - MicroSparc"}, /* SparcClassic STP1010TAB-50*/
  { 4, 2, "Texas Instruments, Inc. - MicroSparc II"},
  { 4, 3, "Texas Instruments, Inc. - SuperSparc 51"},
  { 4, 4, "Texas Instruments, Inc. - SuperSparc 61"},
  { 4, 5, "Texas Instruments, Inc. - unknown"},
  { 5, 0, "Matsushita - MN10501"},
  { 6, 0, "Philips Corporation - unknown"},
  { 7, 0, "Harvest VLSI Design Center, Inc. - unknown"},
  { 8, 0, "Systems and Processes Engineering Corporation (SPEC)"}, /* Gallium arsenide 200MHz, BOOOOGOOOOMIPS!!! */
  { 9, 0, "Fujitsu #3"},
  };
  
  #if (CLIENT_OS == OS_LINUX)
  if (detectedtype == -2L)
  {
    FILE *cpuinfo;
    detectedtype = -1L;
    if ((cpuinfo = fopen( "/proc/cpuinfo", "r")) != NULL)
    {
      char buffer[256];
      while(fgets(buffer, sizeof(buffer), cpuinfo)) 
      {
        const char *p = "cpu model\t\t: ";
        unsigned int n = strlen( p );
        if ( memcmp( buffer, p, n ) == 0 )
        {
          static char namebuf[128];
          p = &buffer[n]; buffer[sizeof(buffer)-1]='\0';
          for ( n = 0; n < (sizeof(cpuridtable)/sizeof(cpuridtable[0])); n++ )
          {
            unsigned int l = strlen( cpuridtable[n].name );
            if ((!p[l] || isspace(p[l])) && memcmp( p, cpuridtable[n].name, l)==0)
            {
              detectedtype = 0x100 | ((cpuridtable[n].psr_impl & 0x0f)<<4) |
                             (cpuridtable[n].psr_vers & 0x0f);
              detectedname = cpuridtable[n].name;
              break;
            }
          }
          if (detectedtype == -1) /* found name but no ID */
          {
            for ( n = 0; *p && *p!='\r' && *p!='\n' && n<(sizeof(namebuf)-1); n++ )
              namebuf[n] = *p++;
            namebuf[n] = '\0';
            detectedname = (const char *)&namebuf[0];
            detectedtype = 0;
          }
          break;
        }
      }
      fclose(cpuinfo);
    }
  }
  #endif
  
  if (detectedtype >= 0x100 && detectedname == NULL)
  {
    int psr_impl = ((detectedtype>>4)&0xf);
    int psr_vers = ((detectedtype   )&0xf);
    unsigned int n;
    for (n = 0; n < (sizeof(cpuridtable)/sizeof(cpuridtable[0])); n++ )
    {
      if (( psr_impl == cpuridtable[n].psr_impl ) &&
          ( psr_vers == cpuridtable[n].psr_vers ))
      {
        detectedname = cpuridtable[n].name;
        break;
      }
    }
  }   
  if (cpuname)
    *cpuname = detectedname;
  return detectedtype;
}
#endif

// --------------------------------------------------------------------------

#if (CLIENT_CPU == CPU_ALPHA)
static long __GetRawProcessorID(const char **cpuname)
{
  static char namebuf[30];
  static int detectedtype = -2L; /* -1 == failed, -2 == not supported */
  static const char *detectedname = NULL;
  static struct { int rid; const char *name;     } cpuridtable[] = {
                {       1, "EV3"                 },
                {       2, "EV4 (21064)"         },
                {       4, "LCA4 (21066/21068)"  },
                {       5, "EV5 (21164)"         },
                {       6, "EV4.5 (21064)"       },
                {       7, "EV5.6 (21164A)"      },
                {       8, "EV6 (21264)",        },
                {       9, "EV5.6 (21164PC)"     },
                {      10, "EV5.7"               }
                };
  
  #if (CLIENT_OS == OS_DEC_UNIX)
  if (detectedtype == -2)
  {
    long buf;
    int st = 0;
    detectedtype = -1;
    if (getsysinfo(GSI_PROC_TYPE,(char *)&buf,sizeof(buf),st,NULL,NULL)!=-1)
      detectedtype = (buf & 0xffff);
    if (detectedtype <= 0)
      detectedtype = -1;
  }
  #elif (CLIENT_OS == OS_LINUX)
  if (detectedtype == -2L)
  {
    FILE *cpuinfo;
    detectedtype = -1L;
    if ((cpuinfo = fopen( "/proc/cpuinfo", "r")) != NULL)
    {
      char buffer[256];
      while(fgets(buffer, sizeof(buffer), cpuinfo)) 
      {
        const char *p = "cpu model\t\t: ";
        unsigned int n = strlen( p );
        if ( memcmp( buffer, p, n ) == 0 )
        {
          static struct 
           { const char *sig;  int rid; } sigs[] = {
           { "EV3",            1      },
           { "EV4",            2      },
           { "Unknown 1",      3      }, /* 2.0.x kernel */
           { "Unknown",        3      }, /* 2.2.x kernel */
           { "LCA4",           4      },
           { "EV5",            5      },
           { "EV45",           6      },
           { "EV56",           7      },
           { "EV6",            8      },
           { "PCA56",          9      },
           { "PCA57",         10      }, /* 2.2.x kernel */
           };
          p = &buffer[n]; buffer[sizeof(buffer)-1]='\0';
          for ( n = 0; n < (sizeof(sigs)/sizeof(sigs[0])); n++ )
          {
            unsigned int l = strlen( sigs[n].sig );
            if ((!p[l] || isspace(p[l])) && memcmp( p, sigs[n].sig, l)==0)
            {
              detectedtype = (long)sigs[n].rid;
              break;
            }
          }
          if (detectedtype == -1L)
          {
            for ( n = 0; *p && *p!='\r' && *p!='\n' && n<(sizeof(namebuf)-1); n++ )  
              namebuf[n] = *p++;
            namebuf[n] = '\0';
            detectedname = (const char *)&namebuf[0];
            detectedtype = 0;
          }
          break;
        }
      }
      fclose(cpuinfo);
    }
  }
  #endif
  
  if (detectedtype > 0 && detectedname == NULL)
  {
    unsigned int n;
    detectedname = "";
    for ( n = 0; n < (sizeof(cpuridtable)/sizeof(cpuridtable[0])); n++ )
    {
      if (detectedtype == cpuridtable[n].rid )
      {
        strcpy( namebuf, "Alpha " );
        strcat( namebuf, cpuridtable[n].name );
        detectedname = (const char *)&namebuf[0];
        break;
      }
    }
  }
  
  if (cpuname)
    *cpuname = detectedname;
  return detectedtype;
}
#endif

// --------------------------------------------------------------------------

int GetProcessorType(int quietly)
{
  int coretouse = -1;
  const char *apd = "Automatic processor type detection ";
  #if (CLIENT_CPU == CPU_ALPHA)   || (CLIENT_CPU == CPU_68K) || \
      (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_X86) || \
      (CLIENT_CPU == CPU_ARM)     || (CLIENT_CPU == CPU_MIPS) || \
      (CLIENT_CPU == CPU_SPARC)
  {
    const char *cpuname = NULL;
    long rawid = __GetRawProcessorID(&cpuname);
    if (!quietly)
    {
      if (rawid < 0)
        LogScreen("%s%s.\n", apd, ((rawid == -1L)?("failed"):("is not supported")));
      else if (rawid == 0)
        LogScreen("%sdid not\nrecognized the processor (tag: %s)\n", apd, (cpuname?cpuname:"???") );
      else if (cpuname == NULL || *cpuname == '\0')
        LogScreen("%sdid not\nrecognized the processor (id: %ld)\n", apd, rawid );
      else
        LogScreen("%sfound\na%s %s processor.\n",apd, 
           ((strchr("aeiou8", tolower(*cpuname)))?("n"):("")), cpuname);
    }
    #if (CLIENT_CPU == CPU_68K)
    if ((coretouse = ((rawid <= 0) ? (-1) : (((int)(rawid-68010L))/10))) == 6)
      coretouse = 5; /* remap 68060 to 68050 */
    #elif (CLIENT_CPU == CPU_ALPHA)
    coretouse = ((rawid <= 0) ? (-1) : ((int)rawid));
    #elif (CLIENT_CPU == CPU_POWERPC)
    coretouse = ((rawid < 0) ? (-1) : ((rawid==1L)?(0/*601*/):(1)));
    #elif (CLIENT_CPU == CPU_X86) /* way too many cpu<->core combinations */
    if (( rawid = __GetRawProcessorID(NULL,'c')) >= 0) coretouse = (int)rawid;
    #elif (CLIENT_CPU == CPU_ARM)
    if (rawid <= 0)                                coretouse =-1;
    else if (rawid == 0x3    || rawid == 0x600 ||
             rawid == 0x610  || rawid == 0x700 ||
             rawid == 0x7500 || rawid == 0x7500FE) coretouse = 0;
    else if (rawid == 0x810  || rawid == 0xA10)    coretouse = 1;
    else if (rawid == 0x200)                       coretouse = 2;
    else if (rawid == 0x710)                       coretouse = 3;
    #endif
  }
  #else
  {
    if (!quietly)
      LogScreen("%sis not supported.\n", apd );
    coretouse = -1;
  }
  #endif

  return (coretouse);
}

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
  return __GetRawProcessorID(NULL, 'k');
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

#if (CLIENT_CPU == CPU_ALPHA)   || (CLIENT_CPU == CPU_68K) || \
    (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_X86) || \
    (CLIENT_CPU == CPU_ARM)     || (CLIENT_CPU == CPU_MIPS) || \
    (CLIENT_CPU == CPU_SPARC)      
  long rawid = __GetRawProcessorID(&cpuid_s);
  if (rawid < 0)
    cpuid_s = ((rawid==-1)?("?\n\t(identification failed)"):
              ("none\n\t(client does not support identification)"));
  else
  {
    static char namebuf[60];
    if (cpuid_s == NULL) cpuid_s = "*unknown*";
    if (*cpuid_s =='\0') cpuid_s = "???";
  #if (CLIENT_CPU == CPU_ARM)
    namebuf[0] = '\0';
    if (rawid != 0) /* if rawid == 0, then cpuid_s == "%lX" */
      sprintf( namebuf, "%lX\n\tname: ", rawid );
    strcat( namebuf, cpuid_s ); /* always valid */
    #if (CLIENT_OS == OS_RISCOS)
    if (riscos_count_cpus() == 2)
      strcat(strcat(namebuf,"\n\t+ "),riscos_x86_determine_name());
    #endif
  #elif (CLIENT_CPU == CPU_X86)
    namebuf[0] = '\0';
    if (rawid != 0) /* if rawid == 0, then cpuid_s == "%04x:%04x" */
      sprintf( namebuf, "%04X:%04X\n\tname: ",(int)((rawid>>16)&0xffff),(int)(rawid&0xffff));
    strcat( namebuf, cpuid_s ); /* always valid */
  #else
    sprintf(namebuf, "%ld\n\tname: %s\n", rawid, cpuid_s );
  #endif
    cpuid_s = ((const char *)(&namebuf[0]));
  }    
#else    
  cpuid_s = "none\n\t(client does not support identification)";
#endif

  #if defined(CLIENT_SUPPORTS_SMP)
    static char maxcpu_b[80];
    sprintf( maxcpu_b, "%d", (int)GetNumberOfSupportedProcessors() );
    maxcpu_s = ((const char *)(&maxcpu_b[0]));
  #elif (!defined(CORES_SUPPORT_SMP))
    maxcpu_s = "1\n\t(cores are not thread-safe)";
  #elif (CLIENT_OS == OS_RISCOS)
    maxcpu_s = "2\n\t(with RiscPC x86 card)";
  #else
    maxcpu_s = "1\n\t(OS or client-build does not support threads)";
  #endif

  int cpucount = GetNumberOfDetectedProcessors();
  if (cpucount < 1)
    foundcpu_s = "1\n\t(OS does not support detection)";
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

  LogScreenRaw("Automatic processor identification tag: %s\n"
    "Number of processors detected by this client: %s\n"
    "Number of processors supported by this client: %s\n",
    scpuid, sfoundcpus, smaxscpus );
  return;
}

// --------------------------------------------------------------------------

